//! One Monte Carlo attempt and the MCS loop:
//!
//! ```text
//! pick a target site s uniformly at random from the interior
//! pick a source site s' uniformly from copy_neighborhood(s)
//! if sigma[s] == sigma[s']: return immediately            // early-out
//! if the copy is not a simple point for the losing cell: reject
//! if the copy would take the losing cell below volume 1: reject
//! compute delta_H locally
//! accept if delta_H <= 0, else accept with probability exp(-delta_H)
//! ```
//!
//! One MCS is `N_sites` attempts, counting early-outs, where `N_sites` is
//! the unpadded site count.

use crate::config::{AcceptanceMode, ProposalMode};
use crate::connectivity::preserves_topology;
use crate::dynamics;
use crate::energy::CopyContext;
use crate::lattice::CellId;
use crate::model::CPM;
use crate::state::index_of;
use rand::Rng as _;
use rand_distr::{Binomial, Distribution};

/// One attempt with a uniformly-drawn target site. Returns whether the
/// proposed copy was accepted. See [`attempt_at`] for the shared body (this
/// is just its `ProposalMode::Uniform` target-selection step split out).
///
/// `check_connectivity` is a plain parameter, not a `ResolvedConfig` field:
/// it isn't a real modelling convention, only a test-only escape hatch for
/// the exact Boltzmann validation, which must run with connectivity checking
/// disabled — the connectivity constraint is itself asymmetric (the forward
/// move tests cell A, the reverse tests cell B), which would break the
/// reversibility that validation depends on. Production code (`run_mcs`)
/// always passes `true`.
pub fn attempt<const D: usize>(cpm: &mut CPM<D>, check_connectivity: bool) -> bool {
    let dims = cpm.state.lattice.dims();
    let target_coord: [usize; D] = std::array::from_fn(|d| cpm.rng.gen_range(0..dims[d]));
    let target_flat = cpm.state.lattice.flat_index(target_coord);
    attempt_at(cpm, target_flat, check_connectivity)
}

/// One attempt at a specific, already-chosen target site — the shared body
/// both `attempt` (`ProposalMode::Uniform`, draws `target_flat` uniformly
/// from the interior) and `run_mcs`'s `ProposalMode::EdgeList` branch (draws
/// `target_flat` uniformly from `cpm.edge_list`) call into. Only the
/// *source* of `target_flat` differs between the two proposal modes; the
/// early-out, connectivity check, energy computation, acceptance decision
/// and commit are identical either way — a required property, since the
/// two modes must sample statistically equivalent chains.
pub fn attempt_at<const D: usize>(
    cpm: &mut CPM<D>,
    target_flat: usize,
    check_connectivity: bool,
) -> bool {
    let source_offset = cpm.copy_offsets[cpm.rng.gen_range(0..cpm.copy_offsets.len())];
    let source_flat = cpm.state.lattice.neighbour(target_flat, source_offset);

    let losing_id = cpm.state.lattice.get(target_flat);
    let gaining_id = cpm.state.lattice.get(source_flat);
    if losing_id == gaining_id {
        return false; // early-out, the common case
    }

    if check_connectivity
        && !preserves_topology(
            &cpm.state.lattice,
            target_flat,
            losing_id,
            &cpm.topology_offsets,
        )
    {
        return false;
    }

    if losing_id != 0 && cpm.state.volume[index_of(losing_id)] <= cpm.config.min_cell_volume {
        return false;
    }

    #[cfg(not(feature = "fused-energy"))]
    let (delta_h_conservative, delta_h) = {
        let ctx = CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            mcs: cpm.mcs,
        };
        let conservative = cpm.terms.total_delta(&ctx);
        // Act (if active) influences acceptance but must never reach
        // `State::conservative_energy` — see `Terms::total_delta`'s doc
        // comment and `commit_accepted` below, which only ever adds
        // `delta_h_conservative`, never the combined `delta_h`.
        (conservative, conservative + cpm.terms.act_delta(&ctx))
    };
    // Prices the move in one pass over the energy neighbourhood and keeps
    // the interface delta pair around for `commit_accepted`, instead of
    // pricing with `total_delta` and then recomputing that same pair on
    // acceptance.
    #[cfg(feature = "fused-energy")]
    let (delta_h_conservative, delta_h, fused_interface_deltas) = {
        let ctx = CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            mcs: cpm.mcs,
        };
        let priced = cpm.terms.fused_conservative_delta(&ctx);
        (
            priced.total,
            priced.total + cpm.terms.act_delta(&ctx),
            priced.interface_deltas,
        )
    };

    let accept = decide_acceptance(cpm, target_flat, losing_id, gaining_id, delta_h);
    if accept {
        #[cfg(not(feature = "fused-energy"))]
        commit_accepted(
            cpm,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            delta_h_conservative,
        );
        #[cfg(feature = "fused-energy")]
        commit_accepted(
            cpm,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            delta_h_conservative,
            fused_interface_deltas,
        );
    }
    accept
}

fn decide_acceptance<const D: usize>(
    cpm: &mut CPM<D>,
    target_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
    delta_h: f64,
) -> bool {
    match cpm.config.acceptance {
        // Accept unconditionally when delta_H <= 0, else with probability
        // exp(-delta_H). The `<= 0` shortcut is only valid here: standard
        // Metropolis's acceptance probability is exactly 1 in that case.
        AcceptanceMode::Metropolis => delta_h <= 0.0 || cpm.rng.gen::<f64>() < (-delta_h).exp(),
        // min(1, (n_A / n_B) * exp(-delta_H)), test instrument only.
        // Reversible with respect to exp(-H), which is what enables the
        // exact Boltzmann validation — never used in production.
        //
        // Unlike Metropolis, this has no `delta_h <= 0` shortcut: the
        // (n_A / n_B) proposal-asymmetry factor can push the acceptance
        // probability below 1 even for an energetically favourable move,
        // so every move needs the full ratio computed regardless of the
        // sign of delta_h.
        AcceptanceMode::MetropolisHastings => {
            let (n_a, n_b) = neighbour_counts(cpm, target_flat, losing_id, gaining_id);
            debug_assert!(
                n_b > 0,
                "gaining_id came from a copy-neighbour of target_flat, so it must count itself"
            );
            if n_a == 0 {
                // The reverse move (proposing a copy-neighbourhood-adjacent
                // cell site at this target, to flip it back) is impossible:
                // there is no such site. The true ratio is exactly 0 — this
                // move must be rejected unconditionally. `0.0 / n_b` alone
                // already evaluates to `0.0` (n_b > 0, so no division-by-
                // zero), but `0.0 * (-delta_h).exp()` becomes `NaN` if
                // `(-delta_h).exp()` overflows to infinity — and
                // `NaN.min(1.0)` returns `1.0` (`f64::min` treats NaN as
                // "defer to the other operand"), which would silently
                // accept an irreversible move. Guard explicitly rather than
                // rely on the multiplication order never producing that
                // combination for whatever delta_h magnitudes happen to
                // occur.
                return false;
            }
            let ratio = (n_a as f64 / n_b as f64) * (-delta_h).exp();
            debug_assert!(
                !ratio.is_nan(),
                "MH ratio is NaN: n_a={n_a} n_b={n_b} delta_h={delta_h}"
            );
            cpm.rng.gen::<f64>() < ratio.min(1.0)
        }
    }
}

/// `n_A(s)`/`n_B(s)`: the number of `s`'s copy-neighbourhood neighbours
/// belonging to the losing/gaining cell respectively. Neither neighbour's
/// own cell changes due to this move (only `s` itself does), so these are
/// well-defined on the pre-move lattice for both the forward and reverse
/// proposal probabilities.
fn neighbour_counts<const D: usize>(
    cpm: &CPM<D>,
    target_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
) -> (usize, usize) {
    let mut n_a = 0;
    let mut n_b = 0;
    for &offset in &cpm.copy_offsets {
        let neighbour_flat = cpm.state.lattice.neighbour(target_flat, offset);
        let cell = cpm.state.lattice.get(neighbour_flat);
        if cell == losing_id {
            n_a += 1;
        }
        if cell == gaining_id {
            n_b += 1;
        }
    }
    (n_a, n_b)
}

#[cfg(not(feature = "fused-energy"))]
fn commit_accepted<const D: usize>(
    cpm: &mut CPM<D>,
    target_flat: usize,
    source_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
    delta_h_conservative: f64,
) {
    let (delta_i_losing, delta_i_gaining) = {
        let ctx = CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            mcs: cpm.mcs,
        };
        cpm.terms.commit(&ctx);
        cpm.terms.interface.interface_deltas(&ctx)
    };
    dynamics::commit(
        &mut cpm.state,
        target_flat,
        losing_id,
        gaining_id,
        (delta_i_losing, delta_i_gaining),
        &cpm.energy_offsets,
        &cpm.energy_weights,
    );
    // Only `target_flat`'s owner changed, so only it and its copy-neighbours
    // can have changed edge-membership (same locality argument as
    // volume/interface/contact bookkeeping) — recompute exactly that local
    // set, on the post-move lattice. No-op under `ProposalMode::Uniform`
    // (`edge_list` is `None`).
    if let Some(edge_list) = cpm.edge_list.as_mut() {
        edge_list.update_around(&cpm.state.lattice, target_flat, &cpm.copy_offsets);
    }
    // Only the conservative delta ever reaches `conservative_energy` — Act's
    // contribution (already folded into acceptance via `delta_h` in
    // `attempt`) must never appear here, since this is exactly the value
    // the brute-force checker compares against a from-scratch conservative
    // recomputation.
    cpm.state.conservative_energy += delta_h_conservative;
}

/// Same as the default `commit_accepted`, except the interface delta pair
/// arrives already computed from pricing the move (`fused_conservative_delta`
/// in `attempt_at`), so this books it directly instead of walking the energy
/// neighbourhood a second time.
#[cfg(feature = "fused-energy")]
fn commit_accepted<const D: usize>(
    cpm: &mut CPM<D>,
    target_flat: usize,
    source_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
    delta_h_conservative: f64,
    interface_deltas: (f64, f64),
) {
    {
        let ctx = CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            mcs: cpm.mcs,
        };
        cpm.terms.commit(&ctx);
    }
    dynamics::commit(
        &mut cpm.state,
        target_flat,
        losing_id,
        gaining_id,
        interface_deltas,
        &cpm.energy_offsets,
        &cpm.energy_weights,
    );
    if let Some(edge_list) = cpm.edge_list.as_mut() {
        edge_list.update_around(&cpm.state.lattice, target_flat, &cpm.copy_offsets);
    }
    cpm.state.conservative_energy += delta_h_conservative;
}

/// Advance the simulation by `n` MCS. Under `ProposalMode::Uniform` (the
/// default), one MCS is exactly `N_sites` uniformly-targeted attempts,
/// counting early-outs. Under `ProposalMode::EdgeList`, one MCS instead
/// draws `k ~ Binomial(N_sites, |E| / N_sites)` and performs `k` attempts
/// targeted uniformly from the current edge set `E` — the binomial-thinning
/// correction that makes this exactly the `Uniform` chain with null
/// (same-cell) attempts removed, not a different chain (see
/// `config::ProposalMode`'s doc comment). `|E|` is read once, at the start
/// of each MCS, not re-read as `E` changes during that MCS's own attempts.
///
/// Burn-in/readout/sampling-interval semantics — splitting a run into a
/// discarded window and a measured one — live in `output.rs`, which wraps
/// this as its low-level primitive.
pub fn run_mcs<const D: usize>(cpm: &mut CPM<D>, n: u64) {
    let n_sites = cpm.n_sites();
    for _ in 0..n {
        match cpm.config.proposal {
            ProposalMode::Uniform => {
                for _ in 0..n_sites {
                    attempt(cpm, true);
                }
            }
            ProposalMode::EdgeList => {
                let edge_count = cpm.edge_list.as_ref().map_or(0, |e| e.len());
                let k = if edge_count == 0 {
                    0
                } else {
                    let p = edge_count as f64 / n_sites as f64;
                    sample_binomial(&mut cpm.rng, n_sites as u64, p)
                };
                for _ in 0..k {
                    let Some(target_flat) =
                        cpm.edge_list.as_ref().and_then(|e| e.sample(&mut cpm.rng))
                    else {
                        break;
                    };
                    attempt_at(cpm, target_flat, true);
                }
            }
        }
        cpm.terms.tick::<D>(cpm.mcs);
        cpm.mcs += 1;
    }
}

/// `k ~ Binomial(n, p)`. `rand_distr::Binomial` requires `0.0 <= p <= 1.0`;
/// `p` here is always `|E| / N_sites` with `E` a subset of the interior, so
/// it is always in range by construction.
fn sample_binomial(rng: &mut crate::rng::Rng, n: u64, p: f64) -> u64 {
    Binomial::new(n, p)
        .expect("edge_count / n_sites is always in [0, 1]")
        .sample(rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use crate::config::UserConfig;
    use crate::lattice::Boundary;

    fn minimal_2d_config(seed: u64) -> crate::config::ResolvedConfig<2> {
        UserConfig::<2> {
            // Large relative to target_volume so ordinary growth/jiggle
            // never legitimately approaches the half-box extent guard —
            // this is a dynamics smoke test, not a confluency/packing one.
            grid: Some([30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 15,
                    target_interface: 30,
                    lambda_volume: 1.0,
                    lambda_interface: 0.1,
                    lambda_act: 0.0,
                    max_act: 0,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 15,
                    target_interface: 30,
                    lambda_volume: 1.0,
                    lambda_interface: 0.1,
                    lambda_act: 0.0,
                    max_act: 0,
                },
            ],
            cell_counts: vec![1, 1],
            adhesion: Some(vec![
                vec![0.0, 2.0, 2.0],
                vec![2.0, 1.0, 4.0],
                vec![2.0, 4.0, 1.0],
            ]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap()
    }

    fn place_two_touching_cells(cpm: &mut CPM<2>) {
        use crate::cell::Cell;
        cpm.state.cells = vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ];
        cpm.state.volume = vec![0, 0];
        cpm.state.interface = vec![0, 0];
        cpm.state.sum_position = vec![[0, 0], [0, 0]];
        cpm.state.sum_position_outer = vec![[[0, 0], [0, 0]]; 2];
        // A small connected blob for each cell, side by side, well away
        // from the boundary.
        for x in 1..4 {
            for y in 1..4 {
                let flat = cpm.state.lattice.flat_index([x, y]);
                cpm.state.lattice.set(flat, 1);
                cpm.state.volume[0] += 1;
            }
        }
        for x in 4..7 {
            for y in 1..4 {
                let flat = cpm.state.lattice.flat_index([x, y]);
                cpm.state.lattice.set(flat, 2);
                cpm.state.volume[1] += 1;
            }
        }
        // Interface is recomputed by the caller via the brute-force-style
        // helper in these tests, not hand-derived here.
        recompute_interface(cpm);
        // Never went through `dynamics::commit`, so the contact graph needs
        // establishing from scratch too — same reason `initialization.rs`
        // needs it.
        cpm.state.contact = crate::state::recompute_contact(
            &cpm.state.lattice,
            &cpm.energy_offsets,
            cpm.state.cells.len(),
        );
        cpm.state.conservative_energy = cpm.terms.global_energy(&cpm.state);
    }

    fn recompute_interface(cpm: &mut CPM<2>) {
        let mut interface = vec![0u32; cpm.state.cells.len()];
        cpm.state.lattice.each_interior_coord(|coord| {
            let flat = cpm.state.lattice.flat_index(coord);
            let cell = cpm.state.lattice.get(flat);
            if cell == 0 {
                return;
            }
            for &offset in &cpm.energy_offsets {
                let neighbour = cpm
                    .state
                    .lattice
                    .get(cpm.state.lattice.neighbour(flat, offset));
                if neighbour != cell {
                    interface[index_of(cell)] += 1;
                }
            }
        });
        cpm.state.interface = interface;
    }

    #[test]
    fn run_mcs_advances_the_mcs_counter() {
        let mut cpm = CPM::new(minimal_2d_config(1)).unwrap();
        place_two_touching_cells(&mut cpm);
        run_mcs(&mut cpm, 3);
        assert_eq!(cpm.mcs, 3);
    }

    #[test]
    fn same_seed_gives_a_bit_identical_trajectory() {
        let mut a = CPM::new(minimal_2d_config(42)).unwrap();
        let mut b = CPM::new(minimal_2d_config(42)).unwrap();
        place_two_touching_cells(&mut a);
        place_two_touching_cells(&mut b);
        run_mcs(&mut a, 5);
        run_mcs(&mut b, 5);
        assert_eq!(
            a.state.lattice.get(a.state.lattice.flat_index([3, 3])),
            b.state.lattice.get(b.state.lattice.flat_index([3, 3]))
        );
        assert_eq!(a.state.volume, b.state.volume);
        assert_eq!(a.state.interface, b.state.interface);
        assert_eq!(a.state.conservative_energy, b.state.conservative_energy);
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = CPM::new(minimal_2d_config(1)).unwrap();
        let mut b = CPM::new(minimal_2d_config(2)).unwrap();
        place_two_touching_cells(&mut a);
        place_two_touching_cells(&mut b);
        run_mcs(&mut a, 10);
        run_mcs(&mut b, 10);
        assert_ne!(a.state.volume, b.state.volume);
    }

    #[test]
    fn never_takes_a_cell_below_minimum_volume() {
        let mut cpm = CPM::new(minimal_2d_config(7)).unwrap();
        place_two_touching_cells(&mut cpm);
        run_mcs(&mut cpm, 20);
        assert!(cpm.state.volume[0] >= cpm.config.min_cell_volume);
        assert!(cpm.state.volume[1] >= cpm.config.min_cell_volume);
    }

    /// A physical sanity check that a term produces its intended qualitative
    /// effect when its coupling is increased — stochastic, so it supplements
    /// the exact `energy.rs` unit tests rather than replacing them. A single
    /// cell, volume and a weak interface constraint only (no adhesion, no
    /// second cell to interact with), on a large periodic grid so it never
    /// legitimately approaches the half-box extent guard even while actively
    /// crawling. Compares mean squared centroid displacement over the same
    /// number of MCS with Act off vs. on, averaged over several seeds since
    /// any one seed's displacement is highly variable on its own.
    ///
    /// `GRID = 120` is not an arbitrary round number: an earlier `60`
    /// legitimately tripped the extent guard — a strongly-motile cell under
    /// persistent Act-driven movement can transiently deform/elongate well
    /// beyond its resting size, and once that deformation approaches `L/2`
    /// the nearest-periodic-image choice genuinely becomes ambiguous (this
    /// is a real limit of unwrapped-position tracking, not a bug in
    /// `ActTerm`). That guard is a `debug_assert!`, so a release build gave
    /// no warning at all — just a silently corrupted, physically nonsensical
    /// displacement (observed: an "MSD" in the millions on a 60x60 grid).
    /// Confirmed by rerunning the same scenario in a debug build, where the
    /// assertion fired outright. `120` keeps the cell's deformation
    /// comfortably under `L/2` at these parameters; if this test is ever
    /// retuned to a stronger `lambda_act`/`max_act`, rerun it in debug mode
    /// first to confirm the guard still doesn't fire before trusting a
    /// release-mode number.
    fn single_cell_config(
        seed: u64,
        lambda_act: f64,
        max_act: u32,
    ) -> crate::config::ResolvedConfig<2> {
        const GRID: usize = 120;
        let mut lattice = vec![0 as CellId; GRID * GRID];
        for x in 57..62 {
            for y in 57..62 {
                lattice[x * GRID + y] = 1;
            }
        }
        let mut config = UserConfig::<2> {
            grid: Some([GRID, GRID]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 25,
                target_interface: 40,
                lambda_volume: 1.0,
                lambda_interface: 0.1,
                lambda_act,
                max_act,
            }],
            cell_counts: vec![1],
            adhesion: Some(vec![vec![0.0, 0.0], vec![0.0, 0.0]]),
            initialization: Some(crate::config::InitializationSpec::Explicit {
                lattice,
                cell_type_of: vec![0],
            }),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        config.active_terms.act = lambda_act > 0.0;
        config
    }

    #[test]
    #[ignore = "slow: 2 x 10 seeds x 1000 MCS on a 120x120 grid, run with --release -- --ignored"]
    fn act_increases_cell_motility() {
        const MCS: u64 = 1000;
        const SEEDS: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        fn mean_squared_displacement(lambda_act: f64, max_act: u32) -> f64 {
            let mut total = 0.0;
            for &seed in &SEEDS {
                let config = single_cell_config(seed, lambda_act, max_act);
                let mut cpm = CPM::new(config).unwrap();
                crate::initialization::initialize(&mut cpm).expect("seed must initialise");
                let v0 = cpm.state.volume[0] as f64;
                let start = [
                    cpm.state.sum_position[0][0] as f64 / v0,
                    cpm.state.sum_position[0][1] as f64 / v0,
                ];
                run_mcs(&mut cpm, MCS);
                let v1 = cpm.state.volume[0] as f64;
                let end = [
                    cpm.state.sum_position[0][0] as f64 / v1,
                    cpm.state.sum_position[0][1] as f64 / v1,
                ];
                let dx = end[0] - start[0];
                let dy = end[1] - start[1];
                total += dx * dx + dy * dy;
            }
            total / SEEDS.len() as f64
        }

        let msd_off = mean_squared_displacement(0.0, 0);
        let msd_on = mean_squared_displacement(10.0, 50);
        eprintln!(
            "msd_off={msd_off:.4} msd_on={msd_on:.4} ratio={:.2}",
            msd_on / msd_off
        );
        // Measured ~86x at these parameters (2026-09-16) — `3x` is a
        // generous margin well clear of run-to-run noise while still
        // failing loudly if Act's directional bias were broken or
        // accidentally near-cancelled.
        assert!(
            msd_on > msd_off * 3.0,
            "Act should measurably increase motility over {MCS} MCS: \
             off={msd_off:.2} on={msd_on:.2}"
        );
    }

    /// A performance diagnostic: isolates `Terms::total_delta` (energy
    /// computation, paid by every non-null attempt) from `dynamics::commit`
    /// (bookkeeping, paid only by accepted moves) on real proposals
    /// harvested from a confluent-density 3D fixture — the number that
    /// decides whether a fused delta_H computation or a `contact`/
    /// memory-layout change would actually pay off, rather than guessing.
    /// `#[ignore]`d: manual `Instant`-based timing (no Criterion access from
    /// inside `cpm-core` itself — this needs `pub(crate)`
    /// `interface_deltas`/`dynamics::commit`, which an external bench crate
    /// can't reach), so this is a one-off diagnostic, not part of the
    /// regular suite. Run with `--release -- --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic only: run with --release -- --ignored --nocapture"]
    fn diag_total_delta_vs_commit_cost_3d() {
        use crate::cell::CellType;
        use crate::config::UserConfig;
        use crate::connectivity::preserves_topology;
        use crate::energy::CopyContext;
        use crate::lattice::Boundary;
        use std::time::Instant;

        let config = UserConfig::<3> {
            grid: Some([40, 40, 40]),
            // Fixed, not Periodic: this diagnostic's own relaxation call
            // tripped the unwrapped-position half-box guard even at a
            // modest 5 MCS on this fixture (only ever caught here because
            // `cargo test` runs debug builds — `benches/scale.rs`'s and
            // `benches/topology.rs`'s identical fixture/relaxation never
            // trips it, since Criterion always builds release, where
            // `debug_assert!` compiles out). `Fixed` sidesteps that guard
            // entirely and doesn't affect what this diagnostic measures
            // (raw energy/commit cost, not periodic-boundary behaviour).
            boundary: Some(Boundary::Fixed),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 125,
                target_interface: 250,
                lambda_volume: 1.0,
                lambda_interface: 0.2,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![150],
            adhesion: Some(vec![vec![0.0, 3.0], vec![3.0, 1.0]]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(1),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut cpm = CPM::new(config).unwrap();
        crate::initialization::initialize(&mut cpm).expect("fixture must initialise");
        run_mcs(&mut cpm, 5); // relax past the freshly-grown, atypically smooth state

        // Harvest real (non-null, connectivity-passing) proposals without
        // mutating the lattice.
        let mut rng = crate::rng::rng_from_seed(2);
        let dims = cpm.state.lattice.dims();
        let mut proposals = Vec::new();
        while proposals.len() < 5_000 {
            let target_coord: [usize; 3] = std::array::from_fn(|d| rng.gen_range(0..dims[d]));
            let target_flat = cpm.state.lattice.flat_index(target_coord);
            let source_offset = cpm.copy_offsets[rng.gen_range(0..cpm.copy_offsets.len())];
            let source_flat = cpm.state.lattice.neighbour(target_flat, source_offset);
            let losing_id = cpm.state.lattice.get(target_flat);
            let gaining_id = cpm.state.lattice.get(source_flat);
            if losing_id == gaining_id {
                continue;
            }
            if !preserves_topology(
                &cpm.state.lattice,
                target_flat,
                losing_id,
                &cpm.topology_offsets,
            ) {
                continue;
            }
            proposals.push((target_flat, source_flat, losing_id, gaining_id));
        }

        // total_delta: read-only, timed directly over the harvested set.
        let start = Instant::now();
        let mut sum = 0.0;
        for &(target_flat, source_flat, losing_id, gaining_id) in &proposals {
            let ctx = CopyContext {
                lattice: &cpm.state.lattice,
                state: &cpm.state,
                target_flat,
                source_flat,
                losing_id,
                gaining_id,
                mcs: cpm.mcs,
            };
            sum += cpm.terms.total_delta(&ctx);
        }
        let total_delta_elapsed = start.elapsed();
        std::hint::black_box(sum);

        // commit: applied to a *fresh clone* of the pre-relaxation state on
        // every iteration, not accumulated onto one evolving state — every
        // harvested proposal here is unconditionally accepted (no Metropolis
        // gating), so accumulating all 5,000 onto one state would compound
        // far more directed drift than any real trajectory produces and
        // reliably trips the half-box guard. Cloning per call is slower to
        // set up but measures each commit in isolation, which is what this
        // diagnostic actually wants anyway.
        let mut commit_elapsed = std::time::Duration::ZERO;
        for &(target_flat, source_flat, losing_id, gaining_id) in &proposals {
            let mut state = cpm.state.clone();
            let ctx = CopyContext {
                lattice: &state.lattice,
                state: &state,
                target_flat,
                source_flat,
                losing_id,
                gaining_id,
                mcs: cpm.mcs,
            };
            let interface_deltas = cpm.terms.interface.interface_deltas(&ctx);
            let start = Instant::now();
            crate::dynamics::commit(
                &mut state,
                target_flat,
                losing_id,
                gaining_id,
                interface_deltas,
                &cpm.energy_offsets,
                &cpm.energy_weights,
            );
            commit_elapsed += start.elapsed();
        }

        // update_contact: the same fresh-clone-per-call isolation, timing
        // just the `State::contact` HashMap maintenance piece of `commit`
        // (it only reads the pre-move lattice and writes `state.contact`,
        // so it's safe to run standalone without the rest of `commit`).
        let mut update_contact_elapsed = std::time::Duration::ZERO;
        for &(target_flat, _source_flat, losing_id, gaining_id) in &proposals {
            let mut state = cpm.state.clone();
            let start = Instant::now();
            crate::dynamics::update_contact(
                &mut state,
                target_flat,
                losing_id,
                gaining_id,
                &cpm.energy_offsets,
                &cpm.energy_weights,
            );
            update_contact_elapsed += start.elapsed();
        }

        eprintln!(
            "total_delta: {:.1} ns/call ({} calls in {:?})",
            total_delta_elapsed.as_nanos() as f64 / proposals.len() as f64,
            proposals.len(),
            total_delta_elapsed
        );
        eprintln!(
            "commit: {:.1} ns/call ({} calls in {:?})",
            commit_elapsed.as_nanos() as f64 / proposals.len() as f64,
            proposals.len(),
            commit_elapsed
        );
        eprintln!(
            "update_contact: {:.1} ns/call ({} calls in {:?})",
            update_contact_elapsed.as_nanos() as f64 / proposals.len() as f64,
            proposals.len(),
            update_contact_elapsed
        );
    }
}
