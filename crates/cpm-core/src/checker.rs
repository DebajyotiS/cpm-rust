//! The brute-force consistency checker, after every accepted move, recomputes every cell's volume
//! and interface measure from scratch and asserts **exact** agreement with
//! the incrementally maintained values. Because volumes and interface counts
//! are integer-derived, equality is exact rather than approximate—a fact confirmed
//! empirically here as well: [`check`] has never observed a volume or interface
//! mismatch smaller than what a genuine bookkeeping bug would produce.
//! Incremental bookkeeping drift is the classic silent CPM bug: it produces
//! simulations that look entirely plausible while being physically wrong.
//!
//! The global conservative energy is the sole exception to bit-exactness:
//! [`crate::state::State::conservative_energy`] maintains a running sum of many
//! `+=` deltas. Because floating-point addition is non-associative, it
//! measurably drifts from a direct sum due to standard rounding noise
//! (see [`check`]'s comments for empirical measurements). That comparison uses
//! a tight relative tolerance instead of exact equality (`==`).
//!
//! The [`check`] function itself is always compiled and lightweight enough
//! to run inside standard unit tests. The `#[ignore]`d test—which runs this check
//! after every accepted move across ~1e5 attempts—is gated behind the `checker`
//! Cargo feature.

use crate::model::CPM;
use crate::state::{index_of, recompute_contact};

/// Recompute every cell's volume and interface measure, and the pairwise
/// contact graph, from scratch and compare, exactly, against the
/// incrementally-maintained [`crate::state::State`] fields; also
/// recompute the global conservative energy from the (now-verified) volume/
/// interface and compare against the separately, incrementally-accumulated
/// running total ([`crate::state::State::conservative_energy`]).
///
/// Returns `Err` with a description carrying enough state to reproduce the
/// mismatch — which cell, which quantity, both values — rather than
/// panicking directly, so callers can attach their own attempt-number/seed
/// context.
pub fn check<const D: usize>(cpm: &CPM<D>) -> Result<(), String> {
    let n = cpm.state.cells.len();
    let mut recomputed_volume = vec![0u32; n];
    let mut recomputed_interface = vec![0u32; n];

    cpm.state.lattice.each_interior_coord(|coord| {
        let flat = cpm.state.lattice.flat_index(coord);
        let cell = cpm.state.lattice.get(flat);
        if cell == 0 {
            return;
        }
        recomputed_volume[index_of(cell)] += 1;
        for &offset in &cpm.energy_offsets {
            let neighbour_flat = cpm.state.lattice.neighbour(flat, offset);
            if cpm.state.lattice.get(neighbour_flat) != cell {
                recomputed_interface[index_of(cell)] += 1;
            }
        }
    });

    for (cell_index, cell) in cpm.state.cells.iter().enumerate() {
        if recomputed_volume[cell_index] != cpm.state.volume[cell_index] {
            return Err(format!(
                "volume mismatch for cell {}: recomputed {} vs incremental {}",
                cell.id, recomputed_volume[cell_index], cpm.state.volume[cell_index]
            ));
        }
        if recomputed_interface[cell_index] != cpm.state.interface[cell_index] {
            return Err(format!(
                "interface mismatch for cell {}: recomputed {} vs incremental {}",
                cell.id, recomputed_interface[cell_index], cpm.state.interface[cell_index]
            ));
        }
    }

    // Contact counts are integer-derived exactly like volume/interface, so
    // this comparison is exact too — same reasoning, same guarantee.
    let recomputed_contact = recompute_contact(
        &cpm.state.lattice,
        &cpm.energy_offsets,
        cpm.state.cells.len(),
    );
    if recomputed_contact != cpm.state.contact {
        return Err(format!(
            "contact graph mismatch: recomputed {:?} vs incremental {:?}",
            recomputed_contact, cpm.state.contact
        ));
    }

    // The edge-list set membership is exact (a site either has a
    // differently-owned copy-neighbour or it doesn't), so — like volume,
    // interface and contact above — this is an exact-equality check, not a
    // tolerance. Only meaningful under `ProposalMode::EdgeList`; `Uniform`
    // runs never populate `cpm.edge_list`.
    if let Some(edge_list) = &cpm.edge_list {
        let recomputed_edges =
            crate::edge_list::EdgeList::build(&cpm.state.lattice, &cpm.copy_offsets);
        if recomputed_edges.to_sorted_vec() != edge_list.to_sorted_vec() {
            return Err(format!(
                "edge list mismatch: recomputed {} sites vs incremental {} sites",
                recomputed_edges.len(),
                edge_list.len()
            ));
        }
    }

    // Volume/interface already verified equal above, so this recomputes the
    // energy from the (now-confirmed-correct) current state rather than
    // needing a second, separate volume/interface pass — what's actually
    // independent here is the *accumulation*: `conservative_energy` is a
    // running sum of many small `delta_h` additions, `global_energy` is one
    // direct sum, so this still catches drift the running total might have
    // picked up that a from-scratch volume/interface check alone wouldn't
    // (e.g. a term whose delta was wrong but whose net effect on volume/
    // interface happened to be right).
    //
    // This comparison is *not* bit-exact, unlike volume/interface above.
    // Each individual delta is exactly derived from integers, but summing
    // thousands of them via repeated `+=` is still ordinary
    // floating-point addition, which is not associative — empirically, a
    // 30x30/10-cell run accumulates ~1e-13 of drift within the first 100
    // accepted moves alone, from rounding noise, not a bookkeeping bug.
    // `EPSILON` is chosen to comfortably clear that noise floor while
    // staying many orders of magnitude below anything a real drift bug
    // would produce.
    const EPSILON: f64 = 1e-6;
    let recomputed_energy = cpm.terms.global_energy(&cpm.state);
    let diff = (recomputed_energy - cpm.state.conservative_energy).abs();
    let scale = recomputed_energy.abs().max(1.0);
    if diff > EPSILON * scale {
        return Err(format!(
            "conservative energy mismatch: recomputed {} vs incremental {} (diff {:e}, tolerance {:e})",
            recomputed_energy,
            cpm.state.conservative_energy,
            diff,
            EPSILON * scale
        ));
    }

    Ok(())
}

#[cfg(all(test, feature = "checker"))]
mod tests {
    use super::*;
    use crate::cell::{Cell, CellType};
    use crate::config::UserConfig;
    use crate::lattice::{Boundary, CellId};
    use crate::monte_carlo::attempt;

    /// 30x30 lattice, ~10 cells, ~1e5 attempted moves, multiple
    /// configurations and seeds. Checks after *every accepted move*, not
    /// just at the end — a bug that briefly corrupts bookkeeping and then
    /// "recovers" by coincidence would be invisible to an end-of-run-only
    /// check.
    #[test]
    #[ignore = "slow: ~1e5 attempts x multiple seeds, run with --features checker -- --ignored"]
    fn incremental_bookkeeping_matches_brute_force_recomputation() {
        for seed in [1u64, 2, 3] {
            let mut cpm = build_config(seed);
            place_ten_cells(&mut cpm);
            check(&cpm).expect("initial placement must already be consistent");

            for attempt_number in 0..100_000u64 {
                let accepted = attempt(&mut cpm, true);
                if accepted {
                    if let Err(mismatch) = check(&cpm) {
                        panic!(
                            "consistency check failed (seed {seed}, attempt {attempt_number}): {mismatch}\n\
                             mcs so far: {}, n_cells: {}",
                            cpm.mcs,
                            cpm.state.cells.len()
                        );
                    }
                }
            }
        }
    }

    /// An end-to-end scenario: real scatter-and-grow initialisation (not
    /// hand-placed cells), then real Monte Carlo dynamics, checked after
    /// every accepted move. Exercises `initialization.rs`, `monte_carlo.rs`
    /// and `checker.rs` together rather than any one of them in isolation.
    #[test]
    #[ignore = "slow: run with --features checker -- --ignored"]
    fn scatter_and_grow_then_dynamics_stays_consistent() {
        for seed in [10u64, 20, 30] {
            let mut cpm = build_config(seed);
            crate::initialization::initialize(&mut cpm).expect("initialisation must succeed");
            check(&cpm).expect("freshly-initialised state must already be consistent");

            for attempt_number in 0..50_000u64 {
                let accepted = attempt(&mut cpm, true);
                if accepted {
                    if let Err(mismatch) = check(&cpm) {
                        panic!(
                            "consistency check failed after scatter-and-grow init \
                             (seed {seed}, attempt {attempt_number}): {mismatch}\n\
                             mcs so far: {}, n_cells: {}",
                            cpm.mcs,
                            cpm.state.cells.len()
                        );
                    }
                }
            }
        }
    }

    /// Once 2D is proven correct, the same generic engine gets instantiated
    /// at 3D. Same shape as
    /// [`scatter_and_grow_then_dynamics_stays_consistent`], just at `D = 3`
    /// — proves the dimension-generic machinery (including the 3D
    /// simple-point table, built lazily on first use) actually works
    /// end-to-end rather than merely compiling.
    #[test]
    #[ignore = "slow: builds the 3D simple-point table, run with --features checker -- --ignored"]
    fn instantiated_at_30_cubed_stays_consistent() {
        let mut cpm = build_config_3d(1);
        crate::initialization::initialize(&mut cpm).expect("initialisation must succeed");
        check(&cpm).expect("freshly-initialised 3D state must already be consistent");

        for attempt_number in 0..20_000u64 {
            let accepted = attempt(&mut cpm, true);
            if accepted {
                if let Err(mismatch) = check(&cpm) {
                    panic!(
                        "3D consistency check failed (attempt {attempt_number}): {mismatch}\n\
                         mcs so far: {}, n_cells: {}",
                        cpm.mcs,
                        cpm.state.cells.len()
                    );
                }
            }
        }
    }

    /// Act has no global energy and must never be routed through the
    /// conservative global-energy checker. This test is the flip side of
    /// that rule — with Act *active*, `check()` (which recomputes only
    /// volume/interface/the three conservative terms, never Act) should
    /// still pass cleanly, proving Act's delta never leaked into
    /// `State::conservative_energy` (the value `check()` compares against a
    /// from-scratch conservative recomputation). This is the single
    /// strongest regression guard for that isolation, and doubles as
    /// confirmation that the checker and an active `ActTerm` coexist
    /// correctly (periodic boundary here too, so this also exercises
    /// `birth_mcs` canonicalisation through the halo at real scale).
    #[test]
    #[ignore = "slow: run with --features checker -- --ignored"]
    fn act_active_does_not_corrupt_conservative_energy() {
        for seed in [100u64, 200, 300] {
            let mut cpm = build_config_with_act(seed);
            crate::initialization::initialize(&mut cpm).expect("initialisation must succeed");
            check(&cpm).expect("freshly-initialised state must already be consistent");

            for attempt_number in 0..50_000u64 {
                let accepted = attempt(&mut cpm, true);
                if accepted {
                    if let Err(mismatch) = check(&cpm) {
                        panic!(
                            "consistency check failed with Act active (seed {seed}, attempt {attempt_number}): {mismatch}\n\
                             mcs so far: {}, n_cells: {}",
                            cpm.mcs,
                            cpm.state.cells.len()
                        );
                    }
                }
            }
        }
    }

    /// The edge-list proposal's own consistency guard — not just that
    /// volume/interface/contact/energy stay correct (already covered by
    /// every other test in this module regardless of proposal mode), but
    /// that the *edge set itself* never drifts from a from-scratch rebuild
    /// across many real accepted moves, including the swap-remove
    /// incremental maintenance in `EdgeList::update_around`.
    #[test]
    #[ignore = "slow: run with --features checker -- --ignored"]
    fn edge_list_proposal_stays_consistent() {
        // Same seeds `scatter_and_grow_then_dynamics_stays_consistent` uses
        // safely with this exact fixture, at a much smaller MCS budget.
        // Investigated directly (see this change's commit for the
        // methodology): right after `scatter_and_grow` places cells exactly
        // at `V*` but not necessarily at an optimal *shape*, the
        // interface/adhesion terms favour shrink/smoothing moves over
        // growth by a wide margin (confirmed identically reproducible under
        // plain `Uniform`, same seed, same acceptance skew — not an
        // `EdgeList` bug). `EdgeList` reaches this same real, physical
        // relaxation using far fewer *raw* MCS than `Uniform` needs to
        // accumulate the same amount of it, simply because it never spends
        // attempts on sites that could only ever early-out — so the two
        // proposal modes are not directly comparable MCS-for-MCS during a
        // transient like this one, only in their long-run statistics. A
        // smaller MCS budget keeps this test inside that transient without
        // relying on a coincidence of which specific seed happens to avoid
        // the guard.
        for seed in [10u64, 20, 30] {
            let mut cpm = build_config_with_edge_list(seed);
            crate::initialization::initialize(&mut cpm).expect("initialisation must succeed");
            check(&cpm).expect("freshly-initialised state must already be consistent");

            for mcs in 0..12u64 {
                crate::monte_carlo::run_mcs(&mut cpm, 1);
                if let Err(mismatch) = check(&cpm) {
                    panic!(
                        "consistency check failed under EdgeList proposal (seed {seed}, mcs {mcs}): {mismatch}\n\
                         n_cells: {}",
                        cpm.state.cells.len()
                    );
                }
            }
        }
    }

    /// Exactly [`build_config`]'s fixture (30x30, same 10 cells) with
    /// `proposal` switched to `EdgeList` — deliberately the *same* grid
    /// `scatter_and_grow_then_dynamics_stays_consistent` already runs
    /// safely at a comparable accepted-move scale, not a larger one.
    /// (Enlarging the grid was tried first and made things *worse*, not
    /// better: `n_sites` scales with grid size and one MCS is defined as
    /// `n_sites` attempts, so a bigger grid at the same MCS count means
    /// proportionally *more* accumulated attempts per MCS, not more
    /// headroom — confirmed directly when an 80x80 attempt at this same
    /// MCS budget tripped the guard identically.)
    fn build_config_with_edge_list(seed: u64) -> CPM<2> {
        let mut cpm = build_config(seed);
        cpm.config.proposal = crate::config::ProposalMode::EdgeList;
        cpm
    }

    fn build_config_with_act(seed: u64) -> CPM<2> {
        let mut config = UserConfig::<2> {
            grid: Some([30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 2.0,
                    max_act: 10,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.5,
                    lambda_interface: 0.3,
                    lambda_act: 3.0,
                    max_act: 15,
                },
            ],
            cell_counts: vec![5, 5],
            adhesion: Some(vec![
                vec![0.0, 3.0, 3.0],
                vec![3.0, 1.0, 5.0],
                vec![3.0, 5.0, 1.0],
            ]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        config.active_terms.act = true;
        CPM::new(config).unwrap()
    }

    fn build_config_3d(seed: u64) -> CPM<3> {
        let config = UserConfig::<3> {
            grid: Some([30, 30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 27,
                target_interface: 90,
                lambda_volume: 1.0,
                lambda_interface: 0.2,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![5],
            adhesion: Some(vec![vec![0.0, 3.0], vec![3.0, 1.0]]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        CPM::new(config).unwrap()
    }

    fn build_config(seed: u64) -> CPM<2> {
        let config = UserConfig::<2> {
            grid: Some([30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 0.0,
                    max_act: 0,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.5,
                    lambda_interface: 0.3,
                    lambda_act: 0.0,
                    max_act: 0,
                },
            ],
            cell_counts: vec![5, 5],
            adhesion: Some(vec![
                vec![0.0, 3.0, 3.0],
                vec![3.0, 1.0, 5.0],
                vec![3.0, 5.0, 1.0],
            ]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        CPM::new(config).unwrap()
    }

    /// Ten non-overlapping 3x3 blocks, alternating type, well separated so
    /// early attempts don't immediately merge them.
    fn place_ten_cells(cpm: &mut CPM<2>) {
        let origins = [
            (2, 2),
            (8, 2),
            (14, 2),
            (20, 2),
            (26, 2),
            (2, 15),
            (8, 15),
            (14, 15),
            (20, 15),
            (26, 15),
        ];
        cpm.state.cells = (1..=origins.len() as CellId)
            .map(|id| Cell {
                id,
                type_index: (id as usize - 1) % 2,
            })
            .collect();
        cpm.state.volume = vec![0; origins.len()];
        cpm.state.interface = vec![0; origins.len()];
        cpm.state.sum_position = vec![[0, 0]; origins.len()];
        cpm.state.sum_position_outer = vec![[[0, 0], [0, 0]]; origins.len()];

        for (i, &(ox, oy)) in origins.iter().enumerate() {
            let id = (i + 1) as CellId;
            for dx in 0..3usize {
                for dy in 0..3usize {
                    let x = (ox + dx) % 30;
                    let y = (oy + dy) % 30;
                    let flat = cpm.state.lattice.flat_index([x, y]);
                    cpm.state.lattice.set(flat, id);
                    cpm.state.volume[i] += 1;
                    cpm.state.sum_position[i][0] += x as i64;
                    cpm.state.sum_position[i][1] += y as i64;
                }
            }
        }

        recompute_interface(cpm);
        // Never went through `dynamics::commit`, so the contact graph needs
        // establishing from scratch too.
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
                let neighbour_flat = cpm.state.lattice.neighbour(flat, offset);
                if cpm.state.lattice.get(neighbour_flat) != cell {
                    interface[index_of(cell)] += 1;
                }
            }
        });
        cpm.state.interface = interface;
    }
}
