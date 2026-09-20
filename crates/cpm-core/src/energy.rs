//! The conservative Hamiltonian `H = H_V + H_I + H_J`, implemented as
//! `EnergyTerm` implementations composed via a plain struct ([`Terms`])
//! rather than `Box<dyn EnergyTerm>` — this is the hot loop, and dispatch
//! must be static.
//!
//! Each term owns a copy of the config-derived parameters it needs, rather
//! than borrowing `ResolvedConfig` through [`CopyContext`], because
//! `EnergyTerm::energy` — used by the brute-force consistency checker —
//! only receives `&State<D>`, not the config.

use crate::config::ResolvedConfig;
use crate::lattice::{CellId, Lattice};
use crate::state::State;

/// Per-attempt context passed to [`EnergyTerm::delta`]: the proposed copy,
/// and read-only access to the lattice/state it would apply to. Each term
/// already owns its own copy of whatever config-derived parameters it
/// needs, so this struct carries only the per-attempt data itself.
pub struct CopyContext<'a, const D: usize> {
    pub lattice: &'a Lattice<D>,
    pub state: &'a State<D>,
    pub target_flat: usize,
    /// The copy's source site (`s'`) — needed by [`ActTerm`], whose activity
    /// neighbourhood is centred on `source_flat` as well as `target_flat`.
    /// The conservative terms (volume, interface, adhesion) only look at
    /// `target_flat`'s own neighbourhood.
    pub source_flat: usize,
    pub losing_id: CellId,
    pub gaining_id: CellId,
    /// The current MCS, threaded explicitly rather than tracked through
    /// [`EnergyTerm::tick`]: `monte_carlo::run_mcs` calls `tick(cpm.mcs)`
    /// *before* incrementing `cpm.mcs`, so a term that tried to track "the
    /// current mcs" purely via `tick` would be one MCS behind for every MCS
    /// after the first. Needed by [`ActTerm`] for `birth_mcs[s] <- mcs`.
    pub mcs: u64,
}

impl<const D: usize> CopyContext<'_, D> {
    pub fn losing_type(&self) -> Option<usize> {
        self.state.type_of(self.losing_id)
    }

    pub fn gaining_type(&self) -> Option<usize> {
        self.state.type_of(self.gaining_id)
    }
}

/// Not necessarily the difference of any global state function — the Act
/// term is non-conservative and returns `None` from `energy`.
pub trait EnergyTerm<const D: usize> {
    /// Energy change for the proposed copy.
    fn delta(&self, ctx: &CopyContext<D>) -> f64;
    /// Update internal state after an accepted copy. A no-op for the
    /// conservative terms here; Act resets activity.
    fn commit(&mut self, ctx: &CopyContext<D>);
    /// Called once per MCS, for time-dependent state. A no-op here.
    fn tick(&mut self, mcs: u64);
    /// Global energy, for conservative terms only — the brute-force
    /// consistency checker iterates only over terms returning `Some`.
    fn energy(&self, state: &State<D>) -> Option<f64>;
}

/// Adhesion matrix index for a type: medium (`None`) is row/column 0, type
/// `t` is row/column `t + 1`.
fn adhesion_index(cell_type: Option<usize>) -> usize {
    cell_type.map(|t| t + 1).unwrap_or(0)
}

/// `H_V = sum over c >= 1 of lambda_V[tau(c)] * (V_c - V*[tau(c)])^2`.
/// Applies to `c >= 1` only — medium has no volume constraint, expressed
/// here simply by `losing_type()`/`gaining_type()` returning `None` for it.
pub struct VolumeTerm {
    lambda: Vec<f64>,
    target: Vec<u32>,
}

impl VolumeTerm {
    pub fn from_config<const D: usize>(config: &ResolvedConfig<D>) -> Self {
        Self {
            lambda: config.cell_types.iter().map(|t| t.lambda_volume).collect(),
            target: config.cell_types.iter().map(|t| t.target_volume).collect(),
        }
    }

    fn term(&self, type_index: usize, volume: f64) -> f64 {
        let target = self.target[type_index] as f64;
        self.lambda[type_index] * (volume - target).powi(2)
    }
}

impl<const D: usize> EnergyTerm<D> for VolumeTerm {
    fn delta(&self, ctx: &CopyContext<D>) -> f64 {
        let mut total = 0.0;
        if let Some(t) = ctx.losing_type() {
            let v = ctx.state.volume[crate::state::index_of(ctx.losing_id)] as f64;
            total += self.term(t, v - 1.0) - self.term(t, v);
        }
        if let Some(t) = ctx.gaining_type() {
            let v = ctx.state.volume[crate::state::index_of(ctx.gaining_id)] as f64;
            total += self.term(t, v + 1.0) - self.term(t, v);
        }
        total
    }

    fn commit(&mut self, _ctx: &CopyContext<D>) {}
    fn tick(&mut self, _mcs: u64) {}

    fn energy(&self, state: &State<D>) -> Option<f64> {
        let mut total = 0.0;
        for (i, cell) in state.cells.iter().enumerate() {
            total += self.term(cell.type_index, state.volume[i] as f64);
        }
        Some(total)
    }
}

/// `H_I = sum over c >= 1 of lambda_I[tau(c)] * (I_c - I*[tau(c)])^2`.
/// `I_c` is the lattice interface count under the configured energy
/// neighbourhood and weighting.
pub struct InterfaceTerm {
    lambda: Vec<f64>,
    target: Vec<u32>,
    offsets: Vec<isize>,
    weights: Vec<f64>,
}

impl InterfaceTerm {
    pub fn from_config<const D: usize>(
        config: &ResolvedConfig<D>,
        offsets: Vec<isize>,
        weights: Vec<f64>,
    ) -> Self {
        Self {
            lambda: config
                .cell_types
                .iter()
                .map(|t| t.lambda_interface)
                .collect(),
            target: config
                .cell_types
                .iter()
                .map(|t| t.target_interface)
                .collect(),
            offsets,
            weights,
        }
    }

    fn term(&self, type_index: usize, interface: f64) -> f64 {
        let target = self.target[type_index] as f64;
        self.lambda[type_index] * (interface - target).powi(2)
    }

    /// `(delta_I_losing, delta_I_gaining)`: flipping site `s` from A to B
    /// changes `I_C` only for `C` in `{A, B}` — a locality property that is
    /// exactly why this is a local, O(neighbourhood) computation. For each
    /// energy-neighbour `n` with weight `w`:
    /// - `s` leaving A removes all of `s`'s own outgoing contributions to
    ///   `I_A` (`-w` for every `n` not in A) but its A-neighbours now see a
    ///   foreign site where they didn't before (`+w` for every `n` in A).
    /// - Symmetric reasoning for `s` joining B, sign-reversed.
    ///
    /// Exposed `pub(crate)` so `dynamics.rs` applies exactly this same
    /// computation when committing an accepted move — one formula, used
    /// both to price the move and to book it, so the two can never drift
    /// apart.
    pub(crate) fn interface_deltas<const D: usize>(&self, ctx: &CopyContext<D>) -> (f64, f64) {
        let mut delta_losing = 0.0;
        let mut delta_gaining = 0.0;
        for (k, &offset) in self.offsets.iter().enumerate() {
            let w = self.weights[k];
            let neighbour_flat = ctx.lattice.neighbour(ctx.target_flat, offset);
            let neighbour_cell = ctx.lattice.get(neighbour_flat);
            delta_losing += if neighbour_cell == ctx.losing_id {
                w
            } else {
                -w
            };
            delta_gaining += if neighbour_cell == ctx.gaining_id {
                -w
            } else {
                w
            };
        }
        (delta_losing, delta_gaining)
    }
}

impl<const D: usize> EnergyTerm<D> for InterfaceTerm {
    fn delta(&self, ctx: &CopyContext<D>) -> f64 {
        let (delta_losing, delta_gaining) = self.interface_deltas(ctx);
        let mut total = 0.0;
        if let Some(t) = ctx.losing_type() {
            let i = ctx.state.interface[crate::state::index_of(ctx.losing_id)] as f64;
            total += self.term(t, i + delta_losing) - self.term(t, i);
        }
        if let Some(t) = ctx.gaining_type() {
            let i = ctx.state.interface[crate::state::index_of(ctx.gaining_id)] as f64;
            total += self.term(t, i + delta_gaining) - self.term(t, i);
        }
        total
    }

    fn commit(&mut self, _ctx: &CopyContext<D>) {}
    fn tick(&mut self, _mcs: u64) {}

    fn energy(&self, state: &State<D>) -> Option<f64> {
        let mut total = 0.0;
        for (i, cell) in state.cells.iter().enumerate() {
            total += self.term(cell.type_index, state.interface[i] as f64);
        }
        Some(total)
    }
}

/// `H_J = sum over pairs <i,j> with sigma_i != sigma_j of J[tau(sigma_i)][tau(sigma_j)]`
/// — summed over **cell IDs**, not types: two distinct cells of the same
/// type still pay `J[alpha][alpha]`, which falls out automatically here
/// since comparisons are always against `losing_id`/`gaining_id`/the
/// neighbour's raw cell id, never against type indices.
pub struct AdhesionTerm {
    adhesion: Vec<Vec<f64>>,
    offsets: Vec<isize>,
    weights: Vec<f64>,
}

impl AdhesionTerm {
    pub fn from_config<const D: usize>(
        config: &ResolvedConfig<D>,
        offsets: Vec<isize>,
        weights: Vec<f64>,
    ) -> Self {
        Self {
            adhesion: config.adhesion.clone(),
            offsets,
            weights,
        }
    }
}

impl<const D: usize> EnergyTerm<D> for AdhesionTerm {
    fn delta(&self, ctx: &CopyContext<D>) -> f64 {
        let losing_idx = adhesion_index(ctx.losing_type());
        let gaining_idx = adhesion_index(ctx.gaining_type());
        let mut total = 0.0;
        for (k, &offset) in self.offsets.iter().enumerate() {
            let w = self.weights[k];
            let neighbour_flat = ctx.lattice.neighbour(ctx.target_flat, offset);
            let neighbour_cell = ctx.lattice.get(neighbour_flat);
            let neighbour_idx = adhesion_index(ctx.state.type_of(neighbour_cell));
            let before = if neighbour_cell != ctx.losing_id {
                self.adhesion[losing_idx][neighbour_idx]
            } else {
                0.0
            };
            let after = if neighbour_cell != ctx.gaining_id {
                self.adhesion[gaining_idx][neighbour_idx]
            } else {
                0.0
            };
            total += w * (after - before);
        }
        total
    }

    fn commit(&mut self, _ctx: &CopyContext<D>) {}
    fn tick(&mut self, _mcs: u64) {}

    fn energy(&self, state: &State<D>) -> Option<f64> {
        let lattice = &state.lattice;
        let periodic = lattice.boundary() == crate::lattice::Boundary::Periodic;
        let mut total = 0.0;
        lattice.each_interior_coord(|coord| {
            let i_flat = lattice.flat_index(coord);
            let i_cell = lattice.get(i_flat);
            let i_idx = adhesion_index(state.type_of(i_cell));
            for (k, &offset) in self.offsets.iter().enumerate() {
                let j_flat = lattice.neighbour(i_flat, offset);
                // Every pair has at least one interior member (only interior
                // sites are ever `i`). Dedupe by comparing against the
                // *interior representative* of `j`, not its raw flat index:
                // under a periodic boundary, `j` may be a halo cell that
                // mirrors some other interior site `j'` — the pair (i, j)
                // is then physically the same pair as (j', i's own halo
                // mirror on j''s side), so both must resolve to the same
                // comparison key or the pair gets counted twice. A `Fixed`
                // boundary's halo is permanently medium, not a stand-in for
                // any real site, so it's always counted (never deduped).
                let dedupe_key = if lattice.is_interior(j_flat) {
                    Some(j_flat)
                } else if periodic {
                    Some(lattice.wrapped_interior_flat(j_flat))
                } else {
                    None
                };
                if dedupe_key.is_some_and(|key| key <= i_flat) {
                    continue;
                }
                let j_cell = lattice.get(j_flat);
                if j_cell == i_cell {
                    continue;
                }
                let j_idx = adhesion_index(state.type_of(j_cell));
                total += self.weights[k] * self.adhesion[i_idx][j_idx];
            }
        });
        Some(total)
    }
}

/// `delta_H_Act = -(lambda_Act / Max_Act) * (GM_src - GM_tgt)`. Non-
/// conservative — no corresponding global energy, so [`EnergyTerm::energy`]
/// returns `None` and this term is never covered by the brute-force
/// consistency checker. Its delta must never be added to
/// [`State::conservative_energy`](crate::state::State::conservative_energy);
/// only [`Terms::total_delta`]'s conservative sum goes there — see
/// `monte_carlo.rs`'s `attempt`, which keeps this term's contribution
/// separate all the way to `commit_accepted`.
///
/// Two edge cases are resolved explicitly rather than left to guesswork:
/// if `gaining_id == 0` (medium invading — no cell type exists to source
/// `lambda_Act`/`Max_Act` from), `delta` is `0.0` unconditionally. If
/// `losing_id == 0` (a cell growing into empty medium — the common case),
/// the term stays active: `GM_tgt` is computed over medium-owned
/// neighbours of the target using whatever `birth_mcs` they carry, which
/// is typically decayed to zero activity for long-medium sites, so fresh
/// territory offers no resistance — matching standard Act-model behaviour
/// (protrusive growth into empty space).
pub struct ActTerm {
    lambda_act: Vec<f64>,
    max_act: Vec<u32>,
    /// The **copy** neighbourhood, not the energy one — the only term that
    /// needs it; every other term uses the energy neighbourhood.
    copy_offsets: Vec<isize>,
    /// Per-site (padded-lattice-indexed, matching `Lattice.cells`'s own
    /// sizing), the MCS at which the site was last acquired by its current
    /// owner. Lives here rather than on `State<D>`: this is `ActTerm`'s own
    /// internal state, exactly what `EnergyTerm::commit` exists to update,
    /// and keeping it off `State<D>` keeps that struct's own scope — values
    /// the brute-force checker recomputes exactly from scratch — honest.
    /// All-zero at construction: a site occupied at `mcs = 0` reads as full
    /// activity from the start (`max(0, Max_Act - (0 - 0)) = Max_Act`), so
    /// no special-casing is needed in `initialization.rs`.
    birth_mcs: Vec<u32>,
}

impl ActTerm {
    pub fn from_config<const D: usize>(
        config: &ResolvedConfig<D>,
        copy_offsets: Vec<isize>,
    ) -> Self {
        // Matches `Lattice::new`'s own padding exactly, so `birth_mcs` is
        // indexable by any flat index `Lattice` itself would ever produce
        // (interior or halo) without needing a `&Lattice<D>` threaded in.
        let padded_sites: usize = config.grid.iter().map(|&d| d + 2).product();
        Self {
            lambda_act: config.cell_types.iter().map(|t| t.lambda_act).collect(),
            max_act: config.cell_types.iter().map(|t| t.max_act).collect(),
            copy_offsets,
            birth_mcs: vec![0u32; padded_sites],
        }
    }

    /// `flat`, canonicalised to the interior site it represents. Only
    /// meaningful to canonicalise under a periodic boundary — a `Fixed`
    /// boundary's halo is permanently medium (never a same-cell site, so
    /// never retained by [`Self::geometric_mean`] regardless of whatever
    /// stale value sits in `birth_mcs` there) and
    /// `Lattice::wrapped_interior_flat` debug-asserts periodic. Mirrors the
    /// exact guard `AdhesionTerm::energy` already uses for the same reason.
    fn canonical_flat<const D: usize>(lattice: &Lattice<D>, flat: usize) -> usize {
        if lattice.boundary() == crate::lattice::Boundary::Periodic {
            lattice.wrapped_interior_flat(flat)
        } else {
            flat
        }
    }

    /// Geometric mean of activity over `{site} ∪ copy-neighbours of site`,
    /// restricted to sites currently owned by `relevant_id` — the candidate
    /// site is included, and the mean is restricted to sites of the
    /// relevant cell. `site` always belongs to its own current owner by
    /// construction, so when `relevant_id` is `site`'s own current owner
    /// (always true for how [`EnergyTerm::delta`] below calls this) the
    /// retained set is never empty. A zero-activity retained site forces
    /// the product — and hence this geometric mean — to zero on its own;
    /// no separate special case is needed for that.
    ///
    /// `max_act` is the **gaining cell's** `Max_Act` — the same value used
    /// in the delta's own `lambda_Act / Max_Act` prefactor — applied
    /// uniformly to every retained site in *both* `GM_src` and `GM_tgt`,
    /// not each site's own owner's `Max_Act`. This is deliberate, not an
    /// approximation: the formula names a single `Max_Act` for the whole
    /// delta, and every retained site in a given `geometric_mean` call
    /// already shares one owner (`relevant_id`, by the same-cell
    /// restriction above) — so a per-owner `Max_Act` would need to be
    /// looked up per call anyway, and doing that naively breaks exactly
    /// when `relevant_id` is medium (`losing_id == 0`, the common
    /// grow-into-empty-space case): medium has no `CellType` to source a
    /// `Max_Act` from. Using the gaining cell's fixed `Max_Act` throughout
    /// sidesteps that entirely and matches the formula's own
    /// single-prefactor reading.
    fn geometric_mean<const D: usize>(
        &self,
        ctx: &CopyContext<D>,
        site: usize,
        relevant_id: CellId,
        max_act: u32,
    ) -> f64 {
        let lattice = ctx.lattice;
        // Canonicalise `site` itself *before* taking its neighbours: `site`
        // may already be a halo index (`source_flat` can be a periodic-halo
        // mirror of a real cell, since a copy-neighbour of an edge-adjacent
        // target can land in the halo), and the lattice's 1-site halo only
        // guarantees a stencil offset stays in bounds when applied to an
        // *interior* site — applying `copy_offsets` a second time to an
        // already-halo index can step outside the padded array entirely.
        // (Under `Fixed`, `site` reaching here as a halo index can't
        // happen: a `Fixed` halo is always medium, and `delta` already
        // returns early whenever the relevant cell for this call would be
        // medium — `gaining_id == 0` for `source_flat`, and `target_flat`
        // is never a halo index at all.)
        let interior_site = Self::canonical_flat(lattice, site);
        let mut product = 1.0f64;
        let mut count = 0u32;
        let mut consider = |flat: usize| {
            let canonical = Self::canonical_flat(lattice, flat);
            if lattice.get(canonical) != relevant_id {
                return;
            }
            let birth = self.birth_mcs[canonical];
            let elapsed = ctx.mcs.saturating_sub(birth as u64);
            let activity = (max_act as u64).saturating_sub(elapsed) as f64;
            product *= activity;
            count += 1;
        };
        consider(interior_site);
        for &offset in &self.copy_offsets {
            consider(lattice.neighbour(interior_site, offset));
        }
        debug_assert!(count > 0, "site always belongs to its own current owner");
        product.powf(1.0 / count as f64)
    }
}

impl<const D: usize> EnergyTerm<D> for ActTerm {
    fn delta(&self, ctx: &CopyContext<D>) -> f64 {
        let Some(gaining_type) = ctx.gaining_type() else {
            // Medium invading: no cell type to source lambda_Act/Max_Act from.
            return 0.0;
        };
        let lambda = self.lambda_act[gaining_type];
        let max_act = self.max_act[gaining_type];
        let gm_src = self.geometric_mean(ctx, ctx.source_flat, ctx.gaining_id, max_act);
        let gm_tgt = self.geometric_mean(ctx, ctx.target_flat, ctx.losing_id, max_act);
        -(lambda / max_act as f64) * (gm_src - gm_tgt)
    }

    fn commit(&mut self, ctx: &CopyContext<D>) {
        // `target_flat` is always an interior site: `attempt()` only ever
        // samples interior target coordinates. No canonicalisation needed
        // on this write side (unlike the reads in `geometric_mean`).
        self.birth_mcs[ctx.target_flat] = ctx.mcs as u32;
    }

    fn tick(&mut self, _mcs: u64) {}

    fn energy(&self, _state: &State<D>) -> Option<f64> {
        None
    }
}

/// The three conservative terms plus the optional non-conservative Act
/// term, composed by static dispatch. Act is never covered by the
/// brute-force consistency checker — [`Terms::total_delta`] and
/// [`Terms::global_energy`] stay conservative-only; [`Terms::act_delta`]
/// is the separate entry point for Act's contribution.
pub struct Terms {
    pub volume: VolumeTerm,
    pub interface: InterfaceTerm,
    pub adhesion: AdhesionTerm,
    /// `Some` iff `config.active_terms.act`. `None` means
    /// [`Terms::act_delta`] always returns `0.0` and no `ActTerm` — with its
    /// padded-lattice-sized `birth_mcs` — is ever allocated.
    pub act: Option<ActTerm>,
}

impl Terms {
    pub fn from_config<const D: usize>(
        config: &ResolvedConfig<D>,
        energy_offsets: &[isize],
        energy_weights: &[f64],
        copy_offsets: &[isize],
    ) -> Self {
        Self {
            volume: VolumeTerm::from_config(config),
            interface: InterfaceTerm::from_config(
                config,
                energy_offsets.to_vec(),
                energy_weights.to_vec(),
            ),
            adhesion: AdhesionTerm::from_config(
                config,
                energy_offsets.to_vec(),
                energy_weights.to_vec(),
            ),
            act: config
                .active_terms
                .act
                .then(|| ActTerm::from_config(config, copy_offsets.to_vec())),
        }
    }

    /// Conservative delta only (volume + interface + adhesion) — the value
    /// that feeds `State::conservative_energy` bookkeeping
    /// (`monte_carlo.rs`'s `commit_accepted`). Deliberately excludes Act:
    /// folding Act's delta in here would silently corrupt the value the
    /// brute-force checker compares against a from-scratch conservative
    /// recomputation. Callers that need Act's contribution to acceptance
    /// add [`Terms::act_delta`] separately — see `monte_carlo::attempt`.
    pub fn total_delta<const D: usize>(&self, ctx: &CopyContext<D>) -> f64 {
        self.volume.delta(ctx) + self.interface.delta(ctx) + self.adhesion.delta(ctx)
    }

    /// Act's contribution alone (`0.0` if inactive). Kept separate from
    /// [`Terms::total_delta`] so callers can price acceptance on their sum
    /// while still booking only the conservative half into
    /// `State::conservative_energy`.
    pub fn act_delta<const D: usize>(&self, ctx: &CopyContext<D>) -> f64 {
        self.act.as_ref().map_or(0.0, |act| act.delta(ctx))
    }

    pub fn commit<const D: usize>(&mut self, ctx: &CopyContext<D>) {
        self.volume.commit(ctx);
        self.interface.commit(ctx);
        self.adhesion.commit(ctx);
        if let Some(act) = &mut self.act {
            act.commit(ctx);
        }
    }

    /// `D` doesn't affect what `tick` does (none of these terms are
    /// time-dependent) — it's only here because `EnergyTerm<D>` is generic
    /// over `D` and each term implements it for every `D` uniformly, so the
    /// compiler needs a concrete `D` to pick an impl. Callers already have
    /// one in scope (their own `CPM<D>`).
    pub fn tick<const D: usize>(&mut self, mcs: u64) {
        EnergyTerm::<D>::tick(&mut self.volume, mcs);
        EnergyTerm::<D>::tick(&mut self.interface, mcs);
        EnergyTerm::<D>::tick(&mut self.adhesion, mcs);
        if let Some(act) = &mut self.act {
            EnergyTerm::<D>::tick(act, mcs);
        }
    }

    /// Sum of the three conservative terms' global energy — the quantity
    /// the brute-force consistency checker recomputes from scratch and
    /// compares, exactly, to the incrementally-maintained running total.
    /// Act is never part of this sum: it is non-conservative and has no
    /// global energy, so `ActTerm::energy` always returns `None`.
    pub fn global_energy<const D: usize>(&self, state: &State<D>) -> f64 {
        self.volume.energy(state).unwrap()
            + self.interface.energy(state).unwrap()
            + self.adhesion.energy(state).unwrap()
    }
}

/// Result of [`Terms::fused_conservative_delta`]: the scalar conservative ΔH
/// alongside the raw interface `(delta_losing, delta_gaining)` pair, so a
/// caller pricing a move can hand that pair straight to `commit_accepted`
/// when the move is accepted.
#[cfg(feature = "fused-energy")]
pub struct ConservativeDelta {
    pub total: f64,
    pub interface_deltas: (f64, f64),
}

#[cfg(feature = "fused-energy")]
impl Terms {
    /// Prices a move by walking the energy neighbourhood once, reading each
    /// neighbour's cell id and type a single time and feeding it into the
    /// interface and adhesion accumulators together, where
    /// `InterfaceTerm::delta` and `AdhesionTerm::delta` each walk the same
    /// neighbourhood on their own.
    ///
    /// `delta_losing`, `delta_gaining` and `adhesion_total` are kept as three
    /// separate running sums, each accumulated in the same neighbour order
    /// their non-fused counterparts use. Floating-point addition isn't
    /// associative, so keeping the sums apart (fusing only the neighbour
    /// reads) is what makes this produce the exact same result as pricing
    /// the terms separately — `fused_tests` below checks that directly.
    pub fn fused_conservative_delta<const D: usize>(
        &self,
        ctx: &CopyContext<D>,
    ) -> ConservativeDelta {
        let volume_delta = self.volume.delta(ctx);

        debug_assert_eq!(
            self.interface.offsets, self.adhesion.offsets,
            "interface and adhesion must share the energy neighbourhood"
        );
        debug_assert_eq!(
            self.interface.weights, self.adhesion.weights,
            "interface and adhesion must share the energy neighbourhood weights"
        );
        let offsets = &self.interface.offsets;
        let weights = &self.interface.weights;

        let losing_adhesion_idx = adhesion_index(ctx.losing_type());
        let gaining_adhesion_idx = adhesion_index(ctx.gaining_type());

        let mut delta_losing = 0.0;
        let mut delta_gaining = 0.0;
        let mut adhesion_total = 0.0;
        for (k, &offset) in offsets.iter().enumerate() {
            let w = weights[k];
            let neighbour_flat = ctx.lattice.neighbour(ctx.target_flat, offset);
            let neighbour_cell = ctx.lattice.get(neighbour_flat);

            // Same condition, same order as InterfaceTerm::interface_deltas.
            delta_losing += if neighbour_cell == ctx.losing_id {
                w
            } else {
                -w
            };
            delta_gaining += if neighbour_cell == ctx.gaining_id {
                -w
            } else {
                w
            };

            // Same condition, same order as AdhesionTerm::delta.
            let neighbour_adhesion_idx = adhesion_index(ctx.state.type_of(neighbour_cell));
            let before = if neighbour_cell != ctx.losing_id {
                self.adhesion.adhesion[losing_adhesion_idx][neighbour_adhesion_idx]
            } else {
                0.0
            };
            let after = if neighbour_cell != ctx.gaining_id {
                self.adhesion.adhesion[gaining_adhesion_idx][neighbour_adhesion_idx]
            } else {
                0.0
            };
            adhesion_total += w * (after - before);
        }

        let mut interface_delta = 0.0;
        if let Some(t) = ctx.losing_type() {
            let i = ctx.state.interface[crate::state::index_of(ctx.losing_id)] as f64;
            interface_delta += self.interface.term(t, i + delta_losing) - self.interface.term(t, i);
        }
        if let Some(t) = ctx.gaining_type() {
            let i = ctx.state.interface[crate::state::index_of(ctx.gaining_id)] as f64;
            interface_delta +=
                self.interface.term(t, i + delta_gaining) - self.interface.term(t, i);
        }

        ConservativeDelta {
            // Same left-to-right order as `Terms::total_delta`'s
            // `volume.delta + interface.delta + adhesion.delta`.
            total: volume_delta + interface_delta + adhesion_total,
            interface_deltas: (delta_losing, delta_gaining),
        }
    }
}

/// Cross-checks `fused_conservative_delta` against pricing the terms
/// separately, on a two-cell fixture with a real, nonzero interface and
/// adhesion delta.
#[cfg(all(test, feature = "fused-energy"))]
mod fused_tests {
    use super::*;
    use crate::cell::{Cell, CellType};
    use crate::config::UserConfig;
    use crate::lattice::Boundary;
    use crate::model::CPM;

    fn two_type_config(seed: u64) -> crate::config::ResolvedConfig<2> {
        UserConfig::<2> {
            grid: Some([10, 10]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.0,
                    lambda_interface: 0.3,
                    lambda_act: 0.0,
                    max_act: 0,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 9,
                    target_interface: 20,
                    lambda_volume: 1.5,
                    lambda_interface: 0.4,
                    lambda_act: 0.0,
                    max_act: 0,
                },
            ],
            cell_counts: vec![1, 1],
            adhesion: Some(vec![
                vec![0.0, 2.0, 3.0],
                vec![2.0, 1.0, 4.0],
                vec![3.0, 4.0, 1.5],
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

    /// Two 3x3 blocks side by side, with interface counts recomputed from
    /// scratch so the fixture carries a real, nonzero interface state
    /// rather than the all-zero one `CPM::new` starts with.
    fn place_two_blocks(cpm: &mut CPM<2>) {
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
        let mut interface = vec![0u32; 2];
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
                    interface[crate::state::index_of(cell)] += 1;
                }
            }
        });
        cpm.state.interface = interface;
    }

    fn ctx_at(cpm: &CPM<2>, target: [usize; 2], source: [usize; 2]) -> CopyContext<'_, 2> {
        let target_flat = cpm.state.lattice.flat_index(target);
        let source_flat = cpm.state.lattice.flat_index(source);
        CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id: cpm.state.lattice.get(target_flat),
            gaining_id: cpm.state.lattice.get(source_flat),
            mcs: 0,
        }
    }

    fn assert_matches_sequential(cpm: &CPM<2>, ctx: &CopyContext<'_, 2>) {
        let sequential_total = cpm.terms.total_delta(ctx);
        let sequential_pair = cpm.terms.interface.interface_deltas(ctx);
        let fused = cpm.terms.fused_conservative_delta(ctx);
        assert_eq!(
            fused.total, sequential_total,
            "fused total must match total_delta exactly"
        );
        assert_eq!(
            fused.interface_deltas, sequential_pair,
            "fused interface pair must match interface_deltas exactly"
        );
    }

    #[test]
    fn fused_matches_sequential_on_a_hand_built_boundary_move() {
        let mut cpm = CPM::new(two_type_config(1)).unwrap();
        place_two_blocks(&mut cpm);
        // (3,2) is cell 1's rightmost column, adjacent to cell 2 at (4,2).
        let ctx = ctx_at(&cpm, [3, 2], [4, 2]);
        assert_matches_sequential(&cpm, &ctx);
    }

    #[test]
    fn fused_matches_sequential_across_random_boundary_moves() {
        use rand::Rng as _;
        let mut cpm = CPM::new(two_type_config(2)).unwrap();
        place_two_blocks(&mut cpm);
        let mut rng = crate::rng::rng_from_seed(99);

        // Every site along the shared boundary, proposed in both
        // directions.
        let candidates: [([usize; 2], [usize; 2]); 6] = [
            ([3, 1], [4, 1]),
            ([3, 2], [4, 2]),
            ([3, 3], [4, 3]),
            ([4, 1], [3, 1]),
            ([4, 2], [3, 2]),
            ([4, 3], [3, 3]),
        ];

        for _ in 0..200 {
            let (target, source) = candidates[rng.gen_range(0..candidates.len())];
            let ctx = ctx_at(&cpm, target, source);
            assert_matches_sequential(&cpm, &ctx);
        }
    }

    /// Isolated pricing-cost comparison, same methodology and fixture as
    /// `monte_carlo::tests::diag_total_delta_vs_commit_cost_3d`: harvest
    /// real, non-null, connectivity-passing proposals from a relaxed
    /// confluent 3D fixture, then time `total_delta` and
    /// `fused_conservative_delta` over the same harvested set. Manual
    /// `Instant` timing rather than Criterion, for the same reason as the
    /// existing diagnostic — `pub(crate)` internals this needs aren't
    /// reachable from an external bench crate. Run with `--release --
    /// --ignored --nocapture`.
    #[test]
    #[ignore = "diagnostic only: run with --release -- --ignored --nocapture"]
    fn diag_fused_vs_sequential_pricing_cost_3d() {
        use crate::config::UserConfig;
        use crate::connectivity::preserves_topology;
        use crate::lattice::Boundary;
        use rand::Rng as _;
        use std::time::Instant;

        let config = UserConfig::<3> {
            grid: Some([40, 40, 40]),
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
        crate::monte_carlo::run_mcs(&mut cpm, 5);

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

        let mut sequential_sum = 0.0;
        let sequential_start = Instant::now();
        for &(target_flat, source_flat, losing_id, gaining_id) in &proposals {
            let ctx = ctx_from_parts(&cpm, target_flat, source_flat, losing_id, gaining_id);
            sequential_sum += cpm.terms.total_delta(&ctx);
        }
        let sequential_elapsed = sequential_start.elapsed();
        std::hint::black_box(sequential_sum);

        let mut fused_sum = 0.0;
        let fused_start = Instant::now();
        for &(target_flat, source_flat, losing_id, gaining_id) in &proposals {
            let ctx = ctx_from_parts(&cpm, target_flat, source_flat, losing_id, gaining_id);
            fused_sum += cpm.terms.fused_conservative_delta(&ctx).total;
        }
        let fused_elapsed = fused_start.elapsed();
        std::hint::black_box(fused_sum);

        eprintln!(
            "total_delta (sequential): {:.1} ns/call ({} calls in {:?})",
            sequential_elapsed.as_nanos() as f64 / proposals.len() as f64,
            proposals.len(),
            sequential_elapsed
        );
        eprintln!(
            "fused_conservative_delta: {:.1} ns/call ({} calls in {:?})",
            fused_elapsed.as_nanos() as f64 / proposals.len() as f64,
            proposals.len(),
            fused_elapsed
        );
    }

    fn ctx_from_parts(
        cpm: &CPM<3>,
        target_flat: usize,
        source_flat: usize,
        losing_id: crate::lattice::CellId,
        gaining_id: crate::lattice::CellId,
    ) -> CopyContext<'_, 3> {
        CopyContext {
            lattice: &cpm.state.lattice,
            state: &cpm.state,
            target_flat,
            source_flat,
            losing_id,
            gaining_id,
            mcs: cpm.mcs,
        }
    }
}

/// Tests for the Act term: the optimised `delta` checked against an
/// independent, deliberately differently-coded reference implementation
/// across a range of scenarios.
#[cfg(test)]
mod act_tests {
    use super::*;
    use crate::cell::Cell;
    use crate::lattice::{Boundary, Lattice};
    use crate::neighborhood::Stencil;
    use crate::state::State;
    use std::collections::HashMap;

    const GRID: usize = 6;

    fn von_neumann_offsets(lattice: &Lattice<2>) -> Vec<isize> {
        crate::lattice::offset_table(&Stencil::<2>::von_neumann(), lattice.strides())
    }

    /// Two cells on a `boundary`-bounded 6x6 grid: cell 1 at
    /// `(1,1),(1,2),(2,1)`, cell 2 at `(2,2),(2,3),(3,2)` — chosen so
    /// `target=(2,1)` (cell 1) and `source=(2,2)` (cell 2) are
    /// von-Neumann-adjacent, and each site's own von-Neumann neighbourhood
    /// is a genuine mix of same-cell, other-cell and medium sites (the
    /// same-cell restriction is only actually exercised if some neighbours
    /// get filtered out).
    fn two_cell_state(boundary: Boundary) -> State<2> {
        let mut lattice = Lattice::<2>::new([GRID, GRID], boundary);
        for (r, c) in [(1, 1), (1, 2), (2, 1)] {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, 1);
        }
        for (r, c) in [(2, 2), (2, 3), (3, 2)] {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, 2);
        }
        if boundary == Boundary::Periodic {
            lattice.refresh_periodic_halo();
        }
        let cells = vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ];
        State::with_cells(lattice, cells)
    }

    fn act_term(lattice: &Lattice<2>, lambda_act: Vec<f64>, max_act: Vec<u32>) -> ActTerm {
        let n_padded: usize = lattice.padded_dims().iter().product();
        ActTerm {
            lambda_act,
            max_act,
            copy_offsets: von_neumann_offsets(lattice),
            birth_mcs: vec![0u32; n_padded],
        }
    }

    /// Sets `birth_mcs` at `coord` consistently on both the production
    /// `ActTerm` (flat-indexed) and the reference's own coordinate-indexed
    /// map, so a test can describe one scenario once.
    fn set_birth(
        lattice: &Lattice<2>,
        act: &mut ActTerm,
        reference: &mut HashMap<[usize; 2], u32>,
        coord: [usize; 2],
        value: u32,
    ) {
        act.birth_mcs[lattice.flat_index(coord)] = value;
        reference.insert(coord, value);
    }

    /// Independent reference for `delta_H_Act` — deliberately coded a
    /// different way than [`ActTerm`]: coordinate-space von-Neumann deltas
    /// and explicit modular arithmetic for periodic wraparound, rather than
    /// `ActTerm`'s flat-index/offset-delta/halo-canonicalisation machinery.
    /// If the two ever disagree, this test is supposed to catch it —
    /// reusing `ActTerm`'s own code to also generate the reference would
    /// defeat the point (mirrors `exact_boltzmann.rs`'s rationale for its
    /// independent `h_two_cell`).
    fn reference_act_delta(
        state: &State<2>,
        birth_mcs: &HashMap<[usize; 2], u32>,
        mcs: u64,
        target: [usize; 2],
        source: [usize; 2],
        lambda_act: &[f64],
        max_act: &[u32],
    ) -> f64 {
        let lattice = &state.lattice;
        let dims = lattice.dims();
        let periodic = lattice.boundary() == Boundary::Periodic;
        let losing_id = lattice.get(lattice.flat_index(target));
        let gaining_id = lattice.get(lattice.flat_index(source));
        let Some(gaining_type) = state.type_of(gaining_id) else {
            return 0.0;
        };
        let max = max_act[gaining_type];
        let lambda = lambda_act[gaining_type];

        // (owner, birth_mcs) for a site, given its owner is already known —
        // avoids a second lattice lookup at call sites that already have it
        // (namely the out-of-grid Fixed-boundary case, permanently medium).
        let owner_and_birth = |coord: [usize; 2]| -> (CellId, u32) {
            let owner = lattice.get(lattice.flat_index(coord));
            (owner, *birth_mcs.get(&coord).unwrap_or(&0))
        };

        let geometric_mean = |site: [usize; 2], relevant_id: CellId| -> f64 {
            let mut members: Vec<(CellId, u32)> = vec![owner_and_birth(site)];
            for &(dr, dc) in &[(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                let raw_r = site[0] as i64 + dr;
                let raw_c = site[1] as i64 + dc;
                let out_of_bounds =
                    raw_r < 0 || raw_r >= dims[0] as i64 || raw_c < 0 || raw_c >= dims[1] as i64;
                members.push(if periodic {
                    let r = raw_r.rem_euclid(dims[0] as i64) as usize;
                    let c = raw_c.rem_euclid(dims[1] as i64) as usize;
                    owner_and_birth([r, c])
                } else if out_of_bounds {
                    (0, 0) // Fixed halo: permanently medium, never written
                } else {
                    owner_and_birth([raw_r as usize, raw_c as usize])
                });
            }
            let mut product = 1.0f64;
            let mut count = 0u32;
            for (owner, birth) in members {
                if owner != relevant_id {
                    continue;
                }
                let elapsed = mcs.saturating_sub(birth as u64);
                product *= (max as u64).saturating_sub(elapsed) as f64;
                count += 1;
            }
            if count == 0 {
                return 0.0;
            }
            product.powf(1.0 / count as f64)
        };

        let gm_src = geometric_mean(source, gaining_id);
        let gm_tgt = geometric_mean(target, losing_id);
        -(lambda / max as f64) * (gm_src - gm_tgt)
    }

    const LAMBDA_ACT: [f64; 2] = [2.0, 3.0];
    const MAX_ACT: [u32; 2] = [10, 20];
    const MCS: u64 = 20;

    fn hand_built_ctx(state: &State<2>) -> ([usize; 2], [usize; 2], CopyContext<'_, 2>) {
        let target_coord = [2usize, 1usize];
        let source_coord = [2usize, 2usize];
        let target_flat = state.lattice.flat_index(target_coord);
        let source_flat = state.lattice.flat_index(source_coord);
        let ctx = CopyContext {
            lattice: &state.lattice,
            state,
            target_flat,
            source_flat,
            losing_id: state.lattice.get(target_flat),
            gaining_id: state.lattice.get(source_flat),
            mcs: MCS,
        };
        (target_coord, source_coord, ctx)
    }

    #[test]
    fn optimized_delta_matches_independent_reference_hand_built() {
        let state = two_cell_state(Boundary::Fixed);
        let mut act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let mut reference_birth = HashMap::new();
        for (coord, value) in [
            ([2, 1], 15), // target itself (cell 1)
            ([1, 1], 12), // cell 1, part of GM_tgt
            ([2, 2], 18), // source itself (cell 2)
            ([3, 2], 10), // cell 2, part of GM_src
            ([2, 3], 19), // cell 2, part of GM_src
        ] {
            set_birth(&state.lattice, &mut act, &mut reference_birth, coord, value);
        }

        let (target_coord, source_coord, ctx) = hand_built_ctx(&state);
        let optimized = act.delta(&ctx);
        let reference = reference_act_delta(
            &state,
            &reference_birth,
            MCS,
            target_coord,
            source_coord,
            &LAMBDA_ACT,
            &MAX_ACT,
        );
        assert!(
            (optimized - reference).abs() < 1e-9,
            "optimized {optimized} vs reference {reference}"
        );
        // Not just "they happen to agree" — confirm this scenario actually
        // exercises a nontrivial computation on both sides.
        assert!(optimized.abs() > 1e-6, "delta should be nonzero here");
    }

    #[test]
    fn zero_activity_site_forces_geometric_mean_to_zero() {
        let state = two_cell_state(Boundary::Fixed);
        let mut act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let mut reference_birth = HashMap::new();
        // (3,2) is fully decayed by MCS (elapsed = MCS - 0 = 20 >= max_act[1] = 20).
        for (coord, value) in [([2, 2], 18), ([3, 2], 0), ([2, 3], 19)] {
            set_birth(&state.lattice, &mut act, &mut reference_birth, coord, value);
        }
        let (_, source_coord, ctx) = hand_built_ctx(&state);
        let gm_src = act.geometric_mean(&ctx, ctx.source_flat, ctx.gaining_id, MAX_ACT[1]);
        assert_eq!(
            gm_src, 0.0,
            "one zero-activity retained site must zero the whole GM"
        );
        let _ = source_coord; // used only for reference cross-check above
    }

    #[test]
    fn commit_resets_birth_mcs_on_cell_to_cell_transfer() {
        let state = two_cell_state(Boundary::Fixed);
        let mut act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let (_, _, ctx) = hand_built_ctx(&state);
        assert_ne!(ctx.losing_id, 0);
        assert_ne!(ctx.gaining_id, 0);
        assert_ne!(ctx.losing_id, ctx.gaining_id); // genuinely cell-to-cell

        assert_eq!(act.birth_mcs[ctx.target_flat], 0);
        act.commit(&ctx);
        assert_eq!(
            act.birth_mcs[ctx.target_flat], MCS as u32,
            "birth_mcs must reset on a cell-to-cell transfer, not only medium-to-cell"
        );
    }

    #[test]
    fn gaining_medium_zeroes_delta() {
        // Medium invading: target currently cell 1, source currently
        // medium — the reverse of the hand-built scenario's direction.
        let state = two_cell_state(Boundary::Fixed);
        let act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let target_flat = state.lattice.flat_index([1, 1]); // cell 1
        let source_flat = state.lattice.flat_index([0, 1]); // medium
        let ctx = CopyContext {
            lattice: &state.lattice,
            state: &state,
            target_flat,
            source_flat,
            losing_id: state.lattice.get(target_flat),
            gaining_id: state.lattice.get(source_flat),
            mcs: MCS,
        };
        assert_eq!(ctx.gaining_id, 0);
        assert_eq!(act.delta(&ctx), 0.0);
    }

    #[test]
    fn losing_medium_still_computes_gm_tgt_from_decayed_medium_activity() {
        // A cell growing into empty medium — gaining_id is real, losing_id
        // is medium. GM_tgt must still be computed (from medium's own
        // birth_mcs), not skipped outright.
        //
        // `target` must be fully interior (every von-Neumann neighbour
        // still on-grid): a target touching the Fixed boundary has an
        // off-grid neighbour that's *permanently* medium with birth_mcs 0
        // forever, which alone zeroes the whole geometric mean regardless
        // of every other site's activity — that would test "an off-grid
        // halo site is always zero," not "medium's own real birth_mcs
        // contributes," which is the property this test is for.
        let mut lattice = Lattice::<2>::new([GRID, GRID], Boundary::Fixed);
        let cell1_flat = lattice.flat_index([2, 3]);
        lattice.set(cell1_flat, 1);
        let cells = vec![Cell {
            id: 1,
            type_index: 0,
        }];
        let state = State::with_cells(lattice, cells);

        let mut act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let mut reference_birth = HashMap::new();
        let target_coord = [2, 2]; // medium, fully interior
        let source_coord = [2, 3]; // cell 1
        for (coord, value) in [
            ([2, 2], 18), // target itself
            ([1, 2], 17), // medium, part of GM_tgt
            ([3, 2], 16), // medium, part of GM_tgt
            ([2, 1], 15), // medium, part of GM_tgt
        ] {
            set_birth(&state.lattice, &mut act, &mut reference_birth, coord, value);
        }

        let target_flat = state.lattice.flat_index(target_coord);
        let source_flat = state.lattice.flat_index(source_coord);
        let ctx = CopyContext {
            lattice: &state.lattice,
            state: &state,
            target_flat,
            source_flat,
            losing_id: state.lattice.get(target_flat),
            gaining_id: state.lattice.get(source_flat),
            mcs: MCS,
        };
        assert_eq!(ctx.losing_id, 0);
        assert_ne!(ctx.gaining_id, 0);

        let optimized = act.delta(&ctx);
        let reference = reference_act_delta(
            &state,
            &reference_birth,
            MCS,
            target_coord,
            source_coord,
            &LAMBDA_ACT,
            &MAX_ACT,
        );
        assert!(
            (optimized - reference).abs() < 1e-9,
            "optimized {optimized} vs reference {reference}"
        );

        // And this must differ from what you'd get if medium's activity
        // were (wrongly) treated as always 0 / GM_tgt skipped outright: all
        // four retained medium sites have recent, nonzero birth_mcs, so
        // GM_tgt should be strictly positive, which a "medium has no
        // activity" implementation would not produce.
        let gm_tgt = act.geometric_mean(&ctx, ctx.target_flat, ctx.losing_id, MAX_ACT[0]);
        assert!(
            gm_tgt > 0.0,
            "medium's own decayed activity must contribute: {gm_tgt}"
        );
    }

    #[test]
    fn periodic_boundary_reads_true_interior_site_through_halo() {
        // cell 1 at (0,1),(1,1) plus a deliberate wraparound partner at
        // (5,1) — the row-0 "up" neighbour of (0,1) under periodic
        // wrapping. cell 2 at (0,2),(1,2),(0,3).
        let mut lattice = Lattice::<2>::new([GRID, GRID], Boundary::Periodic);
        for (r, c) in [(0, 1), (1, 1), (5, 1)] {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, 1);
        }
        for (r, c) in [(0, 2), (1, 2), (0, 3)] {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, 2);
        }
        lattice.refresh_periodic_halo();
        let cells = vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ];
        let state = State::with_cells(lattice, cells);

        let mut act = act_term(&state.lattice, LAMBDA_ACT.to_vec(), MAX_ACT.to_vec());
        let mut reference_birth = HashMap::new();
        for (coord, value) in [
            ([0, 1], 15), // target itself
            ([1, 1], 12), // cell 1
            ([5, 1], 8),  // cell 1, only reachable by wrapping "up" from (0,1)
            ([0, 2], 18), // source itself (cell 2)
            ([1, 2], 10), // cell 2
            ([0, 3], 19), // cell 2
        ] {
            set_birth(&state.lattice, &mut act, &mut reference_birth, coord, value);
        }

        let target_coord = [0, 1];
        let source_coord = [0, 2];
        let target_flat = state.lattice.flat_index(target_coord);
        let source_flat = state.lattice.flat_index(source_coord);
        let ctx = CopyContext {
            lattice: &state.lattice,
            state: &state,
            target_flat,
            source_flat,
            losing_id: state.lattice.get(target_flat),
            gaining_id: state.lattice.get(source_flat),
            mcs: MCS,
        };

        let optimized = act.delta(&ctx);
        let reference = reference_act_delta(
            &state,
            &reference_birth,
            MCS,
            target_coord,
            source_coord,
            &LAMBDA_ACT,
            &MAX_ACT,
        );
        assert!(
            (optimized - reference).abs() < 1e-9,
            "optimized {optimized} vs reference {reference} — periodic wraparound mismatch"
        );

        // And confirm the wrapped site actually mattered: without it, GM_tgt
        // would be a two-member (not three-member) geometric mean, giving a
        // different value. Compute it directly with the wrap partner's
        // birth_mcs reset to 0 (fully decayed) and check the delta moves.
        act.birth_mcs[state.lattice.flat_index([5, 1])] = 0;
        let with_decayed_wrap_partner = act.delta(&ctx);
        assert!(
            (with_decayed_wrap_partner - optimized).abs() > 1e-6,
            "changing the wrapped site's activity should change the delta \
             if it's genuinely being read through the halo"
        );
    }
}
