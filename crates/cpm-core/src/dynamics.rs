//! Incremental bookkeeping: applies an *already-accepted* copy to
//! [`State`] — the lattice cell id, `volume`/`interface` for the losing and
//! gaining cells only, and unwrapped position sums. Energy deltas are
//! computed *before* this runs (`energy.rs`, reading the pre-move state);
//! this module only ever applies a move that has already been decided.

use crate::lattice::{Boundary, CellId};
use crate::state::{index_of, State};

/// Apply an accepted copy. `interface_deltas` (`(delta_i_losing,
/// delta_i_gaining)`) must come from
/// [`crate::energy::InterfaceTerm::interface_deltas`] called on the
/// *pre-move* state — the same computation used to price the move, so the
/// booked `I_c` change can never drift from the priced one. Deliberately
/// takes plain values rather than a `CopyContext` (which borrows the state
/// this function needs to mutate): the caller computes the deltas while its
/// own `CopyContext` is still alive, then drops it before calling here.
/// `energy_offsets`/`energy_weights` are the same energy-neighbourhood
/// stencil `interface_deltas` walked to produce those deltas — reused here,
/// on the still-pre-move lattice, to update [`State::contact`] with the
/// same per-neighbour information rather than walking the neighbourhood a
/// second time with a different derivation.
pub fn commit<const D: usize>(
    state: &mut State<D>,
    target_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
    interface_deltas: (f64, f64),
    energy_offsets: &[isize],
    energy_weights: &[f64],
) {
    let (delta_i_losing, delta_i_gaining) = interface_deltas;

    update_contact(
        state,
        target_flat,
        losing_id,
        gaining_id,
        energy_offsets,
        energy_weights,
    );

    state.lattice.set(target_flat, gaining_id);

    if losing_id != 0 {
        let i = index_of(losing_id);
        state.volume[i] -= 1;
        state.interface[i] = apply_delta(state.interface[i], delta_i_losing);
    }
    if gaining_id != 0 {
        let i = index_of(gaining_id);
        state.volume[i] += 1;
        state.interface[i] = apply_delta(state.interface[i], delta_i_gaining);
    }

    update_position_sums(state, target_flat, losing_id, gaining_id);
    refresh_periodic_halo(state, target_flat);
}

/// Update the pairwise contact graph, reading the *pre-move* lattice
/// (called before `state.lattice.set` below mutates `target_flat`). Mirrors
/// `InterfaceTerm::interface_deltas`'s own per-neighbour condition
/// exactly, just applied to a pair key instead of a signed per-cell delta:
/// for each energy-neighbour `n` (weight `w`) of `target_flat`, the
/// `(losing_id, owner(n))` pair loses `w` whenever `owner(n) != losing_id`
/// (that contact existed before the move and `target_flat` is leaving
/// `losing_id`), and the `(gaining_id, owner(n))` pair gains `w` whenever
/// `owner(n) != gaining_id` (that contact now exists, since `target_flat`
/// is joining `gaining_id`). `energy_weights` are always exactly `1.0` in
/// any config that passed validation (`config.rs` rejects weighted energy
/// neighbourhoods) so every contribution here is a whole count, matching
/// `apply_delta`'s identical assumption for the per-cell interface.
pub(crate) fn update_contact<const D: usize>(
    state: &mut State<D>,
    target_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
    energy_offsets: &[isize],
    energy_weights: &[f64],
) {
    if losing_id == gaining_id {
        return; // early-out at the call site already prevents this in practice
    }
    for (&offset, &weight) in energy_offsets.iter().zip(energy_weights) {
        debug_assert_eq!(
            weight, 1.0,
            "weighted energy neighbourhoods are rejected at config validation"
        );
        let neighbour_flat = state.lattice.neighbour(target_flat, offset);
        let neighbour_owner = state.lattice.get(neighbour_flat);
        if neighbour_owner != losing_id {
            state.contact.decrement(losing_id, neighbour_owner);
        }
        if neighbour_owner != gaining_id {
            state.contact.increment(gaining_id, neighbour_owner);
        }
    }
}

/// `delta` is always integer-valued in practice (weighted energy
/// neighbourhoods are rejected at config-validation time), carried as `f64`
/// only because that's what the energy-delta math uses. Rounds rather than
/// truncates so an accumulated floating-point residue (there shouldn't be
/// one, since every weight is exactly `1.0`) can't silently bias the count
/// downward.
fn apply_delta(current: u32, delta: f64) -> u32 {
    let updated = current as i64 + delta.round() as i64;
    debug_assert!(updated >= 0, "interface count went negative: {updated}");
    updated as u32
}

/// Maintain unwrapped position sums incrementally, so centroids and MSD
/// survive periodic wrapping. On gaining a site, choose the periodic image
/// nearest the cell's *current* centroid (approximated here by its existing
/// sum, before this gain is added — consistent since the sum already
/// reflects every previously-gained site's nearest image).
///
/// Also maintains `sum_position_outer` (the gyration tensor accumulator),
/// reusing exactly the same `image` coordinate computed here for
/// `sum_position`, on the loss side and the gain side alike: a gained
/// site's outer-product contribution is `image ⊗ image`, matching what was
/// actually added to `sum_position` for that site, and a lost site's is
/// removed using its raw (unwrapped-frame) coordinate, matching
/// `sum_position`'s own loss-side convention.
fn update_position_sums<const D: usize>(
    state: &mut State<D>,
    target_flat: usize,
    losing_id: CellId,
    gaining_id: CellId,
) {
    let dims = state.lattice.dims();
    let coord = state.lattice.interior_coord_of(target_flat);

    if losing_id != 0 {
        let i = index_of(losing_id);
        for (d, slot) in state.sum_position[i].iter_mut().enumerate() {
            *slot -= coord[d] as i64;
        }
        for d in 0..D {
            for e in 0..D {
                state.sum_position_outer[i][d][e] -= coord[d] as i64 * coord[e] as i64;
            }
        }
    }
    if gaining_id != 0 {
        let i = index_of(gaining_id);
        let volume_before_gain = state.volume[i].saturating_sub(1).max(1);
        let centroid: [f64; D] =
            std::array::from_fn(|d| state.sum_position[i][d] as f64 / volume_before_gain as f64);
        let boundary = state.lattice.boundary();
        let mut image = [0i64; D];
        for (d, slot) in state.sum_position[i].iter_mut().enumerate() {
            image[d] = nearest_periodic_image(coord[d] as i64, centroid[d], dims[d], boundary);
            // Checked directly rather than via a reconstructed bounding box
            // (which incremental bookkeeping can't maintain exactly under
            // *losses*, only gains): if even the *nearest* image of the
            // newly-gained site is still far from the running centroid, the
            // cell's extent along this axis is approaching half the box,
            // and nearest-image selection is becoming ambiguous.
            // Threshold at 0.48 rather than a tighter fraction: this proxy
            // is inherently noisier than a true bounding-box extent
            // (small-N centroids fluctuate, and nothing here constrains
            // cell shape), so a tighter bound false-positives on ordinary,
            // if elongated, growth well before any real ambiguity. 0.48
            // still catches genuine box-spanning pathologies close to the
            // true L/2 ambiguity point.
            debug_assert!(
                boundary != Boundary::Periodic || dims[d] < 4 || {
                    let distance = (image[d] as f64 - centroid[d]).abs();
                    distance < dims[d] as f64 * 0.48
                },
                "cell {gaining_id} extent along axis {d} approaches half the box size: \
                 unwrapped position tracking may be ambiguous"
            );
            *slot += image[d];
        }
        for d in 0..D {
            for e in 0..D {
                state.sum_position_outer[i][d][e] += image[d] * image[e];
            }
        }
    }
}

/// The periodic image of `coord` (along one axis) nearest `centroid`. For a
/// `Fixed` boundary there is only one image (no wrapping).
fn nearest_periodic_image(coord: i64, centroid: f64, extent: usize, boundary: Boundary) -> i64 {
    if boundary == Boundary::Fixed {
        return coord;
    }
    let extent = extent as i64;
    let mut best = coord;
    let mut best_distance = (coord as f64 - centroid).abs();
    for image in [coord - extent, coord + extent] {
        let distance = (image as f64 - centroid).abs();
        if distance < best_distance {
            best = image;
            best_distance = distance;
        }
    }
    best
}

/// The halo mirrors the opposite face for `Periodic` boundaries, refreshed
/// whenever an accepted copy touches a boundary site. Rare, O(1) per event.
fn refresh_periodic_halo<const D: usize>(state: &mut State<D>, target_flat: usize) {
    if state.lattice.boundary() != Boundary::Periodic {
        return;
    }
    if state.lattice.is_interior(target_flat) {
        let dims = state.lattice.dims();
        let coord = state.lattice.interior_coord_of(target_flat);
        let on_boundary = (0..D).any(|d| coord[d] == 0 || coord[d] == dims[d] - 1);
        if !on_boundary {
            return;
        }
    }
    // A touched interior boundary site (or, defensively, a touched halo
    // site) means at least one mirrored halo cell is now stale. Refresh the
    // whole halo: correctness over cleverness here, since this path is rare
    // by construction (only boundary-adjacent sites trigger it) and O(1)
    // amortised per accepted move.
    state.lattice.refresh_periodic_halo();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::config::test_support::two_type_config;
    use crate::energy::Terms;
    use crate::lattice::Lattice;

    fn setup() -> (State<2>, Terms) {
        let config = two_type_config();
        let lattice = Lattice::<2>::new(config.grid, config.boundary);
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
        let mut state = State::with_cells(lattice, cells);
        // Two adjacent single-site cells.
        let a = state.lattice.flat_index([1, 1]);
        let b = state.lattice.flat_index([2, 1]);
        state.lattice.set(a, 1);
        state.lattice.set(b, 2);
        state.volume = vec![1, 1];
        // Each single-site cell has 8 Moore neighbours; one is the other
        // cell, the other 7 are medium — all 8 differ from the cell itself.
        state.interface = vec![8, 8];
        state.contact = crate::state::recompute_contact(
            &state.lattice,
            &energy_offsets(&state),
            state.cells.len(),
        );

        let copy_offsets = crate::lattice::offset_table(
            &crate::neighborhood::Stencil::<2>::von_neumann(),
            state.lattice.strides(),
        );
        let terms = Terms::from_config(
            &config,
            &energy_offsets(&state),
            &energy_weights(&state),
            &copy_offsets,
        );
        (state, terms)
    }

    fn energy_offsets(state: &State<2>) -> Vec<isize> {
        crate::lattice::offset_table(
            &crate::neighborhood::Stencil::<2>::moore_2d(),
            state.lattice.strides(),
        )
    }

    fn energy_weights(_state: &State<2>) -> Vec<f64> {
        crate::neighborhood::Stencil::<2>::moore_2d().weights
    }

    #[test]
    fn commit_moves_the_site_and_updates_volumes() {
        let (mut state, terms) = setup();
        let target = state.lattice.flat_index([1, 1]); // currently cell 1
        let source = state.lattice.flat_index([2, 1]); // currently cell 2

        let (delta_losing, delta_gaining) = {
            let ctx = crate::energy::CopyContext {
                lattice: &state.lattice,
                state: &state,
                target_flat: target,
                source_flat: source,
                losing_id: 1,
                gaining_id: 2,
                mcs: 0,
            };
            terms.interface.interface_deltas(&ctx)
        };

        let offsets = energy_offsets(&state);
        let weights = energy_weights(&state);
        commit(
            &mut state,
            target,
            1,
            2,
            (delta_losing, delta_gaining),
            &offsets,
            &weights,
        );

        assert_eq!(state.lattice.get(target), 2);
        assert_eq!(state.volume[index_of(1)], 0);
        assert_eq!(state.volume[index_of(2)], 2);
    }
}
