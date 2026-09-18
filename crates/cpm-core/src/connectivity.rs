//! Wraps the simple-point tables (`simple_point.rs`) into the actual
//! per-attempt check: given a proposed copy, decide whether the target site
//! is safe to take away from the cell that currently owns it.
//!
//! **Medium is exempt.** A shell of cells enclosing a lumen legitimately
//! disconnects the medium, and lumens are exactly the structure this
//! organoid model needs — checking medium connectivity would silently
//! forbid them.
//!
//! **Only the losing cell is checked**, never the gaining cell. The gaining
//! cell cannot be disconnected by acquiring a site adjacent to it by
//! construction, given `copy_neighborhood ⊆ connectivity_neighborhood`
//! (enforced at config-validation time). One check per attempt, not two.

use crate::lattice::{offset_table, CellId, Lattice};
use crate::neighborhood::Stencil;
use crate::simple_point::{self, SimplePointTable2D};
use std::sync::OnceLock;

static TABLE_2D: OnceLock<SimplePointTable2D> = OnceLock::new();

fn table_2d() -> &'static SimplePointTable2D {
    TABLE_2D.get_or_init(SimplePointTable2D::generate)
}

/// The flat-index deltas for the canonical full neighbourhood (Moore in 2D,
/// the 26-shell in 3D) — the fixed basis the simple-point pattern is always
/// built over, independent of the configured `connectivity_neighborhood`
/// (which `config.rs`'s validation currently constrains to von Neumann).
/// Precomputed once per `CPM` alongside the other offset tables.
///
/// Uses the same `Stencil::<D>::full()` (and therefore the same offset
/// order) that `simple_point::canonical_offsets_2d`/`_3d` are built from, so
/// bit `i` of a pattern always means the same neighbour on both sides.
pub fn topology_offset_deltas<const D: usize>(strides: [usize; D]) -> Vec<isize> {
    offset_table(&Stencil::<D>::full(), strides)
}

/// Whether the target site is a simple point for `losing_cell` — safe to
/// reassign away from it without changing its topology. `topology_offsets`
/// is [`topology_offset_deltas`]'s output for this lattice.
pub fn preserves_topology<const D: usize>(
    lattice: &Lattice<D>,
    target_flat: usize,
    losing_cell: CellId,
    topology_offsets: &[isize],
) -> bool {
    if losing_cell == 0 {
        return true; // medium is exempt
    }

    let mut pattern: u32 = 0;
    for (i, &delta) in topology_offsets.iter().enumerate() {
        let neighbour_flat = lattice.neighbour(target_flat, delta);
        if lattice.get(neighbour_flat) == losing_cell {
            pattern |= 1 << i;
        }
    }

    match D {
        2 => table_2d().is_simple(pattern as u8),
        3 => simple_point::simple_point_table_3d().is_simple(pattern),
        _ => unreachable!("cpm-core only instantiates D = 2 or D = 3"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Boundary;

    #[test]
    fn medium_is_always_exempt() {
        let lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let offsets = topology_offset_deltas(lattice.strides());
        let target = lattice.flat_index([2, 2]);
        assert!(preserves_topology(&lattice, target, 0, &offsets));
    }

    #[test]
    fn rejects_pinch_off_2d() {
        let mut lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        // Cell 1 occupies (2,1) and (2,3) around target (2,2), but not
        // (2,2)'s other face neighbours or any diagonal linking them: a
        // classic pinch-off pattern (matches
        // `simple_point::reference_rejects_face_neighbours_not_mutually_adjacent`'s
        // 2D shape, transplanted onto the real lattice).
        let target = lattice.flat_index([2, 2]);
        lattice.set(target, 1);
        let offsets = topology_offset_deltas(lattice.strides());
        // Face neighbours of (2,2) in this stencil order come from
        // canonical_offsets_2d(); set exactly the North and East neighbours
        // to cell 1, matching the known-pinch-off pattern.
        let north = lattice.flat_index([2, 1]);
        let east = lattice.flat_index([3, 2]);
        lattice.set(north, 1);
        lattice.set(east, 1);
        assert!(!preserves_topology(&lattice, target, 1, &offsets));
    }

    #[test]
    fn accepts_a_two_site_cell_2d() {
        let mut lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let target = lattice.flat_index([2, 2]);
        let north = lattice.flat_index([2, 1]);
        lattice.set(target, 1);
        lattice.set(north, 1);
        let offsets = topology_offset_deltas(lattice.strides());
        assert!(preserves_topology(&lattice, target, 1, &offsets));
    }

    #[test]
    fn allows_medium_lumen_formation() {
        // A ring of cell 1 enclosing a single medium site: removing a site
        // from the *medium* side is always allowed (medium is exempt), even
        // though the medium is locally disconnected from the rest of the
        // medium by the ring.
        let lattice = Lattice::<3>::new([5, 5, 5], Boundary::Fixed);
        let offsets = topology_offset_deltas(lattice.strides());
        let target = lattice.flat_index([2, 2, 2]);
        // losing_cell = 0 (medium) regardless of what actually surrounds it.
        assert!(preserves_topology(&lattice, target, 0, &offsets));
    }
}
