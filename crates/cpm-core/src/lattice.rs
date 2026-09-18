//! Halo-padded flat lattice storage. A neighbour is reached by adding a
//! precomputed flat `isize` delta to a site's flat index — no modulo
//! arithmetic, no bounds branch, identical code in 2D and 3D.

use crate::neighborhood::Stencil;
use serde::{Deserialize, Serialize};

/// 0 identifies medium. Positive values identify individual cells.
pub type CellId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Boundary {
    /// The halo is permanently medium. A cell touching the wall pays the
    /// normal cell-medium interaction; the boundary interface counts toward
    /// that cell's interface measure.
    Fixed,
    /// The halo mirrors the opposite face, refreshed whenever an accepted
    /// copy touches a boundary site. Refreshing the halo is Monte Carlo
    /// machinery; this lattice only records the choice.
    Periodic,
}

/// Row-major (C-order) strides over `padded_dims`: the last axis is fastest.
fn strides_for<const D: usize>(padded_dims: [usize; D]) -> [usize; D] {
    let mut strides = [1usize; D];
    for d in (0..D.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * padded_dims[d + 1];
    }
    strides
}

/// Precompute the flat-index delta for each stencil offset, given the
/// lattice's padded strides. This is the table the hot loop indexes into;
/// nothing here does bounds checking, which is the point of the halo.
pub fn offset_table<const D: usize>(stencil: &Stencil<D>, strides: [usize; D]) -> Vec<isize> {
    stencil
        .offsets
        .iter()
        .map(|offset| {
            let mut delta: isize = 0;
            for d in 0..D {
                delta += offset[d] as isize * strides[d] as isize;
            }
            delta
        })
        .collect()
}

/// Visit every interior (unpadded) coordinate exactly once, last axis
/// fastest. Generic over `D` so 2D and 3D share one implementation.
pub fn each_interior_coord<const D: usize>(dims: [usize; D], mut visit: impl FnMut([usize; D])) {
    let total: usize = dims.iter().product();
    if total == 0 {
        return;
    }
    let mut coord = [0usize; D];
    for _ in 0..total {
        visit(coord);
        for d in (0..D).rev() {
            coord[d] += 1;
            if coord[d] < dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}

#[derive(Debug, Clone)]
pub struct Lattice<const D: usize> {
    /// Unpadded (interior) extent along each axis.
    dims: [usize; D],
    /// `dims[d] + 2` along each axis.
    padded_dims: [usize; D],
    /// Row-major strides over `padded_dims`.
    strides: [usize; D],
    boundary: Boundary,
    /// Flat storage, length `product(padded_dims)`. Index 0 everywhere
    /// (medium) at construction; the halo ring stays medium for `Fixed`
    /// boundaries and is refreshed by Monte Carlo dynamics for `Periodic`
    /// ones.
    cells: Vec<CellId>,
}

impl<const D: usize> Lattice<D> {
    /// Callers validate grid dimensions before reaching here —
    /// `ResolvedConfig::validate` is the user-facing `Result`-returning
    /// check that guards against panicking on user input. This is an
    /// internal invariant, hence `debug_assert!` rather than `assert!`.
    pub fn new(dims: [usize; D], boundary: Boundary) -> Self {
        debug_assert!(
            dims.iter().all(|&d| d > 0),
            "every grid dimension must be nonzero, got {dims:?}"
        );
        let mut padded_dims = dims;
        for d in padded_dims.iter_mut() {
            *d += 2;
        }
        let strides = strides_for(padded_dims);
        let total_padded: usize = padded_dims.iter().product();
        Self {
            dims,
            padded_dims,
            strides,
            boundary,
            cells: vec![0; total_padded],
        }
    }

    pub fn dims(&self) -> [usize; D] {
        self.dims
    }

    pub fn padded_dims(&self) -> [usize; D] {
        self.padded_dims
    }

    pub fn strides(&self) -> [usize; D] {
        self.strides
    }

    pub fn boundary(&self) -> Boundary {
        self.boundary
    }

    /// `N_sites` in the Monte Carlo step definition: the unpadded site count.
    pub fn n_sites(&self) -> usize {
        self.dims.iter().product()
    }

    /// Flat index of an interior coordinate (0-based, unpadded). Interior
    /// coordinate `c` lives at padded coordinate `c + 1` along every axis.
    pub fn flat_index(&self, interior_coord: [usize; D]) -> usize {
        debug_assert!(
            interior_coord
                .iter()
                .zip(self.dims.iter())
                .all(|(c, d)| c < d),
            "coordinate {interior_coord:?} out of interior bounds {:?}",
            self.dims
        );
        interior_coord
            .iter()
            .zip(self.strides.iter())
            .map(|(c, s)| (c + 1) * s)
            .sum()
    }

    /// Whether a flat (padded-space) index falls in the halo rather than the
    /// interior. Site selection samples the interior directly, so this is a
    /// diagnostic and testing helper rather than a hot-loop function.
    pub fn is_interior(&self, flat: usize) -> bool {
        let mut remainder = flat;
        for d in 0..D {
            let coord_d = remainder / self.strides[d];
            remainder %= self.strides[d];
            if coord_d == 0 || coord_d == self.padded_dims[d] - 1 {
                return false;
            }
        }
        true
    }

    /// `idx + offsets[k]`. No bounds check: the halo guarantees this stays
    /// in range for any stencil offset applied to an interior site.
    pub fn neighbour(&self, idx: usize, delta: isize) -> usize {
        (idx as isize + delta) as usize
    }

    pub fn get(&self, flat: usize) -> CellId {
        self.cells[flat]
    }

    pub fn set(&mut self, flat: usize, id: CellId) {
        self.cells[flat] = id;
    }

    pub fn each_interior_coord(&self, visit: impl FnMut([usize; D])) {
        each_interior_coord(self.dims, visit);
    }

    /// The padded-space coordinate for a flat index (the inverse of the
    /// stride arithmetic `flat_index`/`neighbour` use).
    fn padded_coord_of(&self, flat: usize) -> [usize; D] {
        let mut remainder = flat;
        let mut coord = [0usize; D];
        for (d, slot) in coord.iter_mut().enumerate() {
            *slot = remainder / self.strides[d];
            remainder %= self.strides[d];
        }
        coord
    }

    /// The interior (unpadded) coordinate for a flat index — the inverse of
    /// [`flat_index`](Self::flat_index). Panics via the subtraction
    /// underflowing if `flat` is actually a halo index; callers that might
    /// pass a halo index should check [`is_interior`](Self::is_interior)
    /// first.
    pub fn interior_coord_of(&self, flat: usize) -> [usize; D] {
        let padded = self.padded_coord_of(flat);
        std::array::from_fn(|d| padded[d] - 1)
    }

    /// The interior coordinate a padded-space coordinate maps to under
    /// periodic wraparound: unchanged if already interior, wrapped to the
    /// opposite edge if in the halo. Used both to refresh the halo and (by
    /// `energy.rs`'s global adhesion sum) to recognise that a halo neighbour
    /// under a periodic boundary is a stand-in for a specific *other*
    /// interior site, not an independent non-site the way a `Fixed`
    /// boundary's halo is.
    fn wrapped_interior_coord(&self, padded_coord: [usize; D]) -> [usize; D] {
        std::array::from_fn(|d| {
            if padded_coord[d] == 0 {
                self.dims[d] - 1
            } else if padded_coord[d] == self.padded_dims[d] - 1 {
                0
            } else {
                padded_coord[d] - 1
            }
        })
    }

    /// The flat index of the interior site a (possibly-halo) flat index
    /// represents under periodic wraparound. Only meaningful for `Periodic`
    /// boundaries — a `Fixed` boundary's halo is permanently medium and
    /// doesn't stand in for any real site.
    pub fn wrapped_interior_flat(&self, flat: usize) -> usize {
        debug_assert_eq!(self.boundary, Boundary::Periodic);
        let padded_coord = self.padded_coord_of(flat);
        self.flat_index(self.wrapped_interior_coord(padded_coord))
    }

    /// For `Periodic` boundaries, the halo mirrors the opposite face.
    /// Refreshes the whole halo from the current interior — correctness
    /// over cleverness (a boring `O(N_sites)` scan) — called only when an
    /// accepted copy actually touches a boundary-adjacent site
    /// (`dynamics.rs`), which keeps it rare in practice. Touching only the
    /// affected halo cells would be a further optimisation if this ever
    /// shows up in profiling.
    pub fn refresh_periodic_halo(&mut self) {
        debug_assert_eq!(self.boundary, Boundary::Periodic);
        let padded_dims = self.padded_dims;
        let total_padded: usize = padded_dims.iter().product();
        for flat in 0..total_padded {
            let padded_coord = self.padded_coord_of(flat);
            let is_halo =
                (0..D).any(|d| padded_coord[d] == 0 || padded_coord[d] == padded_dims[d] - 1);
            if !is_halo {
                continue;
            }
            let source_flat = self.flat_index(self.wrapped_interior_coord(padded_coord));
            self.cells[flat] = self.cells[source_flat];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighborhood::Stencil;

    #[test]
    fn padding_adds_one_site_halo_per_axis() {
        let lat = Lattice::new([3, 4], Boundary::Fixed);
        assert_eq!(lat.dims(), [3, 4]);
        assert_eq!(lat.padded_dims(), [5, 6]);
        assert_eq!(lat.n_sites(), 12);
    }

    #[test]
    fn strides_are_row_major_last_axis_fastest() {
        // padded_dims = [5, 6]: stride along axis 1 is 1, axis 0 is 6.
        let lat = Lattice::new([3, 4], Boundary::Fixed);
        assert_eq!(lat.strides(), [6, 1]);

        // 3D: padded_dims = [5, 6, 7]; strides = [42, 7, 1].
        let lat3 = Lattice::<3>::new([3, 4, 5], Boundary::Fixed);
        assert_eq!(lat3.padded_dims(), [5, 6, 7]);
        assert_eq!(lat3.strides(), [42, 7, 1]);
    }

    #[test]
    fn interior_flat_index_matches_manual_coordinate_walk() {
        let lat = Lattice::new([3, 4], Boundary::Fixed);
        // Interior (0, 0) sits at padded coordinate (1, 1) => 1*6 + 1*1 = 7.
        assert_eq!(lat.flat_index([0, 0]), 7);
        // Interior (2, 3) sits at padded coordinate (3, 4) => 3*6 + 4 = 22.
        assert_eq!(lat.flat_index([2, 3]), 22);
    }

    #[test]
    fn offset_table_matches_manual_neighbour_coordinates() {
        let lat = Lattice::new([3, 4], Boundary::Fixed);
        let stencil = Stencil::<2>::von_neumann();
        let table = offset_table(&stencil, lat.strides());

        let base = lat.flat_index([1, 1]);
        for (offset, delta) in stencil.offsets.iter().zip(table.iter()) {
            let expected_coord = [(1 + offset[0]) as usize, (1 + offset[1]) as usize];
            let expected = lat.flat_index(expected_coord);
            assert_eq!(lat.neighbour(base, *delta), expected);
        }
    }

    #[test]
    fn offset_table_is_position_independent() {
        // The whole point of flat halo-padded storage: the same delta table
        // works at every interior site, no per-site recomputation.
        let lat = Lattice::new([5, 5], Boundary::Fixed);
        let stencil = Stencil::<2>::moore_2d();
        let table = offset_table(&stencil, lat.strides());

        for site in [[0, 0], [2, 2], [4, 4], [0, 4]] {
            let base = lat.flat_index(site);
            for (offset, delta) in stencil.offsets.iter().zip(table.iter()) {
                // Compute the expected flat index directly in padded space:
                // a boundary site's neighbour may land in the halo, which
                // is not a valid *interior* coordinate (so `flat_index`
                // can't be reused here) but is still valid flat storage —
                // that's the entire point of padding.
                let padded = [
                    (site[0] as i32 + 1 + offset[0]) as usize,
                    (site[1] as i32 + 1 + offset[1]) as usize,
                ];
                let strides = lat.strides();
                let expected = padded[0] * strides[0] + padded[1] * strides[1];
                assert_eq!(lat.neighbour(base, *delta), expected);
            }
        }
    }

    #[test]
    fn interior_classification() {
        let lat = Lattice::new([2, 2], Boundary::Fixed);
        // padded_dims = [4, 4]; interior is padded coords (1..=2, 1..=2).
        assert!(lat.is_interior(lat.flat_index([0, 0])));
        assert!(lat.is_interior(lat.flat_index([1, 1])));
        // Flat index 0 is the (0, 0) corner of the halo.
        assert!(!lat.is_interior(0));
    }

    #[test]
    fn each_interior_coord_visits_every_site_once() {
        let lat = Lattice::new([2, 3], Boundary::Fixed);
        let mut seen = Vec::new();
        lat.each_interior_coord(|c| seen.push(c));
        assert_eq!(seen.len(), 6);
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 6);
    }

    #[test]
    fn each_interior_coord_generalises_to_3d() {
        let lat = Lattice::<3>::new([2, 2, 2], Boundary::Fixed);
        let mut seen = Vec::new();
        lat.each_interior_coord(|c| seen.push(c));
        assert_eq!(seen.len(), 8);
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 8);
    }
}
