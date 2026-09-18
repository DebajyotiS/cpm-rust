//! Connected-components labelling over the whole lattice: one pass, read two
//! ways — a single mechanism that covers both fragmentation detection and
//! lumen detection rather than two separate ones. Cell sites are labelled
//! under `connectivity_neighborhood` (the cell stencil); medium sites are
//! labelled under the complementary stencil (`Stencil::<D>::full()` — Moore
//! in 2D, the 26-shell in 3D, matching the frozen (4,8)/(6,26) pairing; the
//! same stencil `connectivity.rs` already builds the simple-point pattern
//! from). From the resulting labels: a cell ID whose sites span more than
//! one label is fragmented (the per-move simple-point check only prevents
//! *local* pinch-off — a cell that becomes an annulus or torus can still
//! separate non-locally, which is accepted deliberately at move time — this
//! is the counterpart that actually catches that); a medium component not
//! touching the domain boundary is a lumen.
//!
//! **Fixed/Periodic asymmetry, deliberate, not an oversight:** a lumen's
//! "exterior" is the medium component touching the domain boundary. Under
//! `Boundary::Periodic` there is no boundary — the domain is topologically
//! a torus — so no medium component is ever marked as touching one; every
//! medium component is reported uniformly, with no distinguished exterior.
//! Under `Boundary::Fixed`, the halo is a genuine domain edge.
//!
//! Run only at sampling-interval cadence (`output.rs`), never in the Monte
//! Carlo hot loop: this is a global flood fill, deliberately kept out of
//! `monte_carlo.rs`/`connectivity.rs`.

use crate::lattice::{Boundary, CellId, Lattice};
use std::collections::{HashMap, VecDeque};

/// One medium connected component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediumComponent {
    pub volume: u32,
    /// Always `false` under `Boundary::Periodic` (see the module docs on
    /// the Fixed/Periodic asymmetry).
    pub touches_boundary: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LabellingResult {
    /// Every cell ID whose sites span more than one connected component,
    /// under `connectivity_neighborhood` — non-local fragmentation the
    /// per-move simple-point check cannot catch.
    pub fragmented_cells: Vec<CellId>,
    /// Every medium connected component, in the order first encountered.
    /// Under `Boundary::Fixed`, exclude any with `touches_boundary` to get
    /// the actual lumens; under `Periodic`, every entry is a lumen (no
    /// exterior exists to exclude).
    pub medium_components: Vec<MediumComponent>,
}

impl LabellingResult {
    /// Lumens only: under `Fixed`, medium components that don't touch the
    /// boundary; under `Periodic`, every medium component (see the module
    /// docs on the Fixed/Periodic asymmetry).
    pub fn lumens(&self) -> impl Iterator<Item = &MediumComponent> {
        self.medium_components
            .iter()
            .filter(|c| !c.touches_boundary)
    }
}

/// One pass, `O(N_sites)` with a small constant: a BFS from every
/// not-yet-visited interior site, expanding through same-owner neighbours
/// under the stencil appropriate to that owner (`connectivity_offsets` for
/// a real cell, `medium_offsets` for medium). Halo neighbours are
/// canonicalised to their true interior representative under `Periodic`
/// (`Lattice::wrapped_interior_flat`, the same idiom used throughout
/// `energy.rs`/`dynamics.rs` for the same reason) so wraparound never
/// double-labels or mislabels a site.
pub fn label_components<const D: usize>(
    lattice: &Lattice<D>,
    connectivity_offsets: &[isize],
    medium_offsets: &[isize],
) -> LabellingResult {
    let boundary = lattice.boundary();
    let n_padded: usize = lattice.padded_dims().iter().product();
    let mut visited = vec![false; n_padded];
    let mut component_count_by_cell: HashMap<CellId, u32> = HashMap::new();
    let mut medium_components = Vec::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    let canonical = |flat: usize| -> usize {
        if lattice.is_interior(flat) {
            flat
        } else if boundary == Boundary::Periodic {
            lattice.wrapped_interior_flat(flat)
        } else {
            flat // Fixed halo: permanently medium, never a component seed
        }
    };

    lattice.each_interior_coord(|coord| {
        let seed = lattice.flat_index(coord);
        if visited[seed] {
            return;
        }
        let owner = lattice.get(seed);
        let offsets = if owner == 0 {
            medium_offsets
        } else {
            connectivity_offsets
        };

        visited[seed] = true;
        queue.push_back(seed);
        let mut volume = 0u32;
        let mut touches_boundary = false;

        while let Some(site) = queue.pop_front() {
            volume += 1;
            for &offset in offsets {
                let neighbour = lattice.neighbour(site, offset);
                if !lattice.is_interior(neighbour) {
                    if owner == 0 && boundary == Boundary::Fixed {
                        touches_boundary = true;
                    }
                    if boundary != Boundary::Periodic {
                        continue; // Fixed halo: permanently medium, not a site to label
                    }
                }
                let canonical_neighbour = canonical(neighbour);
                if visited[canonical_neighbour] {
                    continue;
                }
                if lattice.get(canonical_neighbour) != owner {
                    continue;
                }
                visited[canonical_neighbour] = true;
                queue.push_back(canonical_neighbour);
            }
        }

        if owner == 0 {
            medium_components.push(MediumComponent {
                volume,
                touches_boundary,
            });
        } else {
            *component_count_by_cell.entry(owner).or_insert(0) += 1;
        }
    });

    let mut fragmented_cells: Vec<CellId> = component_count_by_cell
        .into_iter()
        .filter(|&(_, count)| count > 1)
        .map(|(id, _)| id)
        .collect();
    fragmented_cells.sort_unstable();

    LabellingResult {
        fragmented_cells,
        medium_components,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::offset_table;
    use crate::neighborhood::Stencil;

    fn stencils_2d(lattice: &Lattice<2>) -> (Vec<isize>, Vec<isize>) {
        let connectivity = offset_table(&Stencil::<2>::von_neumann(), lattice.strides());
        let medium = offset_table(&Stencil::<2>::full(), lattice.strides());
        (connectivity, medium)
    }

    fn stencils_3d(lattice: &Lattice<3>) -> (Vec<isize>, Vec<isize>) {
        let connectivity = offset_table(&Stencil::<3>::von_neumann(), lattice.strides());
        let medium = offset_table(&Stencil::<3>::full(), lattice.strides());
        (connectivity, medium)
    }

    #[test]
    fn solid_blob_has_no_lumens_and_no_fragmentation() {
        let mut lattice = Lattice::<2>::new([7, 7], Boundary::Fixed);
        for x in 2..5 {
            for y in 2..5 {
                let flat = lattice.flat_index([x, y]);
                lattice.set(flat, 1);
            }
        }
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert!(result.fragmented_cells.is_empty());
        assert_eq!(result.lumens().count(), 0);
    }

    #[test]
    fn hollow_ring_has_exactly_one_lumen() {
        // A 5x5 ring of cell 1 (the border of a 5x5 block), medium hole in
        // the centre, well away from the lattice's own Fixed boundary.
        let mut lattice = Lattice::<2>::new([9, 9], Boundary::Fixed);
        for x in 2..7 {
            for y in 2..7 {
                let on_ring = x == 2 || x == 6 || y == 2 || y == 6;
                if on_ring {
                    let flat = lattice.flat_index([x, y]);
                    lattice.set(flat, 1);
                }
            }
        }
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert!(
            result.fragmented_cells.is_empty(),
            "a ring is a single connected component under von Neumann connectivity"
        );
        let lumens: Vec<_> = result.lumens().collect();
        assert_eq!(lumens.len(), 1, "{:?}", result.medium_components);
        assert_eq!(lumens[0].volume, 9); // the 3x3 hole
    }

    #[test]
    fn one_site_breach_in_the_ring_destroys_the_lumen() {
        let mut lattice = Lattice::<2>::new([9, 9], Boundary::Fixed);
        for x in 2..7 {
            for y in 2..7 {
                let on_ring = x == 2 || x == 6 || y == 2 || y == 6;
                if on_ring {
                    let flat = lattice.flat_index([x, y]);
                    lattice.set(flat, 1);
                }
            }
        }
        // Breach: clear one ring site back to medium, opening a path from
        // the hole to the exterior.
        let breach = lattice.flat_index([4, 2]);
        lattice.set(breach, 0);
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert_eq!(
            result.lumens().count(),
            0,
            "a breached ring must not count as enclosing a lumen"
        );
    }

    #[test]
    fn two_adjacent_shells_give_two_lumens() {
        let mut lattice = Lattice::<2>::new([15, 9], Boundary::Fixed);
        for (ox, oy) in [(1, 2), (8, 2)] {
            for x in ox..ox + 5 {
                for y in oy..oy + 5 {
                    let on_ring = x == ox || x == ox + 4 || y == oy || y == oy + 4;
                    if on_ring {
                        let flat = lattice.flat_index([x, y]);
                        lattice.set(flat, if ox == 1 { 1 } else { 2 });
                    }
                }
            }
        }
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert!(result.fragmented_cells.is_empty());
        assert_eq!(result.lumens().count(), 2);
    }

    #[test]
    fn periodic_boundary_reports_every_medium_component_with_no_exterior() {
        let mut lattice = Lattice::<2>::new([9, 9], Boundary::Periodic);
        for x in 2..7 {
            for y in 2..7 {
                let on_ring = x == 2 || x == 6 || y == 2 || y == 6;
                if on_ring {
                    let flat = lattice.flat_index([x, y]);
                    lattice.set(flat, 1);
                }
            }
        }
        lattice.refresh_periodic_halo();
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        // Two medium components: the enclosed hole, and the background
        // that wraps around the torus. Neither is ever marked as touching
        // a boundary under `Periodic` — both come back as "lumens".
        assert!(result.medium_components.iter().all(|c| !c.touches_boundary));
        assert_eq!(result.lumens().count(), result.medium_components.len());
        assert_eq!(result.medium_components.len(), 2);
    }

    #[test]
    fn two_disconnected_pieces_of_the_same_id_are_fragmented() {
        let mut lattice = Lattice::<2>::new([9, 9], Boundary::Fixed);
        // Two separate single-site blobs sharing cell id 1 — not something
        // ordinary dynamics could ever produce given the per-move
        // simple-point check, but exactly the state this function exists
        // to detect if it ever happens (e.g. a non-local pinch).
        let a = lattice.flat_index([1, 1]);
        let b = lattice.flat_index([6, 6]);
        lattice.set(a, 1);
        lattice.set(b, 1);
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert_eq!(result.fragmented_cells, vec![1]);
    }

    #[test]
    fn a_connected_ring_shaped_cell_is_not_fragmented() {
        // The cell itself is a ring (an annulus) — topologically unusual
        // but genuinely one connected component under von Neumann
        // connectivity, so it must NOT be flagged.
        let mut lattice = Lattice::<2>::new([9, 9], Boundary::Fixed);
        for x in 2..7 {
            for y in 2..7 {
                let on_ring = x == 2 || x == 6 || y == 2 || y == 6;
                if on_ring {
                    let flat = lattice.flat_index([x, y]);
                    lattice.set(flat, 1);
                }
            }
        }
        let (connectivity, medium) = stencils_2d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert!(result.fragmented_cells.is_empty());
    }

    #[test]
    fn hollow_shell_3d_has_one_lumen() {
        let mut lattice = Lattice::<3>::new([9, 9, 9], Boundary::Fixed);
        for x in 2..7 {
            for y in 2..7 {
                for z in 2..7 {
                    let on_shell = x == 2 || x == 6 || y == 2 || y == 6 || z == 2 || z == 6;
                    if on_shell {
                        let flat = lattice.flat_index([x, y, z]);
                        lattice.set(flat, 1);
                    }
                }
            }
        }
        let (connectivity, medium) = stencils_3d(&lattice);
        let result = label_components(&lattice, &connectivity, &medium);
        assert!(result.fragmented_cells.is_empty());
        assert_eq!(result.lumens().count(), 1);
        assert_eq!(result.lumens().next().unwrap().volume, 27); // 3x3x3 hole
    }
}
