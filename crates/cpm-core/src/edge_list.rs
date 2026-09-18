//! The edge-list proposal optimisation: maintaining the set of "edge
//! sites" — interior sites with at least one `copy_neighborhood` neighbour
//! under a different owner, i.e. sites where a proposed copy could do
//! anything other than immediately early-out on `sigma[s] == sigma[s']`.
//! `monte_carlo::run_mcs` draws targets from this set under
//! `ProposalMode::EdgeList`, using binomial thinning
//! (`monte_carlo::sample_binomial`) to keep the exact same chain as
//! `ProposalMode::Uniform` with null attempts removed — see
//! `config::ProposalMode`'s doc comment for why the two are statistically,
//! but not bit-for-bit, equivalent.
//!
//! Roughly 60% of 3D attempts are interior nulls in general; this crate's
//! own confluent-density profiling (`benches/scale.rs`) measured ~78% in
//! its 3D fixture and ~85% in its 2D one — either way, a large majority of
//! attempts under `Uniform` proposal do nothing, which is exactly what this
//! set lets `run_mcs` skip.

use crate::lattice::Lattice;

/// A flat lattice index known (at the time of query) to be an edge site.
pub type EdgeSite = usize;

/// Sentinel meaning "not currently a member" in [`EdgeList::position`].
const NOT_PRESENT: u32 = u32::MAX;

/// The current set of edge sites. `sites` backs O(1) uniform sampling by
/// index; `position` is the parallel structure that makes removal O(1) via
/// swap-remove — a dense `Vec<u32>` indexed directly by flat site index
/// (sized to the padded lattice, one `u32` per site), not a `HashMap`.
///
/// **Chosen over `HashMap<EdgeSite, usize>` after benchmarking, not up
/// front**: an initial `HashMap`-backed version measured *slower*
/// end-to-end `run_mcs` throughput under `EdgeList` than plain `Uniform` on
/// confluent-density fixtures — at confluent density the null-skip savings
/// are real but modest (~15-22% of attempts are null, not the ~95%+ of the
/// sparse correctness-milestone fixtures), so `update_around`'s
/// per-accepted-move maintenance cost (hashing + probing on every
/// insert/remove) was eating the savings. `N_sites` is known and bounded up
/// front, so a dense, directly-indexed array is the correct structure here
/// regardless.
#[derive(Debug, Clone)]
pub struct EdgeList {
    sites: Vec<EdgeSite>,
    position: Vec<u32>,
}

impl EdgeList {
    fn with_capacity_for<const D: usize>(lattice: &Lattice<D>) -> Self {
        let total_padded: usize = lattice.padded_dims().iter().product();
        Self {
            sites: Vec::new(),
            position: vec![NOT_PRESENT; total_padded],
        }
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    pub fn contains(&self, site: EdgeSite) -> bool {
        self.position[site] != NOT_PRESENT
    }

    /// Sorted copy of the current membership, for set-equality comparisons
    /// (`checker.rs`) where insertion order is not itself meaningful.
    pub fn to_sorted_vec(&self) -> Vec<EdgeSite> {
        let mut v = self.sites.clone();
        v.sort_unstable();
        v
    }

    fn insert(&mut self, site: EdgeSite) {
        if self.position[site] != NOT_PRESENT {
            return;
        }
        self.position[site] = self.sites.len() as u32;
        self.sites.push(site);
    }

    fn remove(&mut self, site: EdgeSite) {
        let idx = self.position[site];
        if idx == NOT_PRESENT {
            return;
        }
        let idx = idx as usize;
        let last = self.sites.len() - 1;
        if idx != last {
            self.sites.swap(idx, last);
            let moved = self.sites[idx];
            self.position[moved] = idx as u32;
        }
        self.sites.pop();
        self.position[site] = NOT_PRESENT;
    }

    /// Whether `site` currently has any `copy_neighborhood` neighbour under
    /// a different owner than `site` itself.
    fn is_edge_site<const D: usize>(
        lattice: &Lattice<D>,
        site: EdgeSite,
        copy_offsets: &[isize],
    ) -> bool {
        let owner = lattice.get(site);
        copy_offsets
            .iter()
            .any(|&offset| lattice.get(lattice.neighbour(site, offset)) != owner)
    }

    /// From-scratch scan of every interior site. Called once, after initial
    /// cell placement (`initialization.rs`'s shared
    /// `recompute_interface_and_positions`), and by the brute-force checker
    /// for exact cross-validation against the incrementally-maintained set.
    pub fn build<const D: usize>(lattice: &Lattice<D>, copy_offsets: &[isize]) -> Self {
        let mut edge_list = Self::with_capacity_for(lattice);
        lattice.each_interior_coord(|coord| {
            let flat = lattice.flat_index(coord);
            if Self::is_edge_site(lattice, flat, copy_offsets) {
                edge_list.insert(flat);
            }
        });
        edge_list
    }

    /// After an accepted copy changes `target_flat`'s owner: only
    /// `target_flat` itself and its `copy_neighborhood` neighbours can have
    /// changed edge-membership — the same locality argument
    /// volume/interface/contact bookkeeping already relies on (a copy only
    /// changes the owner of `target_flat`; every other site's owner, and
    /// hence every other site's own edge status except through its
    /// relationship to `target_flat`, is unaffected). Recomputes exactly
    /// that local set from the post-move lattice; must be called after the
    /// lattice mutation, not before.
    ///
    /// `target_flat` is always interior by construction (an attempt only
    /// ever samples an interior target). One of its copy-neighbours can
    /// land in the halo, though — under `Periodic` that's a stand-in for a
    /// specific *other* interior site (its wrapped representative, same
    /// idiom as `state::recompute_contact`/`labelling::label_components`),
    /// which must be canonicalised before being tracked as an `EdgeSite`:
    /// passing a raw halo index into [`is_edge_site`] would have it take a
    /// *second* hop from there and walk straight out of the padded array.
    /// Under `Fixed` a halo neighbour isn't a real site at all (never a
    /// member, never will be), so it's simply skipped.
    pub fn update_around<const D: usize>(
        &mut self,
        lattice: &Lattice<D>,
        target_flat: usize,
        copy_offsets: &[isize],
    ) {
        let periodic = lattice.boundary() == crate::lattice::Boundary::Periodic;
        let canonicalize = |site: usize| -> Option<usize> {
            if lattice.is_interior(site) {
                Some(site)
            } else if periodic {
                Some(lattice.wrapped_interior_flat(site))
            } else {
                None
            }
        };
        let sites_to_check = std::iter::once(Some(target_flat)).chain(
            copy_offsets
                .iter()
                .map(|&offset| canonicalize(lattice.neighbour(target_flat, offset))),
        );
        for site in sites_to_check.flatten() {
            if Self::is_edge_site(lattice, site, copy_offsets) {
                self.insert(site);
            } else {
                self.remove(site);
            }
        }
    }

    /// Uniformly sample one edge site. `None` iff the set is empty (e.g.
    /// before any cells are placed, or a single cell filling the entire
    /// domain with no medium and no other cells left to differ from).
    pub fn sample(&self, rng: &mut impl rand::Rng) -> Option<EdgeSite> {
        if self.sites.is_empty() {
            return None;
        }
        Some(self.sites[rng.gen_range(0..self.sites.len())])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{offset_table, Boundary, Lattice};
    use crate::neighborhood::Stencil;

    fn von_neumann_offsets_2d(lattice: &Lattice<2>) -> Vec<isize> {
        offset_table(&Stencil::<2>::von_neumann(), lattice.strides())
    }

    #[test]
    fn empty_lattice_has_no_edge_sites() {
        let lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let offsets = von_neumann_offsets_2d(&lattice);
        let edge_list = EdgeList::build(&lattice, &offsets);
        assert!(edge_list.is_empty());
    }

    #[test]
    fn a_single_cell_surrounded_by_medium_has_exactly_its_boundary_as_edges() {
        let mut lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let target = lattice.flat_index([2, 2]);
        lattice.set(target, 1);
        let offsets = von_neumann_offsets_2d(&lattice);
        let edge_list = EdgeList::build(&lattice, &offsets);

        // The cell site itself, plus its 4 von Neumann neighbours (all
        // medium, each now differing from the cell) — 5 edge sites total.
        assert_eq!(edge_list.len(), 5);
        assert!(edge_list.contains(target));
        for &offset in &offsets {
            assert!(edge_list.contains(lattice.neighbour(target, offset)));
        }
    }

    #[test]
    fn a_solid_block_filling_the_whole_domain_has_no_edge_sites_under_periodic() {
        // Every site is the same cell, and periodic wrap means every
        // neighbour is too — nothing differs anywhere.
        let mut lattice = Lattice::<2>::new([4, 4], Boundary::Periodic);
        for r in 0..4 {
            for c in 0..4 {
                let flat = lattice.flat_index([r, c]);
                lattice.set(flat, 1);
            }
        }
        // Manual `set` calls don't refresh the periodic halo the way
        // `dynamics::commit` does after a real accepted move — without
        // this, the boundary sites' wrapped neighbours would still read the
        // stale all-medium halo and look like edges that aren't real ones.
        lattice.refresh_periodic_halo();
        let offsets = von_neumann_offsets_2d(&lattice);
        let edge_list = EdgeList::build(&lattice, &offsets);
        assert!(edge_list.is_empty());
    }

    #[test]
    fn update_around_matches_a_full_rebuild_after_a_manual_copy() {
        let mut lattice = Lattice::<2>::new([6, 6], Boundary::Fixed);
        // A 2x2 block of cell 1.
        for (r, c) in [(2, 2), (2, 3), (3, 2), (3, 3)] {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, 1);
        }
        let offsets = von_neumann_offsets_2d(&lattice);
        let mut edge_list = EdgeList::build(&lattice, &offsets);

        // Manually grow the cell into (2, 4) — a copy from (2,3) into
        // (2,4), i.e. target_flat = (2,4).
        let target = lattice.flat_index([2, 4]);
        lattice.set(target, 1);
        edge_list.update_around(&lattice, target, &offsets);

        let rebuilt = EdgeList::build(&lattice, &offsets);
        assert_eq!(edge_list.to_sorted_vec(), rebuilt.to_sorted_vec());
    }

    #[test]
    fn sample_only_ever_returns_a_member_of_the_set() {
        let mut lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let target = lattice.flat_index([2, 2]);
        lattice.set(target, 1);
        let offsets = von_neumann_offsets_2d(&lattice);
        let edge_list = EdgeList::build(&lattice, &offsets);

        let mut rng = crate::rng::rng_from_seed(7);
        for _ in 0..100 {
            let site = edge_list.sample(&mut rng).unwrap();
            assert!(edge_list.contains(site));
        }
    }

    #[test]
    fn sample_of_an_empty_set_is_none() {
        let lattice = Lattice::<2>::new([5, 5], Boundary::Fixed);
        let edge_list = EdgeList::with_capacity_for(&lattice);
        let mut rng = crate::rng::rng_from_seed(1);
        assert_eq!(edge_list.sample(&mut rng), None);
    }
}
