//! Per-cell bookkeeping state. The brute-force checker recomputes exactly
//! these quantities from scratch to validate incremental updates. Data
//! layout only — incremental updates on accepted copies belong to
//! `dynamics.rs`.

use crate::cell::Cell;
use crate::lattice::{CellId, Lattice};

/// Canonical `(min, max)` key for an unordered pair of cell IDs — including
/// medium (`0`) — so [`ContactGraph`] has exactly one entry per pair
/// regardless of which side of a move it was observed from.
pub fn contact_key(a: CellId, b: CellId) -> (CellId, CellId) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Pairwise shared-interface counts — for every unordered pair of distinct
/// cell IDs (medium, id `0`, included) that are energy-neighbours somewhere
/// on the lattice, the count of neighbouring site pairs straddling that
/// boundary. A dense, directly-indexed `n x n` matrix (`n = n_cells + 1`;
/// cell IDs are fixed for the run), not a `HashMap`: a
/// `HashMap<(CellId, CellId), u32>` version measured as the dominant cost of
/// an accepted move's bookkeeping (`dynamics::update_contact`, ~240 ns/call
/// in isolation vs. `Terms::total_delta`'s 62-66 ns). The same reasoning
/// applies to `edge_list::EdgeList`'s index structure: a bounded, known-size
/// key domain calls for a dense array, not a hash table. `n_cells` stays in
/// the low thousands for any sizing regime this project targets, so the
/// `O(n^2)` memory (one `u32` per possible pair, most of which are never
/// actually adjacent) is a deliberate trade for O(1) mutation with no
/// hashing or probing at all.
#[derive(Clone, PartialEq)]
pub struct ContactGraph {
    n: usize,
    counts: Vec<u32>,
}

impl ContactGraph {
    /// `n_cells + 1` possible IDs (medium is `0`); all pairs start at `0`.
    pub fn new(n_cells: usize) -> Self {
        let n = n_cells + 1;
        Self {
            n,
            counts: vec![0; n * n],
        }
    }

    fn index(&self, a: CellId, b: CellId) -> usize {
        let (a, b) = contact_key(a, b);
        a as usize * self.n + b as usize
    }

    /// `0` for any pair that has never been in contact — absent pairs are
    /// never distinguished from zero-count ones (there is no "unset" state
    /// here, unlike a `HashMap`'s missing-key case).
    pub fn get(&self, a: CellId, b: CellId) -> u32 {
        self.counts[self.index(a, b)]
    }

    pub fn increment(&mut self, a: CellId, b: CellId) {
        let i = self.index(a, b);
        self.counts[i] += 1;
    }

    pub fn decrement(&mut self, a: CellId, b: CellId) {
        let i = self.index(a, b);
        debug_assert!(
            self.counts[i] > 0,
            "decrementing contact pair ({a}, {b}) that was never recorded"
        );
        self.counts[i] = self.counts[i].saturating_sub(1);
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|&c| c == 0)
    }

    /// Every currently-nonzero pair, in canonical `(min, max)` order —
    /// row-major storage already visits them in ascending order, so no
    /// separate sort is needed. Not on any hot path (readout/checker only).
    pub fn iter_nonzero(&self) -> impl Iterator<Item = ((CellId, CellId), u32)> + '_ {
        let n = self.n;
        self.counts
            .iter()
            .enumerate()
            .filter_map(move |(idx, &count)| {
                if count == 0 {
                    return None;
                }
                Some((((idx / n) as CellId, (idx % n) as CellId), count))
            })
    }

    /// The full dense `n x n` buffer (`n = n_cells + 1`, medium at index `0`)
    /// in row-major order, for callers outside this crate that need the whole
    /// matrix at once (e.g. converting to a NumPy array) rather than walking
    /// nonzero pairs one at a time.
    pub fn to_dense(&self) -> (usize, Vec<u32>) {
        (self.n, self.counts.clone())
    }
}

impl std::fmt::Debug for ContactGraph {
    /// Prints only nonzero pairs, like a `HashMap`'s Debug would — printing
    /// the full dense array (mostly zeros) would be unreadable and wouldn't
    /// match what a mismatch error actually wants to show.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map().entries(self.iter_nonzero()).finish()
    }
}

/// Recomputes the contact graph from scratch by scanning the whole
/// lattice — the same pairwise definition `dynamics::commit`'s incremental
/// updates maintain. Used both by initialisation (a fresh placement never
/// went through `dynamics::commit`, so its contact graph has to be
/// established some other way) and by the brute-force checker (to verify
/// the incrementally-maintained graph hasn't drifted).
///
/// Periodic-boundary dedupe mirrors `AdhesionTerm::energy`'s exact pattern:
/// only interior sites are ever the outer loop variable `i`, so a pair with
/// a halo `j` is deduped by resolving `j` to the interior site it mirrors
/// (`Lattice::wrapped_interior_flat`) under `Periodic`, and always counted
/// under `Fixed` (a `Fixed` halo is permanently medium, never a stand-in
/// for another real site).
pub fn recompute_contact<const D: usize>(
    lattice: &Lattice<D>,
    energy_offsets: &[isize],
    n_cells: usize,
) -> ContactGraph {
    let periodic = lattice.boundary() == crate::lattice::Boundary::Periodic;
    let mut contact = ContactGraph::new(n_cells);
    lattice.each_interior_coord(|coord| {
        let i_flat = lattice.flat_index(coord);
        let i_cell = lattice.get(i_flat);
        for &offset in energy_offsets {
            let j_flat = lattice.neighbour(i_flat, offset);
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
            contact.increment(i_cell, j_cell);
        }
    });
    contact
}

/// Cell IDs are assigned at initialisation, 1-indexed and contiguous
/// (`1..=n_cells`), and stable for the run — cells are never created or
/// destroyed during a simulation. Per-cell arrays are therefore dense,
/// indexed by `id - 1`.
pub fn index_of(id: CellId) -> usize {
    debug_assert!(id >= 1, "cell id 0 is medium, not indexable state");
    (id - 1) as usize
}

#[derive(Debug, Clone)]
pub struct State<const D: usize> {
    pub lattice: Lattice<D>,
    pub cells: Vec<Cell>,
    /// `V_c`, one entry per cell, dense-indexed via [`index_of`].
    pub volume: Vec<u32>,
    /// `I_c` under the configured energy neighbourhood, one entry per cell.
    pub interface: Vec<u32>,
    /// Unwrapped position sums, `[i64; D]` per cell so centroids and MSD
    /// survive periodic wrapping.
    pub sum_position: Vec<[i64; D]>,
    /// Raw (uncentred) second-moment position sums per cell, `sum_i r_i ⊗
    /// r_i` in the same unwrapped coordinates as [`sum_position`]. The
    /// gyration tensor is `sum_position_outer / V - x_c ⊗ x_c` at readout
    /// time, the standard sum-of-squares-minus-square-of-mean
    /// decomposition, so no separate centred accumulator is needed.
    /// Maintained incrementally in `dynamics::update_position_sums`
    /// alongside `sum_position`, reusing the same nearest-periodic-image
    /// coordinate.
    pub sum_position_outer: Vec<[[i64; D]; D]>,
    /// Pairwise shared-interface counts: for every unordered pair of
    /// distinct cell IDs (medium, id `0`, included) that are
    /// energy-neighbours somewhere on the lattice, the count of
    /// neighbouring site pairs straddling that boundary — the same
    /// definition `InterfaceTerm`'s per-cell `interface` uses, just not
    /// collapsed to one side. Keyed via [`contact_key`] so each pair has one
    /// canonical entry. See [`ContactGraph`] for the storage layout.
    pub contact: ContactGraph,
    /// The running total of the conservative Hamiltonian, maintained
    /// incrementally (every accepted move adds its already-computed delta,
    /// `monte_carlo.rs`) — kept *separately* from
    /// `Terms::global_energy(state)` so the brute-force checker has an
    /// independently maintained value to compare a from-scratch
    /// recomputation against.
    pub conservative_energy: f64,
}

impl<const D: usize> State<D> {
    /// An empty state: the lattice exists (all medium) but no cells have
    /// been placed yet. `CPM::new` produces exactly this; cell placement
    /// happens in `initialization.rs`.
    pub fn empty(lattice: Lattice<D>) -> Self {
        Self {
            lattice,
            cells: Vec::new(),
            volume: Vec::new(),
            interface: Vec::new(),
            sum_position: Vec::new(),
            sum_position_outer: Vec::new(),
            contact: ContactGraph::new(0),
            conservative_energy: 0.0,
        }
    }

    pub fn with_cells(lattice: Lattice<D>, cells: Vec<Cell>) -> Self {
        let n = cells.len();
        Self {
            lattice,
            cells,
            volume: vec![0; n],
            interface: vec![0; n],
            sum_position: vec![[0i64; D]; n],
            sum_position_outer: vec![[[0i64; D]; D]; n],
            contact: ContactGraph::new(n),
            conservative_energy: 0.0,
        }
    }

    pub fn n_cells(&self) -> usize {
        self.cells.len()
    }

    /// The type index of `id`, or `None` for medium (id 0).
    pub fn type_of(&self, id: CellId) -> Option<usize> {
        if id == 0 {
            None
        } else {
            Some(self.cells[index_of(id)].type_index)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Boundary;

    #[test]
    fn empty_state_has_no_cells() {
        let lattice = Lattice::<2>::new([4, 4], Boundary::Fixed);
        let state = State::empty(lattice);
        assert_eq!(state.n_cells(), 0);
        assert!(state.volume.is_empty());
    }

    #[test]
    fn with_cells_allocates_zeroed_bookkeeping_per_cell() {
        let lattice = Lattice::<2>::new([4, 4], Boundary::Fixed);
        let cells = vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 0,
            },
        ];
        let state = State::with_cells(lattice, cells);
        assert_eq!(state.n_cells(), 2);
        assert_eq!(state.volume, vec![0, 0]);
        assert_eq!(state.interface, vec![0, 0]);
        assert_eq!(state.sum_position, vec![[0, 0], [0, 0]]);
        assert_eq!(
            state.sum_position_outer,
            vec![[[0, 0], [0, 0]], [[0, 0], [0, 0]]]
        );
        assert!(state.contact.is_empty());
    }

    #[test]
    fn index_of_is_one_indexed() {
        assert_eq!(index_of(1), 0);
        assert_eq!(index_of(2), 1);
    }

    #[test]
    fn contact_key_is_order_independent() {
        assert_eq!(contact_key(1, 2), contact_key(2, 1));
        assert_eq!(contact_key(0, 3), (0, 3));
        assert_eq!(contact_key(3, 0), (0, 3));
    }

    #[test]
    fn to_dense_round_trips_against_iter_nonzero() {
        let mut contact = ContactGraph::new(3);
        contact.increment(0, 1);
        contact.increment(0, 1);
        contact.increment(2, 3);

        let (n, counts) = contact.to_dense();
        assert_eq!(n, 4);
        assert_eq!(counts.len(), n * n);

        let mut from_dense: Vec<((CellId, CellId), u32)> = Vec::new();
        for a in 0..n {
            for b in 0..n {
                let count = counts[a * n + b];
                if count != 0 {
                    from_dense.push(((a as CellId, b as CellId), count));
                }
            }
        }
        let expected: Vec<((CellId, CellId), u32)> = contact.iter_nonzero().collect();
        assert_eq!(from_dense, expected);
    }
}
