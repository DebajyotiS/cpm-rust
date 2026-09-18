//! Simple-point connectivity check: a proposed copy is rejected unless the
//! target site is a "simple point" for the losing cell — removing it changes
//! neither the cell's connectivity nor its topology (it forbids creating or
//! destroying a cavity or tunnel, not just disconnection). Medium is exempt
//! from this check entirely — that exemption lives in `connectivity.rs`,
//! which is the only caller of this module in production code. This module
//! just answers "is this pattern simple," generically over `D`.
//!
//! ## A note on the (6, 26) formula
//!
//! The criterion is easy to state backwards. The general n-simple theorem is
//! "x is n-simple for X iff `T_n(x,X)=1` and `T_n̄(x,X̄)=1`", where `n` is the
//! *object's own* connectivity — for cells that are 6-connected, that reads:
//!
//! ```text
//! T_6(x, losing cell) == 1   and   T_26(x, complement) == 1
//! ```
//!
//! i.e. the losing cell's own members, within the full 3x3(x3) neighbourhood,
//! must form exactly one connected component under *cell* adjacency (4 in
//! 2D, 6 in 3D) that actually touches `x`; everything that is *not* the
//! losing cell must form exactly one connected component under the
//! complementary adjacency (8 in 2D, 26 in 3D). Some presentations state it
//! with the subscripts swapped, which fails for this project's (6, 26)
//! connectivity pairing.
//!
//! Sources: Bertrand & Malandain, "Simple points, topological numbers and
//! geodesic neighborhoods in cubic grids" (Pattern Recognition Letters 15,
//! 1994); the general n-simple theorem restated for 2D at
//! arxiv.org/html/2410.21588 ("x is n-simple for X iff T_n(x,X)=1 and
//! T_n̄(x,X̄)=1"); standard descriptions of the (6,26)/(26,6) thinning
//! criterion (e.g. Lee, Kashyap & Chu 1994) stating the object side uses its
//! own connectivity and the background the complementary one.

use crate::neighborhood::Stencil;
use std::sync::OnceLock;

/// Face adjacency (Manhattan distance exactly 1) — the *cell* connectivity
/// (4 in 2D, 6 in 3D).
fn is_face_adjacent<const D: usize>(a: [i32; D], b: [i32; D]) -> bool {
    let manhattan: i32 = (0..D).map(|d| (a[d] - b[d]).abs()).sum();
    manhattan == 1
}

/// Full adjacency (Chebyshev distance exactly 1) — the complementary
/// connectivity (8 in 2D, 26 in 3D).
fn is_full_adjacent<const D: usize>(a: [i32; D], b: [i32; D]) -> bool {
    a != b && (0..D).all(|d| (a[d] - b[d]).abs() <= 1)
}

/// Connected components of `mask` under `adjacent`, as index sets into
/// `offsets`. O(n^2); this is the slow reference path, never the hot loop.
fn flood_components<const D: usize>(
    offsets: &[[i32; D]],
    mask: &[bool],
    adjacent: impl Fn([i32; D], [i32; D]) -> bool,
) -> Vec<Vec<usize>> {
    let n = offsets.len();
    let mut visited = vec![false; n];
    let mut components = Vec::new();
    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        visited[start] = true;
        let mut stack = vec![start];
        let mut component = vec![start];
        while let Some(i) = stack.pop() {
            for j in 0..n {
                if mask[j] && !visited[j] && adjacent(offsets[i], offsets[j]) {
                    visited[j] = true;
                    stack.push(j);
                    component.push(j);
                }
            }
        }
        components.push(component);
    }
    components
}

/// The slow, obviously-correct reference implementation. `offsets` is the
/// full `3^D - 1` neighbourhood in a fixed canonical order (see
/// [`canonical_offsets_2d`]/[`canonical_offsets_3d`]); `object[i]` says
/// whether `offsets[i]` belongs to the losing cell.
///
/// The all-neighbours-belong-to-the-losing-cell pattern (empty complement)
/// is provably unreachable from real Monte Carlo usage: the proposing cell
/// is always a connectivity-adjacent, different-cell neighbour of the target
/// site (`copy_neighborhood` is a subset of `connectivity_neighborhood`), so
/// that neighbour is always present in the pattern and always non-object.
/// This function does not special-case it regardless: an empty complement
/// naively yields 0 components, `0 != 1`, and the point is (conservatively,
/// harmlessly) reported as not simple. See [`ring_walk_is_simple`]'s doc
/// comment and the module tests for how this (and one other structural
/// case) relate to the independent 2D cross-check.
pub fn is_simple_point<const D: usize>(offsets: &[[i32; D]], object: &[bool]) -> bool {
    debug_assert_eq!(offsets.len(), object.len());
    let origin = [0i32; D];

    let complement: Vec<bool> = object.iter().map(|&b| !b).collect();
    let complement_components = flood_components(offsets, &complement, is_full_adjacent::<D>);
    if complement_components.len() != 1 {
        return false;
    }

    let object_components = flood_components(offsets, object, is_face_adjacent::<D>);
    let touching_origin = object_components
        .iter()
        .filter(|component| {
            component
                .iter()
                .any(|&i| is_face_adjacent(offsets[i], origin))
        })
        .count();
    touching_origin == 1
}

/// Independent 2D-only cross-check: sorts the 8 canonical Moore offsets into
/// their true geometric cyclic order and counts maximal runs of "object"
/// around the ring — the classic hand-verifiable 2D simple point test.
///
/// This is **not** a full substitute for [`is_simple_point`]: it counts
/// contiguous runs but, unlike the reference, never checks whether a run
/// actually reaches `x` via a face step rather than only a diagonal one. A
/// pattern whose only object members are diagonal (e.g. a single corner
/// neighbour with nothing else) forms "1 run" by this method but is
/// correctly rejected by the reference, since it never touches the origin
/// under face adjacency. The module tests prove this diagonal-only-component
/// case is the *exact and only* source of disagreement between the two
/// methods, which is what makes this a meaningful cross-check rather than a
/// coincidentally-similar independent calculation.
pub fn ring_walk_is_simple(offsets: &[[i32; 2]; 8], object: &[bool; 8]) -> bool {
    let mut order: Vec<usize> = (0..8).collect();
    order.sort_by(|&a, &b| {
        let angle = |o: [i32; 2]| (o[1] as f64).atan2(o[0] as f64);
        angle(offsets[a]).partial_cmp(&angle(offsets[b])).unwrap()
    });
    let cyclic: Vec<bool> = order.iter().map(|&i| object[i]).collect();
    count_cyclic_runs(&cyclic) == 1
}

fn count_cyclic_runs(values: &[bool]) -> usize {
    if values.iter().all(|&v| v) {
        return 1;
    }
    if values.iter().all(|&v| !v) {
        return 0;
    }
    let n = values.len();
    (0..n)
        .filter(|&i| values[i] && !values[(i + n - 1) % n])
        .count()
}

/// The canonical 2D pattern basis: the 8 Moore offsets, in the same order
/// `Stencil::moore_2d` produces. Bit `i` of a pattern means "offset `i`
/// belongs to the losing cell."
pub fn canonical_offsets_2d() -> [[i32; 2]; 8] {
    let offsets = Stencil::<2>::moore_2d().offsets;
    offsets.try_into().expect("moore_2d always has 8 offsets")
}

/// The canonical 3D pattern basis: the 26 full-shell offsets, in the same
/// order `Stencil::twenty_six_3d` produces.
pub fn canonical_offsets_3d() -> [[i32; 3]; 26] {
    let offsets = Stencil::<3>::twenty_six_3d().offsets;
    offsets
        .try_into()
        .expect("twenty_six_3d always has 26 offsets")
}

/// Precompute, for each of the `n` canonical offsets, a bitmask of which
/// other offset indices are `adjacent` to it. Used to make table generation
/// a bitwise flood fill instead of the slow reference's nested loops.
fn adjacency_bitmasks<const D: usize>(
    offsets: &[[i32; D]],
    adjacent: impl Fn([i32; D], [i32; D]) -> bool,
) -> Vec<u32> {
    let n = offsets.len();
    let mut masks = vec![0u32; n];
    for i in 0..n {
        for j in 0..n {
            if i != j && adjacent(offsets[i], offsets[j]) {
                masks[i] |= 1 << j;
            }
        }
    }
    masks
}

fn flood_fill_mask(mut component: u32, universe: u32, n: usize, adj: &[u32]) -> u32 {
    loop {
        let frontier = component;
        for (i, &neighbours) in adj.iter().enumerate().take(n) {
            if (component >> i) & 1 == 1 {
                component |= neighbours & universe;
            }
        }
        if component == frontier {
            return component;
        }
    }
}

fn component_count_mask(universe: u32, n: usize, adj: &[u32]) -> usize {
    let mut remaining = universe;
    let mut count = 0;
    while remaining != 0 {
        let start = remaining.trailing_zeros() as usize;
        let component = flood_fill_mask(1 << start, universe, n, adj);
        remaining &= !component;
        count += 1;
    }
    count
}

fn touching_component_count_mask(universe: u32, touch_mask: u32, n: usize, adj: &[u32]) -> usize {
    let mut remaining = universe;
    let mut count = 0;
    while remaining != 0 {
        let start = remaining.trailing_zeros() as usize;
        let component = flood_fill_mask(1 << start, universe, n, adj);
        remaining &= !component;
        if component & touch_mask != 0 {
            count += 1;
        }
    }
    count
}

/// The fast bitmask path used by table generation — it operates only on the
/// 26-bit neighbourhood pattern already computed for the table lookup, so
/// it's a bounded, small computation, not a general flood fill. Logically
/// identical to [`is_simple_point`] (cross-checked exhaustively in 2D and on
/// random samples in 3D — see the tests) but avoids per-pattern allocation:
/// a handful of `u32` bitwise ORs per flood-fill iteration, no heap traffic.
/// This is also the direct-computation alternative to the lookup table,
/// exposed publicly (`direct_checker_3d`) so it can be benchmarked
/// head-to-head against the table on realistic patterns rather than
/// assuming the table wins.
pub struct FastChecker {
    n: usize,
    face_adj: Vec<u32>,
    full_adj: Vec<u32>,
    origin_face_mask: u32,
    full_mask: u32,
}

impl FastChecker {
    pub fn new<const D: usize>(offsets: &[[i32; D]]) -> Self {
        let n = offsets.len();
        let origin = [0i32; D];
        let origin_face_mask = offsets
            .iter()
            .enumerate()
            .filter(|(_, &o)| is_face_adjacent(o, origin))
            .fold(0u32, |mask, (i, _)| mask | (1 << i));
        Self {
            n,
            face_adj: adjacency_bitmasks(offsets, is_face_adjacent::<D>),
            full_adj: adjacency_bitmasks(offsets, is_full_adjacent::<D>),
            origin_face_mask,
            full_mask: (1u64 << n) as u32 - 1,
        }
    }

    pub fn is_simple(&self, object_mask: u32) -> bool {
        let complement_mask = (!object_mask) & self.full_mask;
        if component_count_mask(complement_mask, self.n, &self.full_adj) != 1 {
            return false;
        }
        touching_component_count_mask(object_mask, self.origin_face_mask, self.n, &self.face_adj)
            == 1
    }
}

static DIRECT_CHECKER_3D: OnceLock<FastChecker> = OnceLock::new();

/// Process-wide cached direct 3D checker — built once (cheap: O(26^2)
/// adjacency bitmask setup, not the O(2^26) table generation), on first use.
/// The direct-computation counterpart to [`simple_point_table_3d`].
pub fn direct_checker_3d() -> &'static FastChecker {
    DIRECT_CHECKER_3D.get_or_init(|| FastChecker::new(&canonical_offsets_3d()))
}

/// 2D table: 256 entries, trivial to store and generate directly.
pub struct SimplePointTable2D {
    bits: [bool; 256],
}

impl SimplePointTable2D {
    pub fn generate() -> Self {
        let offsets = canonical_offsets_2d();
        let checker = FastChecker::new(&offsets);
        let mut bits = [false; 256];
        for (pattern, bit) in bits.iter_mut().enumerate() {
            *bit = checker.is_simple(pattern as u32);
        }
        Self { bits }
    }

    pub fn is_simple(&self, pattern: u8) -> bool {
        self.bits[pattern as usize]
    }
}

/// A packed bitset, `u64`-backed. Used for the 3D table (2^26 bits ~ 8 MiB) —
/// far too large to store as `Vec<bool>` (which would be one full byte per
/// entry, 64 MiB, and still not the point: this is meant to model the
/// production storage format).
struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    fn new(len: usize) -> Self {
        Self {
            words: vec![0u64; len.div_ceil(64)],
        }
    }

    fn set(&mut self, index: usize) {
        self.words[index / 64] |= 1 << (index % 64);
    }

    fn get(&self, index: usize) -> bool {
        (self.words[index / 64] >> (index % 64)) & 1 == 1
    }
}

/// 3D table: 2^26 entries. Generation is O(2^26) with a small constant
/// factor (bitmask flood fill) — on the order of seconds, computed once at
/// startup or shipped as a prebuilt artifact. Not something to run inside
/// every `cargo test` invocation: use [`simple_point_table_3d`] to get a
/// process-wide cached instance instead of calling `generate` directly.
pub struct SimplePointTable3D {
    bits: BitSet,
}

const PATTERN_COUNT_3D: usize = 1 << 26;

impl SimplePointTable3D {
    pub fn generate() -> Self {
        let offsets = canonical_offsets_3d();
        let checker = FastChecker::new(&offsets);
        let mut bits = BitSet::new(PATTERN_COUNT_3D);
        for pattern in 0..PATTERN_COUNT_3D as u32 {
            if checker.is_simple(pattern) {
                bits.set(pattern as usize);
            }
        }
        Self { bits }
    }

    pub fn is_simple(&self, pattern: u32) -> bool {
        self.bits.get(pattern as usize)
    }
}

static SIMPLE_POINT_TABLE_3D: OnceLock<SimplePointTable3D> = OnceLock::new();

/// Process-wide cached 3D table — generated once, on first use.
pub fn simple_point_table_3d() -> &'static SimplePointTable3D {
    SIMPLE_POINT_TABLE_3D.get_or_init(SimplePointTable3D::generate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn face_and_full_adjacency_sanity() {
        assert!(is_face_adjacent([0, 0], [1, 0]));
        assert!(!is_face_adjacent([0, 0], [1, 1]));
        assert!(is_full_adjacent([0, 0], [1, 1]));
        assert!(!is_full_adjacent([0, 0], [2, 0]));
        assert!(!is_full_adjacent([0, 0], [0, 0]));
    }

    /// A 2-site cell `{x, N}`: removing `x` leaves the valid 1-site cell
    /// `{N}` — simple.
    #[test]
    fn reference_accepts_a_two_site_cell() {
        let offsets = canonical_offsets_2d();
        let mut object = [false; 8];
        let north = offsets.iter().position(|&o| o == [0, -1]).unwrap();
        object[north] = true;
        assert!(is_simple_point(&offsets, &object));
    }

    /// `{N, E}` only, with the diagonal `NE` between them absent: `N` and
    /// `E` are not face-adjacent to *each other*, only each individually to
    /// `x`, so removing `x` would split them. Not simple — this is the
    /// classic mistake a "just check the 4 face neighbours pairwise" reading
    /// would get wrong by assuming two face neighbours are always mutually
    /// connected.
    #[test]
    fn reference_rejects_face_neighbours_not_mutually_adjacent() {
        let offsets = canonical_offsets_2d();
        let mut object = [false; 8];
        let north = offsets.iter().position(|&o| o == [0, -1]).unwrap();
        let east = offsets.iter().position(|&o| o == [1, 0]).unwrap();
        object[north] = true;
        object[east] = true;
        assert!(!is_simple_point(&offsets, &object));
    }

    /// Two diagonally-opposite, mutually disconnected object cells: not
    /// simple (the object side has 2 components).
    #[test]
    fn reference_rejects_split_object() {
        let offsets = canonical_offsets_2d();
        let mut object = [false; 8];
        let north = offsets.iter().position(|&o| o == [0, -1]).unwrap();
        let south = offsets.iter().position(|&o| o == [0, 1]).unwrap();
        object[north] = true;
        object[south] = true;
        assert!(!is_simple_point(&offsets, &object));
    }

    /// Empty object (the site would be the cell's only member): not simple.
    #[test]
    fn reference_rejects_isolated_singleton() {
        let offsets = canonical_offsets_2d();
        let object = [false; 8];
        assert!(!is_simple_point(&offsets, &object));
    }

    /// Fully-occupied object (empty complement): rejected by convention, and
    /// documented as unreachable in practice.
    #[test]
    fn reference_rejects_fully_interior_point() {
        let offsets = canonical_offsets_2d();
        let object = [true; 8];
        assert!(!is_simple_point(&offsets, &object));
    }

    /// Exhaustive 2D cross-check: whenever the reference and the
    /// independently-coded ring-walk method disagree, it must be because the
    /// object has at least one face-adjacency component that never touches
    /// the origin (a "diagonal-only" component — see
    /// [`ring_walk_is_simple`]'s doc comment) — never for any other reason.
    /// This is the precise, checkable claim behind treating ring-walk as a
    /// meaningful independent cross-check rather than a coincidentally
    /// similar calculation: every disagreement traces back to the one
    /// documented structural gap, on all 256 patterns.
    #[test]
    fn ring_walk_disagreements_are_always_a_diagonal_only_component() {
        let offsets = canonical_offsets_2d();
        let origin = [0i32; 2];
        // The all-object pattern is a separately-documented disagreement
        // (empty complement, see `reference_rejects_fully_interior_point`
        // and `is_simple_point`'s doc comment) — excluded here since it
        // isn't a diagonal-only-component case.
        for pattern in (0u32..256).filter(|&p| p != 0xFF) {
            let mut object = [false; 8];
            for (i, bit) in object.iter_mut().enumerate() {
                *bit = (pattern >> i) & 1 == 1;
            }
            let reference = is_simple_point(&offsets, &object);
            let ring_walk = ring_walk_is_simple(&offsets, &object);
            if reference == ring_walk {
                continue;
            }
            let has_diagonal_only_component =
                flood_components(&offsets, &object, is_face_adjacent::<2>)
                    .iter()
                    .any(|component| {
                        component
                            .iter()
                            .all(|&i| !is_face_adjacent(offsets[i], origin))
                    });
            assert!(
                has_diagonal_only_component,
                "pattern {pattern:#010b} disagreed (reference={reference}, ring_walk={ring_walk}) \
                 for an undocumented reason: no diagonal-only component present"
            );
        }
    }

    /// Hand-verified agreement on realistic patterns — every one of these
    /// has all its object members reachable from an axis position, so the
    /// diagonal-only-component gap above never applies and the two methods
    /// must agree.
    #[test]
    fn ring_walk_agrees_with_reference_on_realistic_patterns() {
        let offsets = canonical_offsets_2d();
        let idx = |o: [i32; 2]| offsets.iter().position(|&x| x == o).unwrap();
        let cases: [(&str, &[[i32; 2]]); 5] = [
            ("empty", &[]),
            ("single axis", &[[0, -1]]),
            ("contiguous 3-arc", &[[0, -1], [1, -1], [1, 0]]),
            ("two separate axis cells", &[[0, -1], [0, 1]]),
            (
                "all 4 axis cells (no diagonal support)",
                &[[0, -1], [1, 0], [0, 1], [-1, 0]],
            ),
        ];
        for (name, members) in cases {
            let mut object = [false; 8];
            for &m in members {
                object[idx(m)] = true;
            }
            assert_eq!(
                is_simple_point(&offsets, &object),
                ring_walk_is_simple(&offsets, &object),
                "case {name:?} should have both methods agree"
            );
        }
    }

    /// Table generation matches the slow reference on all 256 2D patterns.
    #[test]
    fn table_2d_matches_reference_exhaustively() {
        let offsets = canonical_offsets_2d();
        let table = SimplePointTable2D::generate();
        for pattern in 0u32..256 {
            let mut object = [false; 8];
            for (i, bit) in object.iter_mut().enumerate() {
                *bit = (pattern >> i) & 1 == 1;
            }
            assert_eq!(
                table.is_simple(pattern as u8),
                is_simple_point(&offsets, &object),
                "mismatch at pattern {pattern:#010b}"
            );
        }
    }

    /// Table generation (the fast bitmask path) matches the slow reference
    /// on a large random sample of the 2^26 3D patterns. Building the full
    /// 3D table is slow and `#[ignore]`d elsewhere; this test only exercises
    /// the fast-vs-slow agreement on individual patterns, which is cheap.
    #[test]
    fn fast_checker_matches_slow_reference_on_random_3d_patterns() {
        let offsets = canonical_offsets_3d();
        let checker = FastChecker::new(&offsets);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xC9_3D);
        for _ in 0..20_000 {
            let pattern: u32 = rng.gen_range(0..(1 << 26));
            let mut object = [false; 26];
            for (i, bit) in object.iter_mut().enumerate() {
                *bit = (pattern >> i) & 1 == 1;
            }
            assert_eq!(
                checker.is_simple(pattern),
                is_simple_point(&offsets, &object),
                "mismatch at pattern {pattern:#08x}"
            );
        }
    }

    /// The production direct-computation accessor (`direct_checker_3d`,
    /// cached like `simple_point_table_3d`) must agree with the table on a
    /// large random sample — the same property
    /// `fast_checker_matches_slow_reference_on_random_3d_patterns`
    /// establishes for a freshly-built `FastChecker`, re-checked here
    /// against the actual cached singletons both production code paths
    /// would use. `#[ignore]`d: unlike the sibling test above, this one
    /// calls `simple_point_table_3d()`, which actually builds the full
    /// `2^26`-entry table — on the order of seconds in release, but slow
    /// enough in an unoptimized debug build to make the fast suite
    /// unusable; no other non-ignored test currently triggers table
    /// generation (`connectivity.rs`'s 3D tests all pass `losing_cell = 0`,
    /// which exempts medium before ever reaching the table).
    #[test]
    #[ignore = "slow in debug: builds the full 2^26-entry 3D table, run with --release -- --ignored"]
    fn direct_checker_3d_matches_table_on_random_patterns() {
        let table = simple_point_table_3d();
        let direct = direct_checker_3d();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD1_EC7);
        for _ in 0..20_000 {
            let pattern: u32 = rng.gen_range(0..(1 << 26));
            assert_eq!(
                table.is_simple(pattern),
                direct.is_simple(pattern),
                "table/direct mismatch at pattern {pattern:#08x}"
            );
        }
    }

    /// Hand-built cases, 3D.
    #[test]
    fn hand_built_3d_cases() {
        let offsets = canonical_offsets_3d();
        let idx = |o: [i32; 3]| offsets.iter().position(|&x| x == o).unwrap();

        // A straight 6-connected line *through* x: rejected. `(1,0,0)` and
        // `(-1,0,0)` are only mutually connected via x itself, so removing x
        // splits the line into two pieces — the 3D analogue of
        // `reference_rejects_face_neighbours_not_mutually_adjacent`.
        let mut object = [false; 26];
        object[idx([1, 0, 0])] = true;
        object[idx([-1, 0, 0])] = true;
        assert!(!is_simple_point(&offsets, &object));

        // A 2-site cell `{x, (1,0,0)}`: simple (removing x leaves a valid
        // 1-site cell, same shape as `reference_accepts_a_two_site_cell`).
        let mut object = [false; 26];
        object[idx([1, 0, 0])] = true;
        assert!(is_simple_point(&offsets, &object));

        // Two face-neighbours only reachable from each other via a shared
        // edge-neighbour: simple (this is exactly the case a naive
        // "only look at the 6 face neighbours" check would get wrong).
        let mut object = [false; 26];
        object[idx([1, 0, 0])] = true;
        object[idx([0, 1, 0])] = true;
        object[idx([1, 1, 0])] = true; // edge-neighbour linking the two
        assert!(is_simple_point(&offsets, &object));

        // Same two face-neighbours, no linking edge-neighbour: pinch-off,
        // rejected.
        let mut object = [false; 26];
        object[idx([1, 0, 0])] = true;
        object[idx([0, 1, 0])] = true;
        assert!(!is_simple_point(&offsets, &object));

        // Diagonal-only contact (a single corner neighbour, no face
        // neighbour at all): the object side has nothing touching the
        // origin under face adjacency, so it's rejected regardless of the
        // background.
        let mut object = [false; 26];
        object[idx([1, 1, 1])] = true;
        assert!(!is_simple_point(&offsets, &object));
    }
}
