//! Exact Boltzmann validation. With `MetropolisHastings` acceptance,
//! connectivity disabled, and only conservative terms active, the chain is
//! provably reversible with respect to `exp(-H)/Z` — the only test in the
//! suite that checks the energy *expressions* against an independently
//! computed ground truth, as opposed to the brute-force checker, which
//! checks incremental bookkeeping against a from-scratch recomputation
//! using the *same* formula.
//!
//! For that reason `H` is computed here by a small, deliberately
//! independent implementation — hardcoded Moore offsets, a direct
//! popcount/neighbour-count pass, no import of `cpm_core::energy` or
//! `cpm_core::neighborhood` — rather than reusing `Terms`. If the two ever
//! disagree, this test is supposed to catch it; reusing the code under test
//! to also generate the reference would defeat the point.
//!
//! ## Binned by exact energy, not raw microstate
//!
//! The single-cell case has `2^25` microstates, the two-cell case `3^16`.
//! Visiting every one of either enough times for a meaningful chi-squared
//! bin is not something a test that finishes in a reasonable release-mode
//! run can do. Every microstate sharing an exact `H` value is visited with
//! equal probability under `exp(-H)/Z`, so grouping by `H` and comparing the
//! empirical histogram against the *exactly enumerated* degeneracy `g(H)` is
//! still an exact test — it is the standard "enumerate the degeneracy
//! explicitly" alternative to enumerating raw microstates, generalised from
//! volume alone to the full energy `H` (which folds in interface and
//! adhesion too). `H` here is a single direct computation (never an
//! accumulated running sum), so binning by its literal `f64` bits is safe —
//! unlike `State::conservative_energy`, there is no repeated floating-point
//! addition to drift.
//!
//! Run with `cargo test --release -p cpm-core --test exact_boltzmann -- --ignored`;
//! these are `#[ignore]`d (tens of millions of enumerations, not
//! debug-build-friendly) and ~2M-sample runs.

use cpm_core::cell::{Cell, CellType};
use cpm_core::config::{AcceptanceMode, UserConfig};
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::attempt;
use cpm_core::state::{recompute_contact, State};
use std::collections::HashMap;

// ---------------------------------------------------------------------
// Shared chi-squared comparison
// ---------------------------------------------------------------------

/// Standard practice for a chi-squared pooling floor is 5; `20` is a
/// modest, deliberate margin above that, not a workaround for anything.
/// It was not always this tight: while the two-cell test's original
/// (unfixed) energy parameters left the chain unable to explore its tail
/// at all, this constant was pushed up as high as 300 to paper over that —
/// masking exactly the kind of misbinning signature a floor this size
/// exists to catch. Once the real cause was diagnosed (parameter-driven
/// trapping, not a formula or binning bug — see `two_cell_params()`'s doc
/// comment) and fixed by choosing gentler parameters and adding
/// state-space coverage checks instead, this was brought back down and
/// verified empirically at each step: 300 -> 50 -> 20, rerunning both
/// `single_cell_matches_exact_boltzmann_distribution` and
/// `two_cell_matches_exact_boltzmann_distribution` in full at each value,
/// all passing cleanly at 20 on 2026-09-15. Going lower than 20 was not
/// attempted; 20 already sits below the textbook-adjacent value most
/// treatments consider safe. Bins below this floor are pooled into a
/// single "other" bin so the aggregate statistic stays valid; pooling the
/// tail compares its *aggregate* probability instead of any individual
/// rare bin, which is far more sample-efficient.
const MIN_EXPECTED: f64 = 20.0;

/// Expected count per `H` bin under the exact distribution, scaled to
/// `total_samples`, sorted largest first (so pooling collects the smallest
/// tail contributions first). Shared by the chi-squared statistic and the
/// coverage checks below so they agree, by construction, on what "expected"
/// means for a given bin.
fn expected_counts(degeneracy: &HashMap<u64, u64>, total_samples: u64) -> Vec<(u64, f64)> {
    let z: f64 = degeneracy
        .iter()
        .map(|(&bits, &count)| count as f64 * (-f64::from_bits(bits)).exp())
        .sum();
    let mut bins: Vec<(u64, f64)> = degeneracy
        .iter()
        .map(|(&bits, &count)| {
            let h = f64::from_bits(bits);
            let p = (count as f64 * (-h).exp()) / z;
            (bits, p * total_samples as f64)
        })
        .collect();
    bins.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    bins
}

/// Compare an empirical histogram (`H` bits -> visit count) against an
/// exact degeneracy table (`H` bits -> microstate count) via chi-squared,
/// merging any bin whose expected count falls below `MIN_EXPECTED` into a
/// single pooled "other" bin so the test statistic stays valid, plus two
/// coverage checks that chi-squared on its own can miss (a bin pooled into
/// the tail, or a bin below the chi-squared floor but still individually
/// large, can go completely unvisited without moving the aggregate
/// statistic much). `min_distinct_bins` is the caller's floor for how much
/// of the energy landscape a healthy run of this chain, at these
/// parameters, should end up visiting — see the callers for how that
/// number was chosen.
fn assert_matches_exact_distribution(
    degeneracy: &HashMap<u64, u64>,
    empirical: &HashMap<u64, u64>,
    total_states: u64,
    total_samples: u64,
    min_distinct_bins: usize,
) {
    let bins = expected_counts(degeneracy, total_samples);
    assert_no_unexplained_empty_bins(&bins, empirical);
    assert_minimum_state_space_coverage(empirical, min_distinct_bins);

    let (chi_squared, degrees_of_freedom) = chi_squared_statistic(&bins, empirical);
    // A textbook independent-sample envelope (mean `df` plus a handful of
    // standard deviations, `sqrt(2 df)`) is unrealistically strict here:
    // even after thinning, consecutive kept samples retain some residual
    // autocorrelation. `8 * df` is a deliberately blunt, generous multiple
    // of the true mean, still tight enough to catch a formula-sized
    // (order-of-magnitude) error — which is exactly the class of bug this
    // test exists to catch (it caught one during development: the MH
    // acceptance rule was wrongly short-circuiting to "always accept" for
    // `delta_H <= 0`, before `monte_carlo.rs` was fixed).
    let df = degrees_of_freedom as f64;
    let envelope = 8.0 * df;
    assert!(
        chi_squared < envelope,
        "chi-squared statistic {chi_squared:.1} exceeds the {envelope:.1} envelope \
         for {degrees_of_freedom} degrees of freedom ({total_states} states, {total_samples} samples) — \
         the empirical distribution does not match exp(-H)/Z",
    );
}

/// A bin with expected count >= 20 coming back with **zero** observations
/// is not sampling noise: under Poisson, `P(0 hits | expected=20) = e^-20 ~
/// 2e-9`. It means that bin's states were never *visited*, a materially
/// different failure from "visited but with a noisy count," and one
/// chi-squared alone can miss when the bin is small enough to fall below
/// `MIN_EXPECTED` and get pooled into the tail. This is what would have
/// caught this test's own history directly: the two-cell case originally
/// used energy parameters strong enough that the sampler explored only
/// ~1,225 of 42,915,650 reachable states in 20M attempts — nowhere near
/// enough for bins like this to be observable at all.
fn assert_no_unexplained_empty_bins(bins: &[(u64, f64)], empirical: &HashMap<u64, u64>) {
    const COVERAGE_FLOOR: f64 = 20.0;
    let empty_but_expected: Vec<(f64, f64)> = bins
        .iter()
        .filter(|&&(bits, expected)| {
            expected >= COVERAGE_FLOOR && empirical.get(&bits).copied().unwrap_or(0) == 0
        })
        .map(|&(bits, expected)| (f64::from_bits(bits), expected))
        .collect();
    assert!(
        empty_but_expected.is_empty(),
        "{} bin(s) with expected count >= {COVERAGE_FLOOR} were never observed at all: {:?} — \
         these states are reachable (verified separately) and individually near-impossible to \
         miss by chance; the sampler and the exact enumeration disagree about something",
        empty_but_expected.len(),
        empty_but_expected
    );
}

/// Distinct-bin coverage is a coarser, complementary signal to the
/// per-bin check above: it catches the case where the *whole run* got
/// trapped in a small, low-energy corner of the landscape — every bin the
/// chain did visit matches perfectly, but most of the landscape (packed
/// into bins individually too small to trip the >= 20 floor) was never
/// explored at all, so chi-squared has nothing to compare there. This is
/// the guard against silently drifting back into that regime if someone
/// re-tunes the energy parameters later: raising them stiffens the
/// landscape and shrinks this number well before any other assertion here
/// would notice.
fn assert_minimum_state_space_coverage(empirical: &HashMap<u64, u64>, min_distinct_bins: usize) {
    let distinct = empirical.len();
    assert!(
        distinct >= min_distinct_bins,
        "chain visited only {distinct} distinct energy bins (wanted >= {min_distinct_bins}) — \
         the landscape may be too peaked at these parameters for this test to be meaningful; \
         see `two_cell_params_at_scale`'s doc comment"
    );
}

/// Returns `(chi_squared, degrees_of_freedom)` — the statistic only, no
/// assertion, so callers can aggregate across seeds (a single seed's
/// finite-sample noise is not, by itself, a reliable pass/fail signal for a
/// stochastic chain). Takes the same pre-sorted `bins` as the coverage
/// checks so every check in this file agrees on "expected count."
fn chi_squared_statistic(bins: &[(u64, f64)], empirical: &HashMap<u64, u64>) -> (f64, i64) {
    let mut chi_squared = 0.0;
    let mut degrees_of_freedom = 0i64;
    let mut pooled_expected = 0.0;
    let mut pooled_observed = 0.0;

    if std::env::var("BOLTZMANN_DEBUG").is_ok() {
        for (bits, expected) in bins.iter().take(15) {
            let observed = *empirical.get(bits).unwrap_or(&0) as f64;
            eprintln!(
                "H={:>10.4}  expected={:>10.1}  observed={:>10.1}  ratio={:.3}",
                f64::from_bits(*bits),
                expected,
                observed,
                observed / expected.max(1.0)
            );
        }
        eprintln!("distinct bins visited: {}", empirical.len());
    }

    for (bits, expected) in bins {
        if *expected < MIN_EXPECTED {
            pooled_expected += expected;
            pooled_observed += *empirical.get(bits).unwrap_or(&0) as f64;
            continue;
        }
        let observed = *empirical.get(bits).unwrap_or(&0) as f64;
        chi_squared += (observed - expected).powi(2) / expected;
        degrees_of_freedom += 1;
    }
    if pooled_expected >= MIN_EXPECTED {
        chi_squared += (pooled_observed - pooled_expected).powi(2) / pooled_expected;
        degrees_of_freedom += 1;
    }
    degrees_of_freedom -= 1; // one constraint: probabilities sum to 1
    assert!(
        degrees_of_freedom >= 1,
        "not enough well-populated bins ({degrees_of_freedom} df) for a meaningful test; \
         increase total_samples or widen the energy landscape"
    );
    (chi_squared, degrees_of_freedom)
}

// ---------------------------------------------------------------------
// Single cell + medium, 5x5: validates volume and interface
// ---------------------------------------------------------------------

const GRID: usize = 5;
const SITES: usize = GRID * GRID;

fn moore_neighbours_5x5() -> [[Option<usize>; 8]; SITES] {
    const OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut table = [[None; 8]; SITES];
    for r in 0..GRID as i32 {
        for c in 0..GRID as i32 {
            let idx = (r * GRID as i32 + c) as usize;
            for (k, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let (nr, nc) = (r + dr, c + dc);
                if (0..GRID as i32).contains(&nr) && (0..GRID as i32).contains(&nc) {
                    table[idx][k] = Some((nr * GRID as i32 + nc) as usize);
                }
            }
        }
    }
    table
}

fn h_single_cell(
    v: u32,
    interface: u32,
    lambda_v: f64,
    v_star: f64,
    lambda_i: f64,
    i_star: f64,
) -> f64 {
    lambda_v * (v as f64 - v_star).powi(2) + lambda_i * (interface as f64 - i_star).powi(2)
}

/// Exact degeneracy over all `2^25 - 1` nonempty single-cell patterns
/// (pattern 0, `V = 0`, is excluded: the minimum-volume-1 rule makes it
/// unreachable).
fn exact_degeneracy_single_cell(
    lambda_v: f64,
    v_star: f64,
    lambda_i: f64,
    i_star: f64,
) -> HashMap<u64, u64> {
    let neighbours = moore_neighbours_5x5();
    let mut degeneracy: HashMap<u64, u64> = HashMap::new();
    for pattern in 1u32..(1 << SITES) {
        let v = pattern.count_ones();
        let mut interface = 0u32;
        let mut remaining = pattern;
        while remaining != 0 {
            let site = remaining.trailing_zeros() as usize;
            remaining &= remaining - 1;
            for slot in neighbours[site] {
                match slot {
                    Some(n) => {
                        if (pattern >> n) & 1 == 0 {
                            interface += 1;
                        }
                    }
                    None => interface += 1,
                }
            }
        }
        let h = h_single_cell(v, interface, lambda_v, v_star, lambda_i, i_star);
        *degeneracy.entry(h.to_bits()).or_insert(0) += 1;
    }
    degeneracy
}

fn single_cell_config(seed: u64, lambda_v: f64, v_star: f64, lambda_i: f64, i_star: f64) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([GRID, GRID]),
        boundary: Some(Boundary::Fixed),
        acceptance: Some(AcceptanceMode::MetropolisHastings),
        cell_types: vec![CellType {
            name: "a".into(),
            target_volume: v_star.round() as u32,
            target_interface: i_star.round() as u32,
            lambda_volume: lambda_v,
            lambda_interface: lambda_i,
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![1],
        adhesion: Some(vec![vec![0.0, 0.0], vec![0.0, 0.0]]), // isolate volume+interface
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    // Seed with a single site; connectivity is disabled for this test, so
    // the chain is free to grow, shrink and reshape from here.
    let lattice = cpm.state.lattice.clone();
    let start = lattice.flat_index([2, 2]);
    let mut lattice = lattice;
    lattice.set(start, 1);
    cpm.state = State::with_cells(
        lattice,
        vec![Cell {
            id: 1,
            type_index: 0,
        }],
    );
    cpm.state.volume = vec![1];
    cpm.state.interface = vec![8]; // a lone site has all 8 Moore neighbours as medium
                                   // Never went through `dynamics::commit`, so the contact graph needs
                                   // establishing from scratch too before the sampler's first accepted
                                   // move tries to update it incrementally.
    cpm.state.contact = recompute_contact(
        &cpm.state.lattice,
        &cpm.energy_offsets,
        cpm.state.cells.len(),
    );
    cpm
}

#[test]
#[ignore = "slow: 2^25 enumeration + 2M MC samples, run with --release -- --ignored"]
fn single_cell_matches_exact_boltzmann_distribution() {
    let (lambda_v, v_star, lambda_i, i_star) = (0.15, 8.0, 0.05, 20.0);
    let degeneracy = exact_degeneracy_single_cell(lambda_v, v_star, lambda_i, i_star);

    let mut cpm = single_cell_config(1, lambda_v, v_star, lambda_i, i_star);
    const BURN_IN: u64 = 200_000;
    const SAMPLES: u64 = 2_000_000;
    // Sample once per `SITES` attempts (roughly one MCS), not every single
    // attempt: consecutive attempts differ by at most one site, so raw
    // per-attempt samples are highly autocorrelated, and chi-squared
    // assumes independent samples. Thinning to ~one sample per MCS is a
    // standard MCMC decorrelation step.
    const THIN: u64 = 40 * SITES as u64;
    for _ in 0..BURN_IN {
        attempt(&mut cpm, false); // connectivity disabled
    }
    let mut empirical: HashMap<u64, u64> = HashMap::new();
    for _ in 0..SAMPLES {
        for _ in 0..THIN {
            attempt(&mut cpm, false);
        }
        let h = h_single_cell(
            cpm.state.volume[0],
            cpm.state.interface[0],
            lambda_v,
            v_star,
            lambda_i,
            i_star,
        );
        *empirical.entry(h.to_bits()).or_insert(0) += 1;
    }

    // Calibrated on 2026-09-15 from an actual run at these parameters (37
    // distinct bins visited, via `BOLTZMANN_DEBUG=1`'s "distinct bins
    // visited" line). `H` here is a function of only two integers (`V`,
    // `interface`), so even with ~33M microstates the number of distinct
    // *energies* is inherently small — most of that 37 is the
    // well-populated core the chain equilibrates around, not a sign of
    // poor mixing. Headroom below the observed count, not below some
    // larger assumed figure. If a future run reports materially fewer than
    // 37, that is a regression signal, not a reason to lower this further.
    const MIN_DISTINCT_BINS: usize = 25;
    assert_matches_exact_distribution(
        &degeneracy,
        &empirical,
        (1u64 << SITES) - 1,
        SAMPLES,
        MIN_DISTINCT_BINS,
    );
}

// ---------------------------------------------------------------------
// Two cells + medium, 4x4: adds adhesion
// ---------------------------------------------------------------------

const GRID2: usize = 4;
const SITES2: usize = GRID2 * GRID2;

fn moore_neighbours_4x4() -> [[Option<usize>; 8]; SITES2] {
    const OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut table = [[None; 8]; SITES2];
    for r in 0..GRID2 as i32 {
        for c in 0..GRID2 as i32 {
            let idx = (r * GRID2 as i32 + c) as usize;
            for (k, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let (nr, nc) = (r + dr, c + dc);
                if (0..GRID2 as i32).contains(&nr) && (0..GRID2 as i32).contains(&nc) {
                    table[idx][k] = Some((nr * GRID2 as i32 + nc) as usize);
                }
            }
        }
    }
    table
}

struct TwoCellParams {
    lambda_v: [f64; 2],
    v_star: [f64; 2],
    lambda_i: [f64; 2],
    i_star: [f64; 2],
    /// Symmetric 3x3 adhesion matrix, index 0 = medium.
    adhesion: [[f64; 3]; 3],
}

/// Deliberately gentle energy parameters — not the first choice tried, and
/// not, on their own, a wide test of the energy landscape. Stronger values
/// (`lambda_v: [0.2, 0.2]`, `adhesion` magnitudes up to 2.5) were used
/// originally and, per `h_two_cell_matches_production_on_random_patterns`
/// (10,000 random patterns, exact agreement) and a from-real-sampler
/// reachability check, produced a *correctly-computed* equilibrium
/// distribution so sharply peaked that the real sampler visited only
/// ~1,225 of the 42,915,650 valid two-cell states in 20M attempts — nowhere
/// near enough to check the tail via sampling in a test that has to finish
/// in a reasonable time. These weaker values raised that to ~84,000 distinct
/// *patterns* in the same attempt budget (see
/// `crates/cpm-core/examples/boltzmann_diagnose.rs`, a one-off diagnostic
/// kept for anyone re-tuning this test) — a 69x improvement, but still only
/// ~0.2% of the 42.9M reachable two-cell states. At `scale = 1.0` this test
/// is validating the energy expressions over a small, low-energy corner of
/// configuration space; it would not, by itself, reliably catch a bug that
/// only manifests in rare, ragged, high-interface configurations. The
/// somewhere-between-0.4-and-1.0 multiplier is roughly where the cliff
/// is — see `two_cell_params_at_scale`'s doc comment for the tempering
/// counterpart that stresses the stiffer end instead. The energy
/// *expressions* being validated don't depend on the parameter magnitudes;
/// only how practical it is to sample their stationary distribution does.
fn two_cell_params() -> TwoCellParams {
    two_cell_params_at_scale(1.0)
}

/// `two_cell_params()` scaled by `scale`: `lambda_v`, `lambda_i` and every
/// adhesion coefficient are multiplied by it, while `v_star`/`i_star` (which
/// set *positions*, not energy magnitudes) are left alone. `scale = 1.0` is
/// the gentle baseline documented on `two_cell_params()`; `scale > 1.0`
/// stiffens the landscape, trading state-space coverage for a more sharply
/// peaked distribution that stresses the `exp(-H)` weighting where it is
/// doing real work — the two failure modes a single fixed parameter set
/// cannot exercise at once. Run both ends (see
/// `two_cell_matches_exact_boltzmann_distribution_at_stiffer_scales`) rather
/// than either alone.
fn two_cell_params_at_scale(scale: f64) -> TwoCellParams {
    TwoCellParams {
        lambda_v: [0.05 * scale, 0.05 * scale],
        v_star: [5.0, 5.0],
        lambda_i: [0.01 * scale, 0.01 * scale],
        i_star: [14.0, 14.0],
        adhesion: [
            [0.0, 0.5 * scale, 0.5 * scale],
            [0.5 * scale, 0.2 * scale, 0.8 * scale],
            [0.5 * scale, 0.8 * scale, 0.2 * scale],
        ],
    }
}

/// `pattern[site] in {0, 1, 2}` (medium, cell 1, cell 2), base-3 digits of a
/// `u32` over `SITES2` sites.
fn h_two_cell(
    pattern_digits: &[u8; SITES2],
    neighbours: &[[Option<usize>; 8]; SITES2],
    p: &TwoCellParams,
) -> f64 {
    let mut volume = [0u32; 2];
    let mut interface = [0u32; 2];
    let mut adhesion_energy = 0.0;
    for site in 0..SITES2 {
        let owner = pattern_digits[site];
        if owner != 0 {
            volume[owner as usize - 1] += 1;
        }
        for slot in neighbours[site] {
            let neighbour_owner = match slot {
                Some(n) => pattern_digits[n],
                None => 0,
            };
            if neighbour_owner != owner {
                if owner != 0 {
                    interface[owner as usize - 1] += 1;
                }
                // Each unordered pair is visited from both sides (or once,
                // for an out-of-grid pair); halve at the end instead of
                // trying to dedupe here, matching the "sum over pairs"
                // adhesion definition without fragile visited-pair
                // bookkeeping.
                adhesion_energy += p.adhesion[owner as usize][neighbour_owner as usize];
            }
        }
    }
    let adhesion_energy = adhesion_energy / 2.0
        + out_of_grid_adhesion_correction(pattern_digits, neighbours, p) / 2.0;
    let mut h = adhesion_energy;
    for c in 0..2 {
        h += p.lambda_v[c] * (volume[c] as f64 - p.v_star[c]).powi(2);
        h += p.lambda_i[c] * (interface[c] as f64 - p.i_star[c]).powi(2);
    }
    h
}

/// Out-of-grid neighbours are only ever visited from the in-grid side
/// (there is no reciprocal halo site to double-count from), so the blanket
/// `/2.0` in [`h_two_cell`] would under-count them by half. This adds a
/// second copy of exactly those contributions so the halving evens out.
fn out_of_grid_adhesion_correction(
    pattern_digits: &[u8; SITES2],
    neighbours: &[[Option<usize>; 8]; SITES2],
    p: &TwoCellParams,
) -> f64 {
    let mut total = 0.0;
    for site in 0..SITES2 {
        let owner = pattern_digits[site];
        for slot in neighbours[site] {
            if slot.is_none() && owner != 0 {
                total += p.adhesion[owner as usize][0];
            }
        }
    }
    total
}

fn digits_of(mut pattern: u32) -> [u8; SITES2] {
    let mut digits = [0u8; SITES2];
    for slot in digits.iter_mut() {
        *slot = (pattern % 3) as u8;
        pattern /= 3;
    }
    digits
}

fn exact_degeneracy_two_cell(p: &TwoCellParams) -> HashMap<u64, u64> {
    let neighbours = moore_neighbours_4x4();
    let total_patterns = 3u32.pow(SITES2 as u32);
    let mut degeneracy: HashMap<u64, u64> = HashMap::new();
    for pattern in 0..total_patterns {
        let digits = digits_of(pattern);
        let has_cell1 = digits.contains(&1);
        let has_cell2 = digits.contains(&2);
        if !has_cell1 || !has_cell2 {
            continue; // both cells must have volume >= 1
        }
        let h = h_two_cell(&digits, &neighbours, p);
        *degeneracy.entry(h.to_bits()).or_insert(0) += 1;
    }
    degeneracy
}

fn two_cell_config(seed: u64, p: &TwoCellParams) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([GRID2, GRID2]),
        boundary: Some(Boundary::Fixed),
        acceptance: Some(AcceptanceMode::MetropolisHastings),
        cell_types: vec![
            CellType {
                name: "a".into(),
                target_volume: p.v_star[0].round() as u32,
                target_interface: p.i_star[0].round() as u32,
                lambda_volume: p.lambda_v[0],
                lambda_interface: p.lambda_i[0],
                lambda_act: 0.0,
                max_act: 0,
            },
            CellType {
                name: "b".into(),
                target_volume: p.v_star[1].round() as u32,
                target_interface: p.i_star[1].round() as u32,
                lambda_volume: p.lambda_v[1],
                lambda_interface: p.lambda_i[1],
                lambda_act: 0.0,
                max_act: 0,
            },
        ],
        cell_counts: vec![1, 1],
        adhesion: Some(p.adhesion.iter().map(|row| row.to_vec()).collect()),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    let mut lattice = cpm.state.lattice.clone();
    let a = lattice.flat_index([1, 1]);
    let b = lattice.flat_index([2, 2]);
    lattice.set(a, 1);
    lattice.set(b, 2);
    cpm.state = State::with_cells(
        lattice,
        vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ],
    );
    cpm.state.volume = vec![1, 1];
    cpm.state.interface = vec![8, 8]; // diagonal-adjacent lone sites, both fully medium-bounded
    cpm.state.contact = recompute_contact(
        &cpm.state.lattice,
        &cpm.energy_offsets,
        cpm.state.cells.len(),
    );
    cpm
}

#[test]
#[ignore = "slow: 3^16 enumeration + 2M MC samples, run with --release -- --ignored"]
fn two_cell_matches_exact_boltzmann_distribution() {
    // Calibrated on 2026-09-15 from an actual run at `scale = 1.0` (833
    // distinct energy bins visited, via `BOLTZMANN_DEBUG=1`'s "distinct
    // bins visited" line) — note this is *bins*, not raw microstates: the
    // ~0.2% figure in `two_cell_params()`'s doc comment is over the
    // 42.9M-state space, a different and much stricter denominator.
    // Headroom below the observed bin count, not below some larger assumed
    // figure. A future run reporting materially fewer than 833 is a
    // regression signal, not a reason to lower this further.
    const MIN_DISTINCT_BINS: usize = 500;
    run_two_cell_boltzmann_check(2, 2_000_000, 2_000_000, MIN_DISTINCT_BINS);
}

/// Runs the full enumerate/sample/compare pipeline at one energy scale and
/// asserts the result. Shared by the primary (`scale = 1.0`) test above and
/// the stiffer-scale tempering test below.
fn run_two_cell_boltzmann_check(
    seed: u64,
    burn_in: u64,
    samples: u64,
    min_distinct_bins: usize,
) -> f64 {
    run_two_cell_boltzmann_check_at_scale(1.0, seed, burn_in, samples, min_distinct_bins)
}

fn run_two_cell_boltzmann_check_at_scale(
    scale: f64,
    seed: u64,
    burn_in: u64,
    samples: u64,
    min_distinct_bins: usize,
) -> f64 {
    let params = two_cell_params_at_scale(scale);
    let degeneracy = exact_degeneracy_two_cell(&params);

    let mut cpm = two_cell_config(seed, &params);
    // See the single-cell test: thin to decorrelate consecutive samples
    // before comparing via chi-squared. Two interacting cells mix more
    // slowly than one (more ways to rearrange the same aggregate V/I/contact
    // statistics), so this needs a larger multiple of a single MCS than the
    // single-cell case did.
    let thin = 200 * SITES2 as u64;
    let empirical = sample_two_cell_chain(&mut cpm, burn_in, samples, thin, &params);

    let total_states = degeneracy.values().sum();
    assert_matches_exact_distribution(
        &degeneracy,
        &empirical,
        total_states,
        samples,
        min_distinct_bins,
    );
    empirical.len() as f64
}

/// Tempering: `two_cell_matches_exact_boltzmann_distribution` above uses
/// gentle parameters (`scale = 1.0`) chosen for state-space coverage, which
/// leaves the stiffer end of the landscape — where `exp(-H)` swings over
/// many orders of magnitude between neighbouring bins — mostly unexercised.
/// Running the same exact check at `scale = 1.5` and `scale = 2.0` covers
/// that end instead, while the chain still mixes enough for the test to
/// carry real statistical power: `scale = 4.0` was tried too (see git
/// history / the earlier version of this test) and rejected for this
/// purpose — only 7 distinct energy bins were visited in 1M samples at that
/// stiffness, leaving 2-3 degrees of freedom after pooling, and a
/// chi-squared with that few degrees of freedom against an `8 * df`
/// envelope catches almost nothing. That scale is trapped in the same way
/// the *original* (pre-fix) two-cell parameters were, and passing there
/// would mean "nothing to disagree about," not "validated." It is kept as
/// a separate, explicitly-labelled smoke test instead — see
/// `two_cell_sampler_survives_very_stiff_parameters_smoke_test` below —
/// rather than folded into this one where a green result could be
/// misread as coverage of the peaked regime. `MIN_DISTINCT_BINS` is
/// lowered per scale to match observed coverage, but the chi-squared and
/// no-unexplained-empty-bin checks still apply in full to however much of
/// the landscape the chain does reach. Each scale is a separate,
/// independently valid exact test; none of them alone would catch
/// everything the others do.
#[test]
#[ignore = "slow: 2 scales x (3^16 enumeration + 1M samples), run with --release -- --ignored"]
fn two_cell_matches_exact_boltzmann_distribution_at_stiffer_scales() {
    // Calibrated from actual runs on 2026-09-15 (see `BOLTZMANN_DEBUG=1`'s
    // "distinct bins visited" line): coverage falls sharply as `scale`
    // grows — see the per-scale numbers below — so these floors are set
    // with headroom below what was actually observed at each scale, not
    // copied from `scale=1.0`. If a future run reports materially fewer
    // bins than these comments record, treat that as a signal worth
    // investigating (an RNG, move-generator, or energy change upstream),
    // not just a constant to relax back down.
    // scale=1.5: 248 distinct bins observed -> floor 150
    // scale=2.0:  87 distinct bins observed -> floor 50
    for (scale, min_distinct_bins) in [(1.5, 150usize), (2.0, 50usize)] {
        let distinct = run_two_cell_boltzmann_check_at_scale(
            scale,
            2,
            1_000_000,
            1_000_000,
            min_distinct_bins,
        );
        eprintln!("scale={scale}: distinct bins visited={distinct}");
    }
}

/// Not a distribution-matching test: at `scale = 4.0` the chain is back in
/// the trapped state diagnosed for the original (pre-fix) two-cell
/// parameters — 7 distinct energy bins visited in 1M samples, too few for
/// chi-squared to carry any real power (see the doc comment above). A
/// green result here would not mean the sampler matches `exp(-H)/Z` at
/// this stiffness in any meaningful sense; it only confirms the sampler
/// runs to completion, keeps every volume >= 1, and never produces a
/// non-finite energy at parameters this stiff — worth knowing on its own,
/// since production parameter sets can plausibly be this stiff, but not a
/// substitute for the coverage-bearing tests above.
#[test]
#[ignore = "slow, run with --release -- --ignored"]
fn two_cell_sampler_survives_very_stiff_parameters_smoke_test() {
    let params = two_cell_params_at_scale(4.0);
    let mut cpm = two_cell_config(2, &params);
    for _ in 0..500_000 {
        attempt(&mut cpm, false); // connectivity disabled, matching the exact-distribution tests
    }
    assert!(
        cpm.state.volume.iter().all(|&v| v >= 1),
        "a cell was driven below minimum volume: {:?}",
        cpm.state.volume
    );
    let neighbours = moore_neighbours_4x4();
    let mut digits = [0u8; SITES2];
    for (site, slot) in digits.iter_mut().enumerate() {
        let flat = cpm.state.lattice.flat_index([site / GRID2, site % GRID2]);
        *slot = cpm.state.lattice.get(flat) as u8;
    }
    let h = h_two_cell(&digits, &neighbours, &params);
    assert!(h.is_finite(), "energy is non-finite at scale=4.0: {h}");
}

fn sample_two_cell_chain(
    cpm: &mut CPM<2>,
    burn_in: u64,
    samples: u64,
    thin: u64,
    params: &TwoCellParams,
) -> HashMap<u64, u64> {
    for _ in 0..burn_in {
        attempt(cpm, false);
    }
    let neighbours = moore_neighbours_4x4();
    let mut empirical: HashMap<u64, u64> = HashMap::new();
    for _ in 0..samples {
        for _ in 0..thin {
            attempt(cpm, false);
        }
        let mut digits = [0u8; SITES2];
        for (site, slot) in digits.iter_mut().enumerate() {
            let flat = cpm.state.lattice.flat_index([site / GRID2, site % GRID2]);
            *slot = cpm.state.lattice.get(flat) as u8;
        }
        let h = h_two_cell(&digits, &neighbours, params);
        *empirical.entry(h.to_bits()).or_insert(0) += 1;
    }
    empirical
}

/// A single seed's chi-squared is not, on its own, a reliable verdict for a
/// stochastic chain: the same code, run to the same total sample count,
/// gives noticeably different statistics from one seed to the next (this
/// was observed directly while building this test — two runs of the
/// production code under otherwise-identical settings landed at chi-squared
/// values roughly 70 and roughly 300 for the same ~12 degrees of freedom).
/// Averaging several independent seeds is the standard fix: a formula bug
/// biases every seed in the same direction and survives averaging, while
/// per-seed sampling noise mostly cancels — but only if the seeds are
/// actually independent, not correlated draws from one shared stream. Each
/// iteration below builds a fresh `CPM` from a plain `u64`
/// (`two_cell_config` -> `CPM::new` -> `rng_from_seed` ->
/// `Xoshiro256PlusPlus::seed_from_u64`), never sharing or re-deriving state
/// across seeds; `seed_from_u64` is designed (via its internal SplitMix64
/// expansion) to decorrelate even sequential small integers like
/// `[2, 7, 13, 21]` into unrelated full internal states, unlike naively
/// using the seed as raw generator state. This matters here specifically:
/// seed sensitivity is the first symptom of the trapping problem this test
/// exists to catch (see `two_cell_params()`'s doc comment), so a subtle
/// correlation between "independent" replicates would quietly defeat the
/// whole point of averaging across them.
#[test]
#[ignore = "slow: 4 seeds x (3^16 enumeration + 1M samples), run with --release -- --ignored"]
fn two_cell_distribution_across_several_seeds() {
    let params = two_cell_params();
    let degeneracy = exact_degeneracy_two_cell(&params);

    const BURN_IN: u64 = 1_000_000;
    const SAMPLES: u64 = 1_000_000;
    const THIN: u64 = 200 * SITES2 as u64;

    let mut ratios = Vec::new();
    for seed in [2u64, 7, 13, 21] {
        let mut cpm = two_cell_config(seed, &params);
        let empirical = sample_two_cell_chain(&mut cpm, BURN_IN, SAMPLES, THIN, &params);
        let bins = expected_counts(&degeneracy, SAMPLES);
        let (chi_squared, df) = chi_squared_statistic(&bins, &empirical);
        let ratio = chi_squared / df as f64;
        eprintln!("seed {seed}: chi_squared={chi_squared:.1} df={df} ratio={ratio:.2}");
        ratios.push(ratio);
    }

    let mean_ratio: f64 = ratios.iter().sum::<f64>() / ratios.len() as f64;
    eprintln!(
        "mean chi_squared/df across {} seeds: {mean_ratio:.2}",
        ratios.len()
    );
    // An unbiased chain has chi_squared/df averaging to ~1 (plus finite-
    // chain noise). A formula bug biases every seed the same way and would
    // push this mean well above any one seed's noise band; `< 6` is a
    // generous cutoff chosen to comfortably clear realistic residual
    // autocorrelation while still catching a systematic, seed-independent
    // bias.
    assert!(
        mean_ratio < 6.0,
        "mean chi_squared/df = {mean_ratio:.2} across seeds {:?} is too high to be sampling \
         noise alone — this looks like a systematic bias, not per-seed variance",
        ratios
    );
}

/// Diagnostic: cross-check `h_two_cell` (this file's independent formula)
/// against production `Terms::global_energy` for hand-built patterns, to
/// isolate whether a mismatch is in the test harness or in `energy.rs`.
#[test]
fn h_two_cell_matches_terms_global_energy_on_hand_built_patterns() {
    let params = two_cell_params();
    let neighbours = moore_neighbours_4x4();
    let at = |r: usize, c: usize| r * GRID2 + c;

    // Case A: two disjoint singletons, diagonal-adjacent (touch nothing).
    let mut digits_a = [0u8; SITES2];
    digits_a[at(0, 0)] = 1;
    digits_a[at(3, 3)] = 2;

    // Case B: two singletons directly face-adjacent (share an edge).
    let mut digits_b = [0u8; SITES2];
    digits_b[at(1, 1)] = 1;
    digits_b[at(1, 2)] = 2;

    // Case C: 2x2 block cell 1, adjacent 2x1 block cell 2, sharing a face.
    let mut digits_c = [0u8; SITES2];
    for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        digits_c[at(r, c)] = 1;
    }
    digits_c[at(0, 2)] = 2;
    digits_c[at(1, 2)] = 2;

    for (name, digits) in [
        ("A_disjoint", digits_a),
        ("B_adjacent", digits_b),
        ("C_blocks", digits_c),
    ] {
        let independent_h = h_two_cell(&digits, &neighbours, &params);
        let production_h = production_global_energy(&digits, &params);
        eprintln!("{name}: independent={independent_h:.6} production={production_h:.6}");
        assert!(
            (independent_h - production_h).abs() < 1e-9,
            "{name}: independent H {independent_h} != production H {production_h}"
        );
    }
}

/// Deterministic, statistics-free cross-check on 10,000 random 4x4 two-cell
/// patterns: if `h_two_cell` and `Terms::global_energy` ever disagreed by
/// more than an additive constant (harmless — only differences drive
/// acceptance), that would be a formula bug, and this converts the question
/// from "does a stochastic chain's histogram match" into a plain equality
/// check with no sampling noise to explain away.
#[test]
fn h_two_cell_matches_production_on_random_patterns() {
    use rand::Rng as _;
    let params = two_cell_params();
    let neighbours = moore_neighbours_4x4();
    let mut rng = rand::rngs::StdRng::seed_from_u64(0xC9_2C);
    use rand::SeedableRng as _;

    let mut checked = 0;
    let mut offsets: Vec<f64> = Vec::new();
    while checked < 10_000 {
        let mut digits = [0u8; SITES2];
        for slot in digits.iter_mut() {
            *slot = rng.gen_range(0..3u8);
        }
        if !digits.contains(&1) || !digits.contains(&2) {
            continue; // both cells must have volume >= 1
        }
        let independent_h = h_two_cell(&digits, &neighbours, &params);
        let production_h = production_global_energy(&digits, &params);
        offsets.push(production_h - independent_h);
        checked += 1;
    }

    let first = offsets[0];
    for (i, &offset) in offsets.iter().enumerate() {
        assert!(
            (offset - first).abs() < 1e-9,
            "pattern {i}: production - independent offset {offset} differs from the first \
             pattern's offset {first} — this is a formula disagreement, not just a constant"
        );
    }
    eprintln!(
        "checked {checked} random patterns; constant offset (production - independent) = {first:.6}"
    );
    assert!(
        first.abs() < 1e-9,
        "production and independent H agree up to a constant, but that constant is {first}, \
         not 0 — unexpected, since both should compute the same absolute H"
    );
}

fn production_global_energy(digits: &[u8; SITES2], p: &TwoCellParams) -> f64 {
    let config = UserConfig::<2> {
        grid: Some([GRID2, GRID2]),
        boundary: Some(Boundary::Fixed),
        acceptance: Some(AcceptanceMode::MetropolisHastings),
        cell_types: vec![
            CellType {
                name: "a".into(),
                target_volume: p.v_star[0].round() as u32,
                target_interface: p.i_star[0].round() as u32,
                lambda_volume: p.lambda_v[0],
                lambda_interface: p.lambda_i[0],
                lambda_act: 0.0,
                max_act: 0,
            },
            CellType {
                name: "b".into(),
                target_volume: p.v_star[1].round() as u32,
                target_interface: p.i_star[1].round() as u32,
                lambda_volume: p.lambda_v[1],
                lambda_interface: p.lambda_i[1],
                lambda_act: 0.0,
                max_act: 0,
            },
        ],
        cell_counts: vec![1, 1],
        adhesion: Some(p.adhesion.iter().map(|row| row.to_vec()).collect()),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(1),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let neighbours = moore_neighbours_4x4();
    let mut cpm = CPM::new(config).unwrap();
    let mut lattice = cpm.state.lattice.clone();
    let mut volume = [0u32; 2];
    for (site, &owner) in digits.iter().enumerate() {
        if owner != 0 {
            let flat = lattice.flat_index([site / GRID2, site % GRID2]);
            lattice.set(flat, owner as cpm_core::lattice::CellId);
            volume[owner as usize - 1] += 1;
        }
    }
    cpm.state = State::with_cells(
        lattice,
        vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ],
    );
    cpm.state.volume = volume.to_vec();
    let mut interface = [0u32; 2];
    for (site, &owner) in digits.iter().enumerate() {
        if owner == 0 {
            continue;
        }
        for slot in neighbours[site] {
            let neighbour_owner = match slot {
                Some(n) => digits[n],
                None => 0,
            };
            if neighbour_owner != owner {
                interface[owner as usize - 1] += 1;
            }
        }
    }
    cpm.state.interface = interface.to_vec();
    cpm.terms.global_energy(&cpm.state)
}
