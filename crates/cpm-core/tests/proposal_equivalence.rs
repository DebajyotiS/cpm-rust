//! Statistical equivalence of the two proposal modes
//! (`config::ProposalMode`). `EdgeList` is provably the same chain as
//! `Uniform` with null (same-cell) attempts removed via binomial thinning —
//! see `ProposalMode`'s own doc comment — but a "provably the same chain"
//! claim is worth checking directly rather than trusting the argument on
//! its own. Individual trajectories are *not* expected to match (different
//! RNG draw order between the two modes for the same seed) — only their
//! distributions are.
//!
//! This is also the test that resolved a real investigation during
//! development: `checker.rs`'s brute-force consistency test tripped the
//! unwrapped-position half-box guard under `EdgeList` at an MCS budget
//! that `Uniform` handles safely on the same fixture. Direct comparison
//! (recorded in that investigation, not repeated here) showed `Uniform`
//! trips the identical guard, on the identical seed, once given a matched
//! real-attempt budget — i.e. not an `EdgeList` bug, just `EdgeList`
//! reaching the same amount of real dynamics in far fewer raw MCS (it never
//! spends attempts on sites that could only ever early-out). This test
//! formalises that finding: both modes should reach a statistically
//! indistinguishable *conservative energy* after the same MCS budget.

use cpm_core::cell::CellType;
use cpm_core::config::{ProposalMode, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::run_mcs;

const REPLICATES: u64 = 60;
const MCS_BUDGET: u64 = 15;

fn config(seed: u64, proposal: ProposalMode) -> CPM<2> {
    let mut resolved = UserConfig::<2> {
        grid: Some([30, 30]),
        // Fixed, not Periodic: this test's own random seed sweep (120
        // distinct seeds across both arms) hit `dynamics.rs`'s
        // unwrapped-position half-box guard often enough on a Periodic 30x30
        // grid to make the test itself flaky (a property of scatter_and_grow's
        // random *placement* landing close to the guard on this small a grid,
        // independent of proposal mode or MCS budget — see this file's own
        // investigation note above). That guard only exists for `Periodic`
        // unwrapped-position tracking; `Fixed` sidesteps it entirely and
        // doesn't affect what this test is actually checking (proposal-mode
        // equivalence, not periodic-boundary behaviour).
        boundary: Some(Boundary::Fixed),
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
    resolved.proposal = proposal;
    CPM::new(resolved).unwrap()
}

/// Independent replicates (distinct seeds, not a seed shared across modes —
/// this compares *distributions*, not paired trajectories) of
/// `conservative_energy` after `MCS_BUDGET` MCS, starting from a fresh
/// `scatter_and_grow` placement each time.
fn sample_final_energies(proposal: ProposalMode, seed_offset: u64) -> Vec<f64> {
    (0..REPLICATES)
        .map(|i| {
            let mut cpm = config(seed_offset + i, proposal);
            initialize(&mut cpm).expect("fixture must initialise");
            run_mcs(&mut cpm, MCS_BUDGET);
            cpm.state.conservative_energy
        })
        .collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], m: f64) -> f64 {
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

/// Welch's t-statistic for two independent samples with unequal variance —
/// the standard test for "do these two samples plausibly come from
/// distributions with the same mean," without assuming equal variance
/// between the two proposal modes (no reason to assume that a priori).
fn welch_t_statistic(a: &[f64], b: &[f64]) -> f64 {
    let (mean_a, mean_b) = (mean(a), mean(b));
    let (var_a, var_b) = (variance(a, mean_a), variance(b, mean_b));
    let se = (var_a / a.len() as f64 + var_b / b.len() as f64).sqrt();
    (mean_a - mean_b) / se
}

#[test]
fn edge_list_and_uniform_reach_statistically_indistinguishable_energy() {
    let uniform = sample_final_energies(ProposalMode::Uniform, 1_000);
    let edge_list = sample_final_energies(ProposalMode::EdgeList, 2_000);

    let t = welch_t_statistic(&uniform, &edge_list);
    // |t| > ~3.5 would be a strong (p < 0.001-ish, two-sided) signal the two
    // modes are sampling different distributions, at 60 replicates per arm.
    // This is deliberately generous rather than a tight significance
    // threshold: the point is catching a real implementation bug (the kind
    // investigated in this file's own doc comment), not flagging ordinary
    // sampling noise as a failure.
    assert!(
        t.abs() < 3.5,
        "proposal modes diverge: Welch's t = {t:.3} \
         (uniform mean {:.4} over {} samples, edge_list mean {:.4} over {} samples)",
        mean(&uniform),
        uniform.len(),
        mean(&edge_list),
        edge_list.len()
    );
}
