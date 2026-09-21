//! Criterion benchmarks for `batch::run_batch`'s parallel scaling and its
//! behaviour under per-item cost heterogeneity — the two things §44's own
//! 9.05x speedup number (measured once, at N=12, homogeneous thetas, from a
//! Python notebook wall-clock comparison) never exercised. Pure Rust, no
//! PyO3, consistent with "Rayon lives in `cpm-core`, not `cpm-py`" — the
//! parallel loop this benchmarks is dimension-generic and testable without
//! Python.
//!
//! Two questions:
//! - **Scaling**: does `run_batch`'s speedup over a sequential loop hold as
//!   `N` grows well past the one data point (12) `run_batch` was originally
//!   validated at?
//! - **Heterogeneity**: `run_batch` shares one `ResolvedConfig` (grid, cell
//!   count, burn-in/readout MCS budget) across every batch member — only
//!   `theta` (per-type stiffnesses, the adhesion matrix) varies per item.
//!   Under `ProposalMode::Uniform` this means every item does almost
//!   exactly the same amount of work regardless of `theta` (`N_sites`
//!   attempts per MCS, a fixed MCS budget, and `theta` mostly affecting the
//!   accept/reject mix rather than attempt count) — there is little real
//!   heterogeneity for Rayon's scheduler to have to handle. Under
//!   `ProposalMode::EdgeList`, though, each MCS instead performs `k ~
//!   Binomial(N_sites, |E| / N_sites)` attempts, and `|E|` — the edge-site
//!   count — depends on tissue morphology, which does depend on `theta`: a
//!   batch spanning soft to stiff tissue can have genuinely different
//!   per-item attempt counts, hence genuinely different per-item wall-clock
//!   cost, all from theta alone. That is the realistic way `run_batch`
//!   actually produces uneven work today, so this benchmarks that case
//!   rather than inventing an artificial one `run_batch`'s current API
//!   can't actually produce.

use cpm_core::batch::{run_batch, BatchOptions};
use cpm_core::cell::CellType;
use cpm_core::config::{ProposalMode, ResolvedConfig, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::output::{self, RunOptions};
use cpm_core::rng::derive_seed_u64;
use cpm_core::theta;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

/// Matches `benches/monte_carlo.rs`'s `milestone_config_2d` cell layout,
/// with a real (if modest) burn-in/readout MCS budget rather than that
/// file's near-no-op `(0, 1, 1)` window — this benchmark cares about a
/// realistic per-simulation wall-clock cost, not per-attempt pricing.
fn base_config() -> ResolvedConfig<2> {
    UserConfig::<2> {
        grid: Some([30, 30]),
        boundary: Some(Boundary::Periodic),
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
        burn_in_mcs: Some(10),
        readout_mcs: Some(10),
        sampling_interval_mcs: Some(5),
        seed: Some(0), // overwritten per simulation by run_batch/run_sequential
        ..Default::default()
    }
    .resolve()
    .unwrap()
}

fn base_config_edge_list() -> ResolvedConfig<2> {
    let mut config = base_config();
    config.proposal = ProposalMode::EdgeList;
    config
}

fn homogeneous_thetas(config: &ResolvedConfig<2>, n: usize) -> Vec<Vec<f64>> {
    vec![theta::extract(config); n]
}

/// Spreads `theta`'s `lambda_V`/`lambda_I`/`J` entries by a per-item
/// multiplicative factor ranging 0.4x to 2.5x across the batch — soft to
/// stiff tissue, meaningfully different equilibrium morphologies and thus
/// (under `EdgeList`) meaningfully different edge-site counts. The
/// `lambda_Act`/`Max_Act` tail is 0 in `base_config`, and a multiplicative
/// factor leaves 0 exactly 0 (zero `lambda_Act` legally allows zero
/// `Max_Act`), so this never risks the Act-parameter validation regardless
/// of `factor`.
fn heterogeneous_thetas(config: &ResolvedConfig<2>, n: usize) -> Vec<Vec<f64>> {
    let base = theta::extract(config);
    (0..n)
        .map(|i| {
            let factor = if n <= 1 {
                1.0
            } else {
                0.4 + 2.1 * (i as f64 / (n - 1) as f64)
            };
            base.iter().map(|v| v * factor).collect()
        })
        .collect()
}

fn options(master_seed: u64) -> BatchOptions {
    BatchOptions {
        master_seed,
        include_lattice: false,
    }
}

/// The same work `run_batch` does, one simulation at a time, no Rayon —
/// `run_batch`'s own sequential baseline for computing achieved speedup.
fn run_sequential(base_config: &ResolvedConfig<2>, thetas: &[Vec<f64>], master_seed: u64) {
    for (index, theta_vec) in thetas.iter().enumerate() {
        let mut config = base_config.clone();
        theta::inject(&mut config, theta_vec).expect("theta must inject cleanly");
        config.seed = derive_seed_u64(master_seed, index as u64);
        let mut cpm = CPM::new(config).expect("injected config must be valid");
        initialize(&mut cpm).expect("fixture must initialise");
        std::hint::black_box(output::run_with(
            &mut cpm,
            RunOptions {
                include_lattice: false,
            },
        ));
    }
}

/// `run_batch` vs `run_sequential` at increasing `N`, homogeneous thetas
/// (every batch member gets the identical `theta`), `ProposalMode::Uniform`
/// — the scaling axis §44's single N=12 data point never swept.
fn batch_scaling_homogeneous(c: &mut Criterion) {
    let config = base_config();

    let mut group = c.benchmark_group("batch_scaling_homogeneous");
    for &n in &[12usize, 50, 200] {
        let thetas = homogeneous_thetas(&config, n);
        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(if n <= 50 { 20 } else { 10 });

        group.bench_with_input(BenchmarkId::new("parallel", n), &n, |b, _| {
            b.iter_batched(
                || (),
                |()| {
                    run_batch(&config, &thetas, options(1)).expect("batch must succeed");
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_with_input(BenchmarkId::new("sequential", n), &n, |b, _| {
            b.iter_batched(
                || (),
                |()| run_sequential(&config, &thetas, 1),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `run_batch` vs `run_sequential` at a fixed `N`, comparing homogeneous
/// against heterogeneous thetas under `ProposalMode::EdgeList` — the
/// proposal mode whose per-MCS attempt count actually varies with `theta`
/// (see the module doc comment). If Rayon's default work-stealing handles
/// this as well as the homogeneous case, the achieved speedup
/// (sequential/parallel, computed from this group's own reported medians)
/// should be close between the two; a real imbalance problem would show up
/// as materially worse speedup under `heterogeneous` than `homogeneous`.
fn batch_heterogeneity_edge_list(c: &mut Criterion) {
    const N: usize = 100;
    let config = base_config_edge_list();
    let homogeneous = homogeneous_thetas(&config, N);
    let heterogeneous = heterogeneous_thetas(&config, N);

    let mut group = c.benchmark_group("batch_heterogeneity_edge_list");
    group.throughput(Throughput::Elements(N as u64));
    group.sample_size(10);

    for (label, thetas) in [
        ("parallel_homogeneous", &homogeneous),
        ("parallel_heterogeneous", &heterogeneous),
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || (),
                |()| {
                    run_batch(&config, thetas, options(2)).expect("batch must succeed");
                },
                BatchSize::SmallInput,
            );
        });
    }

    for (label, thetas) in [
        ("sequential_homogeneous", &homogeneous),
        ("sequential_heterogeneous", &heterogeneous),
    ] {
        group.bench_function(label, |b| {
            b.iter_batched(
                || (),
                |()| run_sequential(&config, thetas, 2),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    batch_scaling_homogeneous,
    batch_heterogeneity_edge_list,
);
criterion_main!(benches);
