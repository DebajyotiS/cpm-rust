//! Criterion benchmarks for attempts/sec and MCS/sec on the 2D and 3D
//! milestone configs (matching `checker.rs`'s own `build_config`/
//! `build_config_3d` fixtures, so these numbers describe the same systems
//! the correctness tests already validate), plus how `attempt()` throughput
//! scales with 2D lattice size. A fuller matrix — connectivity-table cache
//! behaviour vs. direct computation, per-energy-term cost, `u16` vs `u32`
//! ids.
//!
//! Every benchmark seeds and initialises fresh inside the `iter` closure's
//! setup (`Criterion::iter_batched`), never reusing one `CPM` across
//! samples: reusing state would let the chain equilibrate over the course
//! of the benchmark and measure a shifting, non-representative mix of
//! early-transient and steady-state attempts instead of one clear
//! throughput number.

use cpm_core::cell::CellType;
use cpm_core::config::UserConfig;
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::{attempt, run_mcs};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

/// Matches `checker.rs`'s `build_config` (2D, 30x30, 10 cells) — the
/// smallest system already exercised by the correctness suite, so these
/// throughput numbers are for a config known to behave correctly, not a
/// throughput-only fixture nobody has validated.
fn milestone_config_2d(seed: u64) -> CPM<2> {
    let config = UserConfig::<2> {
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
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("milestone config must initialise");
    cpm
}

/// Matches `checker.rs`'s `build_config_3d` (3D, 30^3, 5 cells) — the
/// 3D instantiation target.
fn milestone_config_3d(seed: u64) -> CPM<3> {
    let config = UserConfig::<3> {
        grid: Some([30, 30, 30]),
        boundary: Some(Boundary::Periodic),
        cell_types: vec![CellType {
            name: "a".into(),
            target_volume: 27,
            target_interface: 90,
            lambda_volume: 1.0,
            lambda_interface: 0.2,
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![5],
        adhesion: Some(vec![vec![0.0, 3.0], vec![3.0, 1.0]]),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("3D milestone config must initialise");
    cpm
}

/// A 2D config at a given grid size, cell count and per-cell target volume
/// held fixed (so confluency, not just raw site count, changes with grid
/// size) — used only by the lattice-size scaling group below.
fn scaling_config_2d(seed: u64, grid: usize) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([grid, grid]),
        boundary: Some(Boundary::Periodic),
        cell_types: vec![CellType {
            name: "a".into(),
            target_volume: 9,
            target_interface: 20,
            lambda_volume: 1.0,
            lambda_interface: 0.2,
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![10],
        adhesion: Some(vec![vec![0.0, 3.0], vec![3.0, 1.0]]),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("scaling config must initialise");
    cpm
}

fn attempt_throughput_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("attempt_2d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("attempt", |b| {
        b.iter_batched(
            || milestone_config_2d(1),
            |mut cpm| attempt(&mut cpm, true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn attempt_throughput_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("attempt_3d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("attempt", |b| {
        b.iter_batched(
            || milestone_config_3d(1),
            |mut cpm| attempt(&mut cpm, true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// MCS/sec, not just attempts/sec: `run_mcs`'s own definition of one MCS
/// (`N_sites` attempts, counting early-outs) is the unit every published
/// parameter set is quoted in, so this is the throughput number that maps
/// directly onto "how many MCS of simulated time can a training run afford."
fn mcs_throughput_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcs_2d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || milestone_config_2d(1),
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn mcs_throughput_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcs_3d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || milestone_config_3d(1),
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// How per-`attempt()` cost scales with 2D lattice size at fixed cell count
/// and target volume — separates "cost grows with the lattice" (would show
/// up here) from "cost grows with confluency or cell count" (out of scope
/// for this group; the milestone benchmarks above are held at one fixed,
/// already-validated system instead).
fn attempt_scaling_with_lattice_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("attempt_2d_lattice_size_scaling");
    for grid in [16usize, 32, 64, 128] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(grid), &grid, |b, &grid| {
            b.iter_batched(
                || scaling_config_2d(1, grid),
                |mut cpm| attempt(&mut cpm, true),
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// Isolates raw RNG generation cost against total `attempt()` cost, to
/// answer whether RNG generation dominates attempt cost at all — a
/// prerequisite to caring which RNG algorithm is faster. Every `attempt()`
/// call draws exactly `D + 1` `gen_range` calls unconditionally (one per
/// target coordinate, one for the source-neighbour index), plus one more
/// `gen::<f64>()` on the fraction of attempts where `delta_H > 0` — so this
/// reproduces the unconditional `D + 1` draws exactly and is a slight
/// underestimate of the true RNG share, not an overestimate.
fn rng_generation_cost(c: &mut Criterion) {
    use cpm_core::rng::rng_from_seed;
    use rand::Rng as _;

    let mut group = c.benchmark_group("rng_generation_cost");
    group.throughput(Throughput::Elements(1));
    group.bench_function("three_gen_range_calls_2d", |b| {
        b.iter_batched(
            || rng_from_seed(1),
            |mut rng| {
                let a = rng.gen_range(0..30usize);
                let b = rng.gen_range(0..30usize);
                let c = rng.gen_range(0..4usize);
                (a, b, c)
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("four_gen_range_calls_3d", |b| {
        b.iter_batched(
            || rng_from_seed(1),
            |mut rng| {
                let a = rng.gen_range(0..30usize);
                let b = rng.gen_range(0..30usize);
                let c = rng.gen_range(0..30usize);
                let d = rng.gen_range(0..6usize);
                (a, b, c, d)
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Same shape as [`milestone_config_2d`], with Act active. `attempt()` with
/// `Terms.act: None` (Act off) showed no measurable regression against the
/// pre-Act baseline (noise-level differences per Criterion's own
/// significance test), so this benchmark measures Act's marginal
/// per-attempt cost when it is active, rather than re-measuring the off
/// case.
fn milestone_config_2d_with_act(seed: u64) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([30, 30]),
        boundary: Some(Boundary::Periodic),
        cell_types: vec![
            CellType {
                name: "a".into(),
                target_volume: 9,
                target_interface: 20,
                lambda_volume: 1.0,
                lambda_interface: 0.2,
                lambda_act: 2.0,
                max_act: 10,
            },
            CellType {
                name: "b".into(),
                target_volume: 9,
                target_interface: 20,
                lambda_volume: 1.5,
                lambda_interface: 0.3,
                lambda_act: 3.0,
                max_act: 15,
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
    let mut config = config;
    config.active_terms.act = true;
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("milestone config must initialise");
    cpm
}

fn attempt_throughput_2d_with_act(c: &mut Criterion) {
    let mut group = c.benchmark_group("attempt_2d_milestone_with_act");
    group.throughput(Throughput::Elements(1));
    group.bench_function("attempt", |b| {
        b.iter_batched(
            || milestone_config_2d_with_act(1),
            |mut cpm| attempt(&mut cpm, true),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    attempt_throughput_2d,
    attempt_throughput_3d,
    mcs_throughput_2d,
    mcs_throughput_3d,
    attempt_scaling_with_lattice_size,
    rng_generation_cost,
    attempt_throughput_2d_with_act,
);
criterion_main!(benches);
