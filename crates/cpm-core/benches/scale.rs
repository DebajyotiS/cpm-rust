//! Profiling fixtures at inference-scale sizing regimes — confluent
//! (`phi ≈ 0.8`), not the sparse correctness-milestone configs
//! `benches/monte_carlo.rs` uses (10 cells in 900 or 27,000 sites, under 1%
//! occupied in 3D). Those milestone numbers answer "is the simulator
//! correct at a size we can also brute-force check"; these answer "what
//! does an inference-scale run actually cost."
//!
//! The 3D fixture starts from a worked inference example (40^3, 100 cells,
//! `V* ≈ 512`, `phi = 0.8`) but had to be adjusted — see
//! [`build_confluent_3d`]'s doc comment for why `phi = 0.8` isn't reachable
//! through `scatter_and_grow` at that cell size in 3D, and what density is
//! used instead. The 2D fixture targets "~200x200, ~200 cells" at
//! `phi = 0.8`, which `scatter_and_grow` seeds without trouble in 2D.
//!
//! **Methodology fix, not just new fixtures.** The existing
//! `attempt_*_milestone` benchmarks in `monte_carlo.rs` use `iter_batched`
//! with a fixed seed, so every criterion iteration re-measures the exact
//! same deterministic first RNG draw from a freshly re-initialised lattice —
//! a sample size of one, not an average over the population of attempts a
//! real run experiences. Here, each timed iteration instead runs a batch of
//! many attempts (`ATTEMPTS_PER_BATCH`) against an already fully-grown
//! fixture, so the measured cost is a genuine average over many different
//! sampled sites, not one repeated deterministic event.
//!
//! Growing the confluent fixtures (`initialize`'s scatter-and-grow, placing
//! and growing every cell to its full target volume) is real work — up to
//! ~50k accepted growth steps for the 3D fixture — so each fixture is built
//! **once** per benchmark function, outside any timed region, and every
//! criterion iteration gets a cheap clone of the already-grown `State`
//! (`State` derives `Clone`; rebuilding a fresh `CPM` from the same
//! `ResolvedConfig` is itself O(1)-ish — lattice/offset-table allocation
//! only — so `CPM::new(config.clone())` plus overwriting `.state` is far
//! cheaper than re-running scatter-and-grow every iteration).

use cpm_core::cell::CellType;
use cpm_core::config::{ProposalMode, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::{attempt, run_mcs};
use cpm_core::rng::rng_from_seed;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::Rng as _;

const ATTEMPTS_PER_BATCH: u64 = 1_000;

/// "~200x200, ~200 cells" 2D inference sizing, at `phi = 0.8` (not pinned
/// to an exact numeric target for 2D, but held consistent with the 3D
/// fixture below rather than left arbitrary). `target_interface` is scaled
/// from `monte_carlo.rs`'s milestone ratio (`I* = 20` at `V* = 9`) by
/// `sqrt(160 / 9)`, since interface scales with the square root of volume in
/// 2D — a plausible fixture shape, not a calibrated one (this bench cares
/// about throughput, not morphology).
fn build_confluent_2d_uninitialised(seed: u64) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([200, 200]),
        boundary: Some(Boundary::Periodic),
        cell_types: vec![
            CellType {
                name: "a".into(),
                target_volume: 160,
                target_interface: 84,
                lambda_volume: 1.0,
                lambda_interface: 0.2,
                lambda_act: 0.0,
                max_act: 0,
            },
            CellType {
                name: "b".into(),
                target_volume: 160,
                target_interface: 84,
                lambda_volume: 1.5,
                lambda_interface: 0.3,
                lambda_act: 0.0,
                max_act: 0,
            },
        ],
        cell_counts: vec![100, 100],
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
    CPM::new(config).unwrap()
}

fn build_confluent_2d(seed: u64) -> CPM<2> {
    let mut cpm = build_confluent_2d_uninitialised(seed);
    initialize(&mut cpm).expect("confluent 2D fixture must initialise");
    cpm
}

/// A worked inference-scale example (40^3, 100 cells, `V* ≈ 512`) targets
/// `phi = 0.8`, but `initialization::scatter_and_grow`'s seed-placement
/// heuristic derives its minimum pairwise separation as `sqrt(V* / pi)` — a
/// 2D circle-packing radius applied without a dimension correction — so it
/// rejects far more candidate seed points in 3D than the same `phi` would
/// need in 2D, and `phi = 0.8` at `V* = 512` on a 40^3 grid is infeasible
/// for it to seed at all (confirmed directly: it fails to place 100 seeds
/// at that `V*`, "too crowded for this configuration"). That's a real
/// limitation of the scatter-and-grow seeding heuristic in 3D, so this
/// fixture instead uses the densest configuration that heuristic can
/// actually seed at this grid size (`V* = 125`, smaller cells so more seeds
/// fit under the same separation formula), landing at `phi ≈ 0.29`: still a
/// large step up in density from the correctness-milestone fixture's ~0.5%
/// occupancy, and an honest number rather than a `phi` this initialisation
/// path can't reach. `target_interface` is scaled from `monte_carlo.rs`'s
/// 3D milestone ratio (`I* = 90` at `V* = 27`) by `(125 / 27)^(2/3)`.
fn build_confluent_3d_uninitialised(seed: u64) -> CPM<3> {
    let config = UserConfig::<3> {
        grid: Some([40, 40, 40]),
        boundary: Some(Boundary::Periodic),
        cell_types: vec![CellType {
            name: "a".into(),
            target_volume: 125,
            target_interface: 250,
            lambda_volume: 1.0,
            lambda_interface: 0.2,
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![150],
        adhesion: Some(vec![vec![0.0, 3.0], vec![3.0, 1.0]]),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    CPM::new(config).unwrap()
}

fn build_confluent_3d(seed: u64) -> CPM<3> {
    let mut cpm = build_confluent_3d_uninitialised(seed);
    initialize(&mut cpm).expect("confluent 3D fixture must initialise");
    cpm
}

/// Same fixture as [`build_confluent_2d`], with `proposal` set to
/// `EdgeList` *before* `initialize` runs, so the edge list actually gets
/// built (`initialization.rs` only builds it when `config.proposal ==
/// EdgeList` at the time cells are placed).
fn build_confluent_2d_edge_list(seed: u64) -> CPM<2> {
    let mut cpm = build_confluent_2d_uninitialised(seed);
    cpm.config.proposal = ProposalMode::EdgeList;
    initialize(&mut cpm).expect("confluent 2D edge-list fixture must initialise");
    cpm
}

fn build_confluent_3d_edge_list(seed: u64) -> CPM<3> {
    let mut cpm = build_confluent_3d_uninitialised(seed);
    cpm.config.proposal = ProposalMode::EdgeList;
    initialize(&mut cpm).expect("confluent 3D edge-list fixture must initialise");
    cpm
}

/// Samples `n` proposed (target, source) pairs directly, without mutating
/// `cpm`, and reports the same-cell early-out fraction — the number that
/// actually determines the edge list's expected win at this fixture's
/// density, rather than relying on a prior 3D estimate (~60%) measured on a
/// different configuration. Printed once, when the fixture is built, via
/// `eprintln!` (visible in `cargo bench`'s output; this is a diagnostic
/// observation, not a timed criterion measurement).
fn report_null_attempt_fraction<const D: usize>(label: &str, cpm: &CPM<D>, n: u64, seed: u64) {
    let mut rng = rng_from_seed(seed);
    let dims = cpm.state.lattice.dims();
    let mut null_count = 0u64;
    for _ in 0..n {
        let target_coord: [usize; D] = std::array::from_fn(|d| rng.gen_range(0..dims[d]));
        let target_flat = cpm.state.lattice.flat_index(target_coord);
        let source_offset = cpm.copy_offsets[rng.gen_range(0..cpm.copy_offsets.len())];
        let source_flat = cpm.state.lattice.neighbour(target_flat, source_offset);
        if cpm.state.lattice.get(target_flat) == cpm.state.lattice.get(source_flat) {
            null_count += 1;
        }
    }
    eprintln!(
        "[scale.rs] {label}: same-cell early-out fraction over {n} sampled attempts = {:.4}",
        null_count as f64 / n as f64
    );
}

fn attempt_batch_2d_confluent(c: &mut Criterion) {
    let base = build_confluent_2d(1);
    report_null_attempt_fraction("confluent_2d (200x200, phi=0.8)", &base, 1_000_000, 2);
    let config = base.config.clone();

    let mut group = c.benchmark_group("attempt_batch_2d_confluent");
    group.throughput(Throughput::Elements(ATTEMPTS_PER_BATCH));
    group.bench_function("attempt_x1000", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm
            },
            |mut cpm| {
                for _ in 0..ATTEMPTS_PER_BATCH {
                    attempt(&mut cpm, true);
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn attempt_batch_3d_confluent(c: &mut Criterion) {
    let base = build_confluent_3d(1);
    report_null_attempt_fraction("confluent_3d (40^3, phi≈0.29)", &base, 1_000_000, 2);
    let config = base.config.clone();

    let mut group = c.benchmark_group("attempt_batch_3d_confluent");
    group.throughput(Throughput::Elements(ATTEMPTS_PER_BATCH));
    group.bench_function("attempt_x1000", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm
            },
            |mut cpm| {
                for _ in 0..ATTEMPTS_PER_BATCH {
                    attempt(&mut cpm, true);
                }
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// MCS/sec at inference scale — the baseline later throughput work is
/// measured against.
fn mcs_2d_confluent(c: &mut Criterion) {
    let base = build_confluent_2d(1);
    let config = base.config.clone();

    let mut group = c.benchmark_group("mcs_2d_confluent");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm
            },
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn mcs_3d_confluent(c: &mut Criterion) {
    let base = build_confluent_3d(1);
    let config = base.config.clone();

    let mut group = c.benchmark_group("mcs_3d_confluent");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm
            },
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// Throughput comparison: `Uniform` vs `EdgeList` MCS/sec at the same
/// confluent-density fixture — the number that confirms (or corrects) a
/// prior "~2.5x" 3D estimate for this crate's actual target configurations.
fn mcs_2d_confluent_edge_list(c: &mut Criterion) {
    let base = build_confluent_2d_edge_list(1);
    let config = base.config.clone();

    let mut group = c.benchmark_group("mcs_2d_confluent_edge_list");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm.edge_list = base.edge_list.clone();
                cpm
            },
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn mcs_3d_confluent_edge_list(c: &mut Criterion) {
    let base = build_confluent_3d_edge_list(1);
    let config = base.config.clone();

    let mut group = c.benchmark_group("mcs_3d_confluent_edge_list");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run_mcs", |b| {
        b.iter_batched(
            || {
                let mut cpm = CPM::new(config.clone()).unwrap();
                cpm.state = base.state.clone();
                cpm.edge_list = base.edge_list.clone();
                cpm
            },
            |mut cpm| run_mcs(&mut cpm, 1),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    attempt_batch_2d_confluent,
    attempt_batch_3d_confluent,
    mcs_2d_confluent,
    mcs_3d_confluent,
    mcs_2d_confluent_edge_list,
    mcs_3d_confluent_edge_list,
);
criterion_main!(benches);
