//! Criterion benchmarks for readout cost, separate from `monte_carlo.rs`'s
//! dynamics benchmarks. `labelling::label_components` is a whole-lattice
//! flood fill kept out of the hot loop and run only at
//! `sampling_interval_mcs` cadence (`output::run`) — this file measures
//! that cadence's cost, especially in 3D, which is the inference target
//! (roughly 64^3 sites and 200 cells). `label_components_3d_confluent`
//! below measures the actual affordability conclusion (~7.5% of one MCS at
//! this project's confluent-density regime) rather than asserting one.
//!
//! Every benchmark seeds and initialises fresh inside the `iter` closure's
//! setup, matching `monte_carlo.rs`'s own rationale: reusing one `CPM`
//! across samples would let readout cost drift as the population's shape
//! and contact structure evolve, rather than measuring one representative
//! system.

use cpm_core::cell::CellType;
use cpm_core::config::UserConfig;
use cpm_core::initialization::initialize;
use cpm_core::labelling::label_components;
use cpm_core::lattice::{offset_table, Boundary};
use cpm_core::model::CPM;
use cpm_core::neighborhood::Stencil;
use cpm_core::output::{run, shape_descriptors};
use cpm_core::state::index_of;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};

/// Matches `checker.rs`'s `build_config` / `benches/monte_carlo.rs`'s
/// `milestone_config_2d` (2D, 30x30, 10 cells) — a system already validated
/// for correctness, not a throughput-only fixture.
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

/// Matches `benches/monte_carlo.rs`'s `milestone_config_3d` (3D, 30^3, 5
/// cells) — the 3D correctness instantiation target.
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

/// A 3D config at a given grid size, cell count and per-cell target volume
/// held fixed (so confluency, not just raw site count, changes with grid
/// size) — mirrors `monte_carlo.rs`'s 2D lattice-size scaling group, but for
/// `label_components`, and in 3D specifically since 64^3 is the 3D
/// inference target this readout's cost is measured against.
fn scaling_config_3d(seed: u64, grid: usize) -> CPM<3> {
    let config = UserConfig::<3> {
        grid: Some([grid, grid, grid]),
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
        cell_counts: vec![20],
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

fn label_components_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_components_2d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("label_components", |b| {
        b.iter_batched(
            || milestone_config_2d(1),
            |cpm| {
                let medium_offsets =
                    offset_table(&Stencil::<2>::full(), cpm.state.lattice.strides());
                label_components(
                    &cpm.state.lattice,
                    &cpm.connectivity_offsets,
                    &medium_offsets,
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn label_components_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_components_3d_milestone");
    group.throughput(Throughput::Elements(1));
    group.bench_function("label_components", |b| {
        b.iter_batched(
            || milestone_config_3d(1),
            |cpm| {
                let medium_offsets =
                    offset_table(&Stencil::<3>::full(), cpm.state.lattice.strides());
                label_components(
                    &cpm.state.lattice,
                    &cpm.connectivity_offsets,
                    &medium_offsets,
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// How `label_components` cost scales with 3D lattice size at fixed cell
/// count/target volume — the direct answer to "is the sampling-cadence
/// flood fill affordable at the 3D inference target size," not just at the
/// smaller correctness-instantiation size the milestone benchmark above
/// uses.
fn label_components_scaling_with_lattice_size_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_components_3d_lattice_size_scaling");
    for grid in [16usize, 30, 48, 64] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(grid), &grid, |b, &grid| {
            b.iter_batched(
                || scaling_config_3d(1, grid),
                |cpm| {
                    let medium_offsets =
                        offset_table(&Stencil::<3>::full(), cpm.state.lattice.strides());
                    label_components(
                        &cpm.state.lattice,
                        &cpm.connectivity_offsets,
                        &medium_offsets,
                    )
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// `shape_descriptors`' own cost, isolated from the flood fill above: a
/// closed-form 2x2/3x3 eigensolver per cell, expected to be negligible next
/// to `label_components`, but measured rather than assumed (same principle
/// as the rest of this file).
fn shape_descriptors_cost(c: &mut Criterion) {
    let cpm = milestone_config_3d(1);
    let i = index_of(cpm.state.cells[0].id);
    let volume = cpm.state.volume[i];
    let centroid: [f64; 3] =
        std::array::from_fn(|d| cpm.state.sum_position[i][d] as f64 / volume as f64);
    let sum_position_outer = cpm.state.sum_position_outer[i];

    let mut group = c.benchmark_group("shape_descriptors_3d");
    group.throughput(Throughput::Elements(1));
    group.bench_function("shape_descriptors", |b| {
        b.iter(|| shape_descriptors(&sum_position_outer, volume, &centroid));
    });
    group.finish();
}

/// The confluent-density 3D fixture (40^3, `V* = 125`, 150 cells, phi≈0.29)
/// `benches/scale.rs`'s `build_confluent_3d` already established as this
/// project's working inference-representative regime — duplicated here
/// rather than imported, since bench files are separate binaries and can't
/// share private helpers, the same reason
/// `monte_carlo::tests::diag_total_delta_vs_commit_cost_3d` duplicates it.
/// `scaling_config_3d`'s 20-cells-at-every-grid-size sweep above answers
/// "how does the flood fill scale with lattice size," but never actually
/// reaches this crate's own established confluent density — this fixture
/// does, at the specific size/cell-count/density `scale.rs`'s throughput
/// numbers are already measured against, so the flood-fill and full-window
/// costs below are directly comparable to those MCS/sec figures.
fn confluent_config_3d(seed: u64) -> CPM<3> {
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
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("confluent 3D fixture must initialise");
    cpm
}

/// `label_components` cost at the confluent fixture above, rather than at
/// the sparse (~0.5% occupancy) milestone fixture `label_components_3d`
/// uses or the fixed-20-cells sweep above — the actual density this
/// project's inference-scale throughput is measured at.
fn label_components_3d_confluent(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_components_3d_confluent");
    group.throughput(Throughput::Elements(1));
    group.bench_function("label_components", |b| {
        b.iter_batched(
            || confluent_config_3d(1),
            |cpm| {
                let medium_offsets =
                    offset_table(&Stencil::<3>::full(), cpm.state.lattice.strides());
                label_components(
                    &cpm.state.lattice,
                    &cpm.connectivity_offsets,
                    &medium_offsets,
                )
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

/// The cost a training run actually pays per readout sample: `output::run`
/// with a short window (one sampling-interval chunk of burn-in, one of
/// readout), covering the full per-sample pipeline — per-cell
/// volume/interface/centroid/shape plus one `label_components` pass — not
/// just its pieces in isolation.
fn run_one_readout_window_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("run_3d_milestone_one_window");
    group.throughput(Throughput::Elements(1));
    group.bench_function("run", |b| {
        b.iter_batched(
            || {
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
                    burn_in_mcs: Some(1),
                    readout_mcs: Some(1),
                    sampling_interval_mcs: Some(1),
                    seed: Some(1),
                    ..Default::default()
                }
                .resolve()
                .unwrap();
                let mut cpm = CPM::new(config).unwrap();
                initialize(&mut cpm).expect("3D milestone config must initialise");
                cpm
            },
            |mut cpm| run(&mut cpm),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    label_components_2d,
    label_components_3d,
    label_components_scaling_with_lattice_size_3d,
    label_components_3d_confluent,
    shape_descriptors_cost,
    run_one_readout_window_3d,
);
criterion_main!(benches);
