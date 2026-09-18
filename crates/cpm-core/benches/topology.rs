//! Compares simple-point table lookups against direct computation in 3D.
//! The 8 MiB 3D table exceeds L2 cache capacity and generates effectively
//! random lookup patterns, so its performance advantage over direct computation
//! must be measured directly rather than assumed. Both paths process the
//! **exact same patterns**, harvested from real attempts against a confluent-density
//! 3D fixture rather than synthetic random 26-bit patterns. This ensures cache
//! behavior reflects what `preserves_topology` encounters during an actual run
//! (and not an artificial worst or best case).

use cpm_core::cell::CellType;
use cpm_core::config::UserConfig;
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::run_mcs;
use cpm_core::rng::rng_from_seed;
use cpm_core::simple_point::{direct_checker_3d, simple_point_table_3d};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::Rng as _;

/// Same fixture as `benches/scale.rs`'s `build_confluent_3d` (40^3, 150
/// cells, `V* = 125`, `phi ≈ 0.29` — see that file's doc comment for why
/// `phi = 0.8` isn't reachable via `scatter_and_grow` at this cell size).
fn build_confluent_3d(seed: u64) -> CPM<3> {
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

/// Harvests `n` real 26-bit simple-point patterns: relaxes the fixture for a
/// few MCS first (a freshly-grown lattice is atypically smooth), then
/// repeatedly draws (target, source) pairs the same way `attempt()` does,
/// keeping only the ones that would actually reach the connectivity check
/// (different owners, losing cell non-medium) and recording the pattern
/// `preserves_topology` would build for it — without mutating the lattice,
/// so the same relaxed state is sampled throughout.
fn harvest_patterns(cpm: &CPM<3>, n: usize, seed: u64) -> Vec<u32> {
    let mut rng = rng_from_seed(seed);
    let dims = cpm.state.lattice.dims();
    let mut patterns = Vec::with_capacity(n);
    while patterns.len() < n {
        let target_coord: [usize; 3] = std::array::from_fn(|d| rng.gen_range(0..dims[d]));
        let target_flat = cpm.state.lattice.flat_index(target_coord);
        let source_offset = cpm.copy_offsets[rng.gen_range(0..cpm.copy_offsets.len())];
        let source_flat = cpm.state.lattice.neighbour(target_flat, source_offset);
        let losing_id = cpm.state.lattice.get(target_flat);
        let gaining_id = cpm.state.lattice.get(source_flat);
        if losing_id == gaining_id || losing_id == 0 {
            continue;
        }
        let mut pattern: u32 = 0;
        for (i, &delta) in cpm.topology_offsets.iter().enumerate() {
            let neighbour_flat = cpm.state.lattice.neighbour(target_flat, delta);
            if cpm.state.lattice.get(neighbour_flat) == losing_id {
                pattern |= 1 << i;
            }
        }
        patterns.push(pattern);
    }
    patterns
}

fn table_vs_direct_3d(c: &mut Criterion) {
    let mut cpm = build_confluent_3d(1);
    run_mcs(&mut cpm, 5); // relax past the freshly-grown, atypically smooth state
    let patterns = harvest_patterns(&cpm, 200_000, 2);
    eprintln!(
        "[topology.rs] harvested {} real simple-point patterns from a relaxed confluent 3D fixture",
        patterns.len()
    );

    let table = simple_point_table_3d();
    let direct = direct_checker_3d();

    // Cross-check once, outside the timed region: both paths must agree on
    // every harvested pattern, or a throughput comparison between them is
    // meaningless.
    for &p in &patterns {
        assert_eq!(
            table.is_simple(p),
            direct.is_simple(p),
            "table/direct disagree on pattern {p:#x}"
        );
    }

    let mut group = c.benchmark_group("simple_point_3d_table_vs_direct");
    group.throughput(Throughput::Elements(patterns.len() as u64));
    group.bench_function("table_lookup", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for &p in &patterns {
                if table.is_simple(p) {
                    count += 1;
                }
            }
            count
        });
    });
    group.bench_function("direct_computation", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for &p in &patterns {
                if direct.is_simple(p) {
                    count += 1;
                }
            }
            count
        });
    });
    group.finish();
}

criterion_group!(benches, table_vs_direct_3d);
criterion_main!(benches);
