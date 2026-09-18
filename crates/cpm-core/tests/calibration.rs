//! Guards the calibrated `target_interface` fixture in
//! `config::test_support::two_type_config`: reruns the same single-cell
//! relaxation as `crates/cpm-core/examples/calibrate_targets.rs` used to
//! produce those numbers, and asserts the result still lands within a few
//! standard deviations of what was actually measured. A later change to
//! the energy neighbourhood, its weighting, the boundary convention, or
//! the interface definition would otherwise silently invalidate the
//! stored `target_interface` values without this catching it.
//!
//! `#[ignore]`d like the other slow release-mode tests in this crate
//! (tens of millions of attempts per cell type); run with
//! `cargo test --release -- --ignored`.

use cpm_core::cell::CellType;
use cpm_core::config::{InitializationSpec, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::{Boundary, CellId};
use cpm_core::model::CPM;
use cpm_core::monte_carlo::run_mcs;

const GRID: usize = 60;
const BURN_IN: u64 = 20_000;
const SAMPLES: u64 = 2_000;
const THIN: u64 = 10;

fn compact_seed(grid: usize, target_volume: u32) -> Vec<CellId> {
    let side = (target_volume as f64).sqrt().ceil() as usize + 1;
    let origin = grid / 2 - side / 2;
    let mut lattice = vec![0 as CellId; grid * grid];
    let mut placed = 0u32;
    'fill: for r in origin..origin + side {
        for c in origin..origin + side {
            if placed == target_volume {
                break 'fill;
            }
            lattice[r * grid + c] = 1;
            placed += 1;
        }
    }
    assert_eq!(
        placed, target_volume,
        "starting blob must reach target_volume"
    );
    lattice
}

/// Mirrors `calibrate_targets.rs::relax_and_measure` exactly (independent
/// copy, not `#[path]`-including the example: this crate has no non-test
/// mechanism to share code with `examples/`, and duplicating ~30 lines
/// here is cheaper than adding one) — same conventions, same
/// `lambda_interface = 0`, same production-representative cell-medium
/// adhesion.
fn relax_and_measure(target_volume: u32, adhesion_medium: f64) -> (f64, f64) {
    let config = UserConfig::<2> {
        grid: Some([GRID, GRID]),
        boundary: Some(Boundary::Fixed),
        cell_types: vec![CellType {
            name: "calibration".into(),
            target_volume,
            target_interface: 0,
            lambda_volume: 5.0,
            lambda_interface: 0.0,
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![1],
        adhesion: Some(vec![vec![0.0, adhesion_medium], vec![adhesion_medium, 0.0]]),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(1),
        initialization: Some(InitializationSpec::Explicit {
            lattice: compact_seed(GRID, target_volume),
            cell_type_of: vec![0],
        }),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("calibration seed must initialise");
    run_mcs(&mut cpm, BURN_IN);

    let mut values = Vec::with_capacity(SAMPLES as usize);
    for _ in 0..SAMPLES {
        run_mcs(&mut cpm, THIN);
        values.push(cpm.state.interface[0] as f64);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, variance.sqrt())
}

/// Tolerance is a small multiple of the standard deviation *measured at
/// calibration time* (0.973 for epithelial, 0.430 for stem — see
/// `config.rs`'s fixture doc comment), not of whatever this run's own
/// `std` comes out to: using this run's own spread would let the band
/// silently widen if a convention change made the relaxation noisier,
/// which is exactly the kind of drift this test exists to catch.
fn assert_matches_calibrated_fixture(
    name: &str,
    target_volume: u32,
    adhesion_medium: f64,
    expected_mean: f64,
    calibration_std: f64,
) {
    let (mean, std) = relax_and_measure(target_volume, adhesion_medium);
    let tolerance = 5.0 * calibration_std;
    eprintln!("{name}: measured I* mean={mean:.3} std={std:.3} (expected {expected_mean:.3} +/- {tolerance:.3})");
    assert!(
        (mean - expected_mean).abs() < tolerance,
        "{name}: measured I* mean {mean:.3} is more than {tolerance:.3} away from the \
         calibrated fixture value {expected_mean:.3} — a convention this measurement depends \
         on (energy neighbourhood/weighting, boundary, interface definition) may have changed \
         since `config::test_support::two_type_config`'s `target_interface` was calibrated; \
         rerun `examples/calibrate_targets.rs` and update the fixture deliberately, don't just \
         widen this tolerance"
    );
}

#[test]
#[ignore = "slow: two single-cell relaxations, run with --release -- --ignored"]
fn epithelial_target_interface_matches_calibrated_fixture() {
    assert_matches_calibrated_fixture("epithelial", 50, 5.0, 74.753, 0.973);
}

#[test]
#[ignore = "slow: two single-cell relaxations, run with --release -- --ignored"]
fn stem_target_interface_matches_calibrated_fixture() {
    assert_matches_calibrated_fixture("stem", 40, 6.0, 66.097, 0.430);
}
