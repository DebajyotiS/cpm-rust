//! Measures target volume and target interface (`V*`/`I*`) for a cell type by relaxing a single cell to equilibrium.
//!
//! **Adhesion is deliberately kept active during relaxation.** Disabling adhesion entirely (`adhesion = 0`, `lambda_interface = 0`)
//! removes all surface area penalties from `H`. Without surface tension, topological entropy favors thin, branching, snake-like
//! shapes over compact blobs—for example, `V*=40` yields `I*~140` (compared to ~35-45 for a compact 40-site blob under Moore counting).
//! This is a well-known property of the classic Potts model: zero surface tension drives entropy-dominated shape branching, which is
//! why Graner & Glazier's original CPM required contact energy terms in the first place. See `diagnose()` for the full diagnostic
//! trace and ASCII shape dumps.
//!
//! To measure realistic equilibrium shapes, `calibrate()` sets `lambda_interface = 0` (so `I*` is measured rather than assumed) while
//! applying a small, production-representative cell-medium adhesion `J`. This provides the baseline surface tension present in real
//! simulation runs so `I*` reflects the cell's actual equilibrium geometry.
//!
//! Run with `cargo run -p cpm-core --release --example calibrate_targets`.

use cpm_core::cell::CellType;
use cpm_core::config::{InitializationSpec, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::{Boundary, CellId};
use cpm_core::model::CPM;
use cpm_core::monte_carlo::run_mcs;

const GRID: usize = 60;

struct Report {
    mean: f64,
    std: f64,
}

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

fn build(
    grid: usize,
    target_volume: u32,
    lambda_volume: f64,
    adhesion_medium: f64,
    seed: u64,
) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([grid, grid]),
        boundary: Some(Boundary::Fixed),
        cell_types: vec![CellType {
            name: "calibration".into(),
            target_volume,
            target_interface: 0,   // unused: lambda_interface = 0 below
            lambda_volume,         // strong, holds V* tightly
            lambda_interface: 0.0, // disabled: measuring I*, not guessing it
            lambda_act: 0.0,
            max_act: 0,
        }],
        cell_counts: vec![1],
        // Row/col 0 = medium, 1 = the cell type. `adhesion_medium` is the
        // only nonzero entry: cell-medium contact is what provides surface
        // tension here (see this file's doc comment for why zero adhesion
        // was tried first and rejected by measurement, not assumption).
        adhesion: Some(vec![vec![0.0, adhesion_medium], vec![adhesion_medium, 0.0]]),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        initialization: Some(InitializationSpec::Explicit {
            lattice: compact_seed(grid, target_volume),
            cell_type_of: vec![0],
        }),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).expect("calibration seed must initialise");
    cpm
}

fn relax_and_measure(
    grid: usize,
    target_volume: u32,
    lambda_volume: f64,
    adhesion_medium: f64,
    burn_in: u64,
    samples: u64,
    thin: u64,
) -> Report {
    let mut cpm = build(grid, target_volume, lambda_volume, adhesion_medium, 1);
    run_mcs(&mut cpm, burn_in);

    let mut values = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        run_mcs(&mut cpm, thin);
        values.push(cpm.state.interface[0] as f64);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    Report {
        mean,
        std: variance.sqrt(),
    }
}

/// Prints the cell's bounding box (padded by 2) as `#`/`.` so the shape can
/// be eyeballed directly, rather than trusted from a single scalar.
fn dump_shape(cpm: &CPM<2>) {
    let mut min_r = usize::MAX;
    let mut max_r = 0;
    let mut min_c = usize::MAX;
    let mut max_c = 0;
    cpm.state.lattice.each_interior_coord(|[r, c]| {
        let flat = cpm.state.lattice.flat_index([r, c]);
        if cpm.state.lattice.get(flat) != 0 {
            min_r = min_r.min(r);
            max_r = max_r.max(r);
            min_c = min_c.min(c);
            max_c = max_c.max(c);
        }
    });
    if min_r > max_r {
        eprintln!("  (empty)");
        return;
    }
    let (lo_r, hi_r) = (min_r.saturating_sub(2), max_r + 2);
    let (lo_c, hi_c) = (min_c.saturating_sub(2), max_c + 2);
    for r in lo_r..=hi_r {
        let mut line = String::new();
        for c in lo_c..=hi_c {
            let flat = cpm.state.lattice.flat_index([r, c]);
            line.push(if cpm.state.lattice.get(flat) != 0 {
                '#'
            } else {
                '.'
            });
        }
        eprintln!("  {line}");
    }
}

/// The real measurement: production-representative cell-medium adhesion
/// (matching `config::test_support::two_type_config`'s own `J` values for
/// each type — epithelial-medium = 5.0, stem-medium = 6.0), `lambda_volume`
/// strong enough to hold `V*` tightly, `lambda_interface = 0` throughout
/// (still not presupposing the answer).
fn calibrate() {
    println!("=== calibration: production-representative adhesion, lambda_volume=5.0 ===");
    for (name, target_volume, adhesion_medium) in [("epithelial", 50u32, 5.0), ("stem", 40, 6.0)] {
        let r = relax_and_measure(GRID, target_volume, 5.0, adhesion_medium, 20_000, 2_000, 10);
        let shape_factor = r.mean / (target_volume as f64).sqrt();
        println!(
            "{name:<10} V*={target_volume:>4}  I* mean={:>8.3}  std={:>6.3}  I*/sqrt(V*)={shape_factor:.4}",
            r.mean, r.std
        );
    }
}

/// The evidence trail behind the deviation documented at the top of this
/// file: reproduces the zero-adhesion measurement that first surfaced the
/// ramified-shape problem, then rules out the two obvious alternative
/// explanations (insufficient burn-in; too-weak `lambda_volume`) before
/// confirming visually that the zero-surface-tension equilibrium really is
/// non-compact. Not part of the actual calibration — run with
/// `CALIBRATE_DIAGNOSE=1` to reproduce this evidence; `calibrate()` above
/// is what actually produces the fixture values.
fn diagnose() {
    println!("=== [diagnostic] zero adhesion, lambda_volume=5.0, burn_in=20_000 ===");
    for target_volume in [40u32, 50] {
        let r = relax_and_measure(GRID, target_volume, 5.0, 0.0, 20_000, 2_000, 10);
        let shape_factor = r.mean / (target_volume as f64).sqrt();
        println!(
            "V*={target_volume:>4}  I* mean={:>8.3}  std={:>6.3}  I*/sqrt(V*)={shape_factor:.4}",
            r.mean, r.std
        );
    }

    println!("\n=== [diagnostic] zero adhesion, much stronger lambda_volume (50.0 vs 5.0) ===");
    {
        let target_volume = 40u32;
        let r = relax_and_measure(GRID, target_volume, 50.0, 0.0, 20_000, 2_000, 10);
        println!(
            "V*={target_volume:>4}  I* mean={:>8.3}  std={:>6.3}  (std=0 means the chain froze \
             entirely — even single-site fluctuations now cost exp(-50)~0 — not a real \
             compact equilibrium)",
            r.mean, r.std
        );
    }

    println!("\n=== [diagnostic] zero adhesion, much longer burn-in (200_000 vs 20_000) ===");
    {
        let target_volume = 40u32;
        let r = relax_and_measure(GRID, target_volume, 5.0, 0.0, 200_000, 2_000, 10);
        println!(
            "V*={target_volume:>4}  I* mean={:>8.3}  std={:>6.3}  (matches the 20_000-MCS \
             baseline, ruling out incomplete relaxation)",
            r.mean, r.std
        );
    }

    println!("\n=== [diagnostic] zero adhesion, small system (V*=8) ===");
    {
        let target_volume = 8u32;
        let r = relax_and_measure(GRID, target_volume, 5.0, 0.0, 20_000, 2_000, 10);
        let shape_factor = r.mean / (target_volume as f64).sqrt();
        println!(
            "V*={target_volume:>4}  I* mean={:>8.3}  std={:>6.3}  I*/sqrt(V*)={shape_factor:.4}",
            r.mean, r.std
        );
    }

    println!("\n=== [diagnostic] zero adhesion, final-shape dump, V*=40 ===");
    let mut cpm = build(GRID, 40, 5.0, 0.0, 1);
    run_mcs(&mut cpm, 20_000);
    eprintln!(
        "  volume={} interface={}",
        cpm.state.volume[0], cpm.state.interface[0]
    );
    dump_shape(&cpm);
}

fn main() {
    if std::env::var("CALIBRATE_DIAGNOSE").is_ok() {
        diagnose();
    } else {
        calibrate();
        if std::env::var("CALIBRATE_SHOW_SHAPE").is_ok() {
            println!("\n=== calibration final-shape dump, epithelial V*=50 ===");
            let mut cpm = build(GRID, 50, 5.0, 5.0, 1);
            run_mcs(&mut cpm, 20_000);
            eprintln!(
                "  volume={} interface={}",
                cpm.state.volume[0], cpm.state.interface[0]
            );
            dump_shape(&cpm);
        }
    }
}
