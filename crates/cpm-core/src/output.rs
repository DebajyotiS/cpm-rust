//! Run output and metadata: the run-metadata bundle, [`SimulationStatus`],
//! gyration-tensor shape descriptors, and the windowed [`run`] that ties
//! burn-in, readout sampling, and per-sample readouts together.
//! `monte_carlo::run_mcs` is the low-level primitive this wraps, exactly
//! as `run_mcs`'s own doc comment anticipated.
//!
//! Every SBI-facing summary here is population-level or distributional —
//! never indexed by a specific cell ID — so the summary vector's shape
//! stays fixed regardless of population size. This is a design constraint
//! chosen ahead of cell division/proliferation, which is not yet
//! supported: the current design fixes cell count and requires stable IDs
//! for the run, both of which division would break, so introducing it
//! later would otherwise force a redesign of this layer. Per-cell detail
//! (`Sample::cells`) stays available for the separate debug/trajectory
//! output mode, never folded into a population-level SBI summary by this
//! module itself.
//!
//! Cell-cell/type-type "contact" thresholds, sorting/mixing/clustering
//! indices, and graph embeddings live outside this module.
//! `State::contact` (the weighted contact graph) is the one unambiguous,
//! convention-hash-worthy object this crate exports; every downstream
//! reduction of it is contested in the literature and free to evolve, so
//! it belongs in Python, outside the hash, not frozen here.

use crate::cell::Cell;
use crate::config::{convention_hash, model_hash, SeedPolicy};
use crate::labelling::{label_components, MediumComponent};
use crate::lattice::CellId;
use crate::model::CPM;
use crate::monte_carlo::run_mcs;
use crate::neighborhood::Stencil;
use crate::state::{index_of, ContactGraph};
use crate::theta::THETA_LAYOUT_VERSION;
use std::collections::HashMap;

/// The window fraction for the [`SimulationStatus::NotEquilibrated`]
/// check: the final quarter of `burn_in_mcs`.
const NOT_EQUILIBRATED_WINDOW_FRACTION: f64 = 0.25;
/// Flag [`SimulationStatus::NotEquilibrated`] when the fitted energy
/// trend's total drift over the window exceeds this multiple of the fit's
/// residual standard deviation — "is the drift bigger than the noise."
/// Both constants are frozen conventions in [`crate::config::CONVENTION_VERSION`]'s
/// scope, not tunable per run.
const NOT_EQUILIBRATED_SLOPE_THRESHOLD: f64 = 2.0;
/// A [`SimulationStatus::CellLost`] cell must sit at `min_cell_volume` at
/// every sample across at least this fraction of the readout window
/// (counted from the end) before it's flagged — a single sample at the
/// floor during ordinary jiggling is expected and not pathological.
const CELL_LOST_WINDOW_FRACTION: f64 = 0.5;

/// The run-metadata bundle: every output carries at minimum these
/// fields, assembled once per run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunMetadata {
    pub cpm_version: String,
    pub convention_version: u32,
    pub convention_hash: String,
    pub model_hash: String,
    pub seed: u64,
    pub seed_policy: SeedPolicy,
    pub derived_phi: f64,
    pub theta_layout_version: u32,
}

impl RunMetadata {
    fn assemble<const D: usize>(cpm: &CPM<D>) -> Self {
        Self {
            cpm_version: env!("CARGO_PKG_VERSION").to_string(),
            convention_version: crate::config::CONVENTION_VERSION,
            convention_hash: convention_hash(&cpm.config),
            model_hash: model_hash(&cpm.config, &cpm.state),
            seed: cpm.config.seed,
            seed_policy: cpm.config.seed_policy,
            derived_phi: derived_phi(cpm),
            theta_layout_version: THETA_LAYOUT_VERSION,
        }
    }
}

/// `phi = sum over c >= 1 of V_c / N_sites`, computed directly from the
/// current state — never a settable field. The user sets packing density
/// indirectly, through grid size and cell volumes; `phi` is derived and
/// reported.
pub fn derived_phi<const D: usize>(cpm: &CPM<D>) -> f64 {
    let occupied: u32 = cpm.state.volume.iter().sum();
    occupied as f64 / cpm.n_sites() as f64
}

/// Non-`Ok` statuses record enough to reproduce the finding (which cell,
/// which MCS) rather than just a bare variant — matching `checker::check`'s
/// own "carry enough state to reproduce the mismatch" convention. Detect
/// and report, never repair (a fragmented or lost cell is left exactly as
/// found; deciding how to repair it, if ever, is a modelling decision that
/// doesn't belong here).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationStatus {
    Ok,
    /// A cell held at `min_cell_volume` across at least
    /// `CELL_LOST_WINDOW_FRACTION` of the readout window's samples,
    /// counted from the end — not a single transient floor touch.
    CellLost {
        id: CellId,
        at_mcs: u64,
    },
    /// A cell ID whose sites span more than one connected component under
    /// `connectivity_neighborhood` — non-local fragmentation the per-move
    /// simple-point check cannot catch (`labelling.rs`). Records the MCS
    /// of the *first* sample at which it was observed.
    Fragmented {
        id: CellId,
        at_mcs: u64,
    },
    /// The conservative energy was still trending, not just fluctuating
    /// noisily, at the end of burn-in — see `not_equilibrated` below for
    /// the exact criterion.
    NotEquilibrated,
    /// The cell-adjacency graph (contact-graph pairs with
    /// `shared_interface > 0`, medium excluded) formed a single connected
    /// component spanning every cell at the final sample — the simplest
    /// defensible version of "collapsed into one contact cluster," stated
    /// explicitly as one to revisit empirically once real runs are
    /// available (mirroring how `NOT_EQUILIBRATED_SLOPE_THRESHOLD` is
    /// presented).
    Degenerate,
}

/// Per-cell gyration-tensor-derived shape descriptors: sorted eigenvalues,
/// their sum (`R_g^2`, the squared radius of gyration), and `kappa^2`
/// (relative shape anisotropy) — dimension-generic, unbiased, and
/// reported instead of a hard-coded "elongation"/"flatness"/"asphericity"
/// split, since those terms are inherently 3D-specific in the literature
/// (no clean 2D analogue) and no particular downstream ratio is mandated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeDescriptors<const D: usize> {
    /// Descending: `eigenvalues[0] >= eigenvalues[1] >= ...`.
    pub eigenvalues: [f64; D],
    pub radius_of_gyration_sq: f64,
    pub kappa_sq: f64,
}

/// `G_ab = (1/V_c) * sum_i (r_i,a - x_c,a)(r_i,b - x_c,b)`, computed from
/// the raw second-moment accumulator via the standard sum-of-squares-
/// minus-square-of-mean decomposition (`State::sum_position_outer`'s own
/// doc comment), then eigen-decomposed with a hand-rolled closed-form
/// solver (`D` is always 2 or 3, so a closed form is exact and avoids a
/// new linear-algebra dependency).
pub fn shape_descriptors<const D: usize>(
    sum_position_outer: &[[i64; D]; D],
    volume: u32,
    centroid: &[f64; D],
) -> ShapeDescriptors<D> {
    let v = volume as f64;
    let mut g = [[0.0f64; D]; D];
    for a in 0..D {
        for b in 0..D {
            g[a][b] = sum_position_outer[a][b] as f64 / v - centroid[a] * centroid[b];
        }
    }
    let mut eigenvalues = symmetric_eigenvalues(&g);
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let sum: f64 = eigenvalues.iter().sum();
    let sum_sq: f64 = eigenvalues.iter().map(|l| l * l).sum();
    // kappa^2 = (D/(D-1)) * sum(l_i^2)/sum(l_i)^2 - 1/(D-1), the
    // dimension-generic form of relative shape anisotropy (0 for a perfect
    // sphere/disk, 1 for a line) — reduces to the standard 3D formula
    // `(3/2)*sum(l^2)/sum(l)^2 - 1/2` at D=3.
    let kappa_sq = if sum > 0.0 {
        let d = D as f64;
        (d / (d - 1.0)) * (sum_sq / (sum * sum)) - 1.0 / (d - 1.0)
    } else {
        0.0 // a single-site cell has zero spread in every direction
    };
    ShapeDescriptors {
        eigenvalues,
        radius_of_gyration_sq: sum,
        kappa_sq,
    }
}

/// Closed-form eigenvalues of a real symmetric `D x D` matrix — `D` is
/// always 2 or 3 in this crate. Not returned in any particular order;
/// [`shape_descriptors`] sorts afterwards.
fn symmetric_eigenvalues<const D: usize>(g: &[[f64; D]; D]) -> [f64; D] {
    let mut out = [0.0f64; D];
    match D {
        2 => {
            let (a, b, d) = (g[0][0], g[0][1], g[1][1]);
            let tr = a + d;
            let disc = ((a - d) * (a - d) + 4.0 * b * b).max(0.0).sqrt();
            out[0] = (tr + disc) / 2.0;
            out[1] = (tr - disc) / 2.0;
        }
        3 => {
            // Standard closed-form for a real symmetric 3x3 matrix (the
            // trigonometric method via the characteristic polynomial's
            // trace/second-invariant decomposition — see e.g. Smith,
            // "Eigenvalues of a symmetric 3x3 matrix," 1961).
            let p1 = g[0][1] * g[0][1] + g[0][2] * g[0][2] + g[1][2] * g[1][2];
            if p1 == 0.0 {
                out[0] = g[0][0];
                out[1] = g[1][1];
                out[2] = g[2][2];
            } else {
                let q = (g[0][0] + g[1][1] + g[2][2]) / 3.0;
                let p2 = (g[0][0] - q).powi(2)
                    + (g[1][1] - q).powi(2)
                    + (g[2][2] - q).powi(2)
                    + 2.0 * p1;
                let p = (p2 / 6.0).sqrt();
                // B = (1/p) * (G - q*I)
                let b = |i: usize, j: usize| {
                    let delta = if i == j { q } else { 0.0 };
                    (g[i][j] - delta) / p
                };
                let det_b = b(0, 0) * (b(1, 1) * b(2, 2) - b(1, 2) * b(2, 1))
                    - b(0, 1) * (b(1, 0) * b(2, 2) - b(1, 2) * b(2, 0))
                    + b(0, 2) * (b(1, 0) * b(2, 1) - b(1, 1) * b(2, 0));
                let r = (det_b / 2.0).clamp(-1.0, 1.0);
                let phi = r.acos() / 3.0;
                let eig1 = q + 2.0 * p * phi.cos();
                let eig3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
                let eig2 = 3.0 * q - eig1 - eig3;
                out[0] = eig1;
                out[1] = eig2;
                out[2] = eig3;
            }
        }
        _ => unreachable!("cpm-core only instantiates D = 2 or D = 3"),
    }
    out
}

/// One readout-window sample: per-cell detail (the debug/trajectory
/// output; never itself an SBI summary, per this module's population-level
/// constraint) plus the whole-lattice contact graph and medium-component
/// (lumen) report from that instant.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample<const D: usize> {
    pub mcs: u64,
    /// Same order as `State::cells`; `cells[i]` describes the cell at
    /// `State::cells[i]`.
    pub cells: Vec<CellSample<D>>,
    pub contact: ContactGraph,
    pub medium_components: Vec<MediumComponent>,
    pub fragmented_cells: Vec<CellId>,
    /// `State::conservative_energy` at this instant — an exact,
    /// incrementally-maintained running total, included here mainly as a
    /// cross-language reproducibility fingerprint (comparing this sequence
    /// between a Rust-driven and a Python-driven run of the same seed is a
    /// stronger, cheaper check than comparing the raw lattice), though
    /// it's also a useful standalone diagnostic on its own.
    pub conservative_energy: f64,
    /// Row-major, unpadded cell-id lattice at this instant — `None` unless
    /// [`RunOptions::include_lattice`] asked for it. Deliberately opt-in
    /// and out of the always-computed `Sample` fields above: full lattice
    /// and trajectory output stays available for debugging and
    /// visualisation, but shouldn't always be paid for — a real 3D
    /// inference run taking many samples would otherwise carry a full
    /// lattice copy per sample it never looks at.
    pub lattice: Option<Vec<CellId>>,
}

/// Controls what an [`RunResult`] carries beyond the always-computed
/// per-cell/contact/status readouts. `Default` matches every existing
/// caller's expectations (`run`, below) — opting in to `include_lattice`
/// is a deliberate choice at the call site, e.g. for a debug/visualisation
/// consumer.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunOptions {
    pub include_lattice: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellSample<const D: usize> {
    pub id: CellId,
    pub volume: u32,
    pub interface: u32,
    pub centroid: [f64; D],
    pub shape: ShapeDescriptors<D>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RunResult<const D: usize> {
    pub metadata: RunMetadata,
    pub status: SimulationStatus,
    pub samples: Vec<Sample<D>>,
}

fn cell_sample<const D: usize>(cpm: &CPM<D>, cell: &Cell) -> CellSample<D> {
    let i = index_of(cell.id);
    let volume = cpm.state.volume[i];
    let centroid: [f64; D] =
        std::array::from_fn(|d| cpm.state.sum_position[i][d] as f64 / volume as f64);
    let shape = shape_descriptors(&cpm.state.sum_position_outer[i], volume, &centroid);
    CellSample {
        id: cell.id,
        volume,
        interface: cpm.state.interface[i],
        centroid,
        shape,
    }
}

fn take_sample<const D: usize>(
    cpm: &CPM<D>,
    medium_offsets: &[isize],
    include_lattice: bool,
) -> Sample<D> {
    let cells = cpm
        .state
        .cells
        .iter()
        .map(|cell| cell_sample(cpm, cell))
        .collect();
    let labels = label_components(
        &cpm.state.lattice,
        &cpm.connectivity_offsets,
        medium_offsets,
    );
    let lattice = include_lattice.then(|| {
        let mut lattice = Vec::with_capacity(cpm.state.lattice.n_sites());
        cpm.state.lattice.each_interior_coord(|coord| {
            let flat = cpm.state.lattice.flat_index(coord);
            lattice.push(cpm.state.lattice.get(flat));
        });
        lattice
    });
    Sample {
        mcs: cpm.mcs,
        cells,
        contact: cpm.state.contact.clone(),
        medium_components: labels.medium_components,
        fragmented_cells: labels.fragmented_cells,
        conservative_energy: cpm.state.conservative_energy,
        lattice,
    }
}

/// Population-averaged MSD: the observational time origin is the end of
/// burn-in, so `t` is fixed at the *first* readout-window sample — never
/// a sliding origin — and each `tau` lag (one per later sample) is
/// averaged across cells. Population-level only, matching every other
/// SBI-facing readout in this module: no per-cell curve is reported,
/// only the population mean at each lag.
///
/// A cell absent from the origin sample, or from a later sample (e.g.
/// [`SimulationStatus::CellLost`]), is skipped for the lags it's missing
/// from rather than dropping the whole curve; a lag with zero surviving
/// cells reports `0.0`.
pub fn msd<const D: usize>(samples: &[Sample<D>]) -> Vec<f64> {
    let Some(origin) = samples.first() else {
        return Vec::new();
    };
    let origin_centroids: HashMap<CellId, [f64; D]> =
        origin.cells.iter().map(|c| (c.id, c.centroid)).collect();
    samples[1..]
        .iter()
        .map(|sample| {
            let mut sum_sq = 0.0;
            let mut n = 0usize;
            for cell in &sample.cells {
                if let Some(x0) = origin_centroids.get(&cell.id) {
                    sum_sq += (0..D)
                        .map(|d| (cell.centroid[d] - x0[d]).powi(2))
                        .sum::<f64>();
                    n += 1;
                }
            }
            if n == 0 {
                0.0
            } else {
                sum_sq / n as f64
            }
        })
        .collect()
}

/// The frozen equilibration criterion: fit an OLS line to the
/// conservative energy sampled at `sampling_interval_mcs` cadence over the
/// final
/// `NOT_EQUILIBRATED_WINDOW_FRACTION` of `burn_in_mcs`, and flag if the
/// fitted trend's total drift across the window exceeds
/// `NOT_EQUILIBRATED_SLOPE_THRESHOLD` times the fit's residual standard
/// deviation — "is the drift bigger than the noise." Needs at least 3
/// points to fit meaningfully; with fewer, the window (or `burn_in_mcs`)
/// is too short for this check to say anything, so it's skipped (never
/// flagged) rather than guessed at.
fn not_equilibrated(energy_trace: &[f64]) -> bool {
    let n = energy_trace.len();
    if n < 3 {
        return false;
    }
    let xs: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let x_mean = xs.iter().sum::<f64>() / n as f64;
    let y_mean = energy_trace.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        num += (xs[i] - x_mean) * (energy_trace[i] - y_mean);
        den += (xs[i] - x_mean).powi(2);
    }
    if den == 0.0 {
        return false;
    }
    let slope = num / den;
    let intercept = y_mean - slope * x_mean;
    let residual_var: f64 = (0..n)
        .map(|i| (energy_trace[i] - (intercept + slope * xs[i])).powi(2))
        .sum::<f64>()
        / n as f64;
    let residual_std = residual_var.sqrt();
    if residual_std == 0.0 {
        // A perfectly flat trace with zero fluctuation but nonzero slope
        // is definitionally still trending (no noise floor to compare
        // against) — flag whenever the drift itself is nonzero.
        return slope.abs() * (n - 1) as f64 > 0.0;
    }
    let total_drift = slope.abs() * (n - 1) as f64;
    total_drift > NOT_EQUILIBRATED_SLOPE_THRESHOLD * residual_std
}

/// [`run_with`] with every option at its default (no lattice snapshots) —
/// what every existing caller wants.
pub fn run<const D: usize>(cpm: &mut CPM<D>) -> RunResult<D> {
    run_with(cpm, RunOptions::default())
}

/// Windowed run: advances `burn_in_mcs` (discarded, except for periodic
/// energy sampling feeding [`not_equilibrated`]), evaluates equilibration
/// at the end of burn-in, then advances `readout_mcs` in
/// `sampling_interval_mcs`-sized chunks, taking one [`Sample`] after each
/// chunk — the low-level primitive `monte_carlo::run_mcs`'s own doc
/// comment anticipated.
pub fn run_with<const D: usize>(cpm: &mut CPM<D>, options: RunOptions) -> RunResult<D> {
    let medium_offsets =
        crate::lattice::offset_table(&Stencil::<D>::full(), cpm.state.lattice.strides());
    let sampling_interval = cpm.config.sampling_interval_mcs;

    // Burn-in, with periodic energy sampling over the final window
    // fraction feeding the equilibration check.
    let energy_sample_start = cpm
        .config
        .burn_in_mcs
        .saturating_sub((cpm.config.burn_in_mcs as f64 * NOT_EQUILIBRATED_WINDOW_FRACTION) as u64);
    let mut energy_trace = Vec::new();
    let mut burned_in = 0u64;
    while burned_in < cpm.config.burn_in_mcs {
        let chunk = sampling_interval.min(cpm.config.burn_in_mcs - burned_in);
        run_mcs(cpm, chunk);
        burned_in += chunk;
        if burned_in >= energy_sample_start {
            energy_trace.push(cpm.state.conservative_energy);
        }
    }

    let mut status = if not_equilibrated(&energy_trace) {
        SimulationStatus::NotEquilibrated
    } else {
        SimulationStatus::Ok
    };

    // Readout window: one sample per `sampling_interval_mcs`-sized chunk.
    let n_readout_samples = (cpm.config.readout_mcs / sampling_interval.max(1)).max(1);
    let mut samples = Vec::with_capacity(n_readout_samples as usize);
    for _ in 0..n_readout_samples {
        run_mcs(cpm, sampling_interval);
        samples.push(take_sample(cpm, &medium_offsets, options.include_lattice));
    }

    // Fragmentation: the *first* sample at which each cell was seen
    // fragmented wins (status), overriding a still-`Ok` status but never
    // overriding an already-set `NotEquilibrated` — burn-in trending is
    // evaluated first and takes precedence, since it calls into question
    // whether the run even reached a state worth reading at all.
    if matches!(status, SimulationStatus::Ok) {
        if let Some((sample, &id)) = samples
            .iter()
            .find_map(|s| s.fragmented_cells.first().map(|id| (s, id)))
        {
            status = SimulationStatus::Fragmented {
                id,
                at_mcs: sample.mcs,
            };
        }
    }

    // CellLost: a cell at the volume floor for at least the trailing
    // `CELL_LOST_WINDOW_FRACTION` of readout samples.
    if matches!(status, SimulationStatus::Ok) {
        let window = ((samples.len() as f64) * CELL_LOST_WINDOW_FRACTION).ceil() as usize;
        let window = window.max(1).min(samples.len());
        if window > 0 {
            let trailing = &samples[samples.len() - window..];
            'cells: for cell in &cpm.state.cells {
                let all_at_floor = trailing.iter().all(|s| {
                    s.cells
                        .iter()
                        .find(|c| c.id == cell.id)
                        .map(|c| c.volume <= cpm.config.min_cell_volume)
                        .unwrap_or(false)
                });
                if all_at_floor {
                    status = SimulationStatus::CellLost {
                        id: cell.id,
                        at_mcs: trailing[0].mcs,
                    };
                    break 'cells;
                }
            }
        }
    }

    // Degenerate: the cell-adjacency graph (contact pairs with both sides
    // non-medium and a positive count) spans every cell as one component,
    // checked at the final sample only.
    if matches!(status, SimulationStatus::Ok) {
        if let Some(last) = samples.last() {
            if is_fully_connected_cell_graph(last, cpm.state.cells.len()) {
                status = SimulationStatus::Degenerate;
            }
        }
    }

    RunResult {
        metadata: RunMetadata::assemble(cpm),
        status,
        samples,
    }
}

fn is_fully_connected_cell_graph<const D: usize>(sample: &Sample<D>, n_cells: usize) -> bool {
    if n_cells < 2 {
        return false; // "collapsed into one cluster" is meaningless below 2 cells
    }
    let mut adjacency: HashMap<CellId, Vec<CellId>> = HashMap::new();
    for ((a, b), _count) in sample.contact.iter_nonzero() {
        if a == 0 || b == 0 {
            continue;
        }
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    let Some(&start) = adjacency.keys().next() else {
        return false;
    };
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![start];
    visited.insert(start);
    while let Some(node) = stack.pop() {
        if let Some(neighbours) = adjacency.get(&node) {
            for &n in neighbours {
                if visited.insert(n) {
                    stack.push(n);
                }
            }
        }
    }
    visited.len() == n_cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigenvalues_of_a_diagonal_2d_matrix_are_the_diagonal_entries() {
        let g = [[9.0, 0.0], [0.0, 4.0]];
        let eig = symmetric_eigenvalues(&g);
        let mut sorted = eig;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((sorted[0] - 9.0).abs() < 1e-9);
        assert!((sorted[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn eigenvalues_of_a_diagonal_3d_matrix_are_the_diagonal_entries() {
        let g = [[9.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 1.0]];
        let mut eig = symmetric_eigenvalues(&g);
        eig.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((eig[0] - 9.0).abs() < 1e-9);
        assert!((eig[1] - 4.0).abs() < 1e-9);
        assert!((eig[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn eigenvalues_of_a_nondiagonal_2d_matrix_match_hand_computation() {
        // [[2,1],[1,2]] has eigenvalues 3 and 1 (eigenvectors (1,1)/(1,-1)).
        let g = [[2.0, 1.0], [1.0, 2.0]];
        let mut eig = symmetric_eigenvalues(&g);
        eig.sort_by(|a, b| b.partial_cmp(a).unwrap());
        assert!((eig[0] - 3.0).abs() < 1e-9);
        assert!((eig[1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn kappa_sq_is_zero_for_an_isotropic_shape() {
        // Equal eigenvalues (a perfect sphere/disk in the gyration sense).
        let sum_position_outer = [[10i64, 0], [0, 10]];
        let centroid = [0.0, 0.0];
        let d = shape_descriptors(&sum_position_outer, 1, &centroid);
        assert!(d.kappa_sq.abs() < 1e-9, "kappa_sq={}", d.kappa_sq);
    }

    #[test]
    fn kappa_sq_is_one_for_a_degenerate_line() {
        // All spread along one axis only.
        let sum_position_outer = [[100i64, 0], [0, 0]];
        let centroid = [0.0, 0.0];
        let d = shape_descriptors(&sum_position_outer, 1, &centroid);
        assert!((d.kappa_sq - 1.0).abs() < 1e-9, "kappa_sq={}", d.kappa_sq);
    }

    #[test]
    fn not_equilibrated_flags_a_clear_trend() {
        let trace: Vec<f64> = (0..20).map(|i| i as f64 * 10.0).collect();
        assert!(not_equilibrated(&trace));
    }

    #[test]
    fn not_equilibrated_does_not_flag_a_plateau_with_noise() {
        let trace = vec![
            100.0, 101.0, 99.5, 100.3, 99.8, 100.1, 99.9, 100.2, 100.0, 99.7,
        ];
        assert!(!not_equilibrated(&trace));
    }

    #[test]
    fn derived_phi_matches_hand_computation() {
        use crate::cell::CellType;
        use crate::config::UserConfig;
        use crate::initialization::initialize;
        use crate::lattice::Boundary;

        let config = UserConfig::<2> {
            grid: Some([10, 10]),
            boundary: Some(Boundary::Fixed),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 9,
                target_interface: 20,
                lambda_volume: 1.0,
                lambda_interface: 0.1,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![1],
            adhesion: Some(vec![vec![0.0, 0.0], vec![0.0, 0.0]]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(1),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut cpm = CPM::new(config).unwrap();
        initialize(&mut cpm).unwrap();
        let expected = cpm.state.volume[0] as f64 / 100.0;
        assert!((derived_phi(&cpm) - expected).abs() < 1e-12);
    }

    fn cell_sample_at(id: CellId, centroid: [f64; 2]) -> CellSample<2> {
        CellSample {
            id,
            volume: 1,
            interface: 0,
            centroid,
            shape: ShapeDescriptors {
                eigenvalues: [0.0, 0.0],
                radius_of_gyration_sq: 0.0,
                kappa_sq: 0.0,
            },
        }
    }

    fn sample_at(mcs: u64, cells: Vec<CellSample<2>>) -> Sample<2> {
        Sample {
            mcs,
            cells,
            contact: ContactGraph::new(0),
            medium_components: Vec::new(),
            fragmented_cells: Vec::new(),
            conservative_energy: 0.0,
            lattice: None,
        }
    }

    #[test]
    fn msd_is_zero_at_the_origin_sample_and_grows_with_displacement() {
        let samples = vec![
            sample_at(
                0,
                vec![
                    cell_sample_at(1, [0.0, 0.0]),
                    cell_sample_at(2, [10.0, 0.0]),
                ],
            ),
            sample_at(
                1,
                vec![
                    cell_sample_at(1, [1.0, 0.0]),
                    cell_sample_at(2, [11.0, 0.0]),
                ],
            ),
            sample_at(
                2,
                vec![
                    cell_sample_at(1, [3.0, 4.0]),
                    cell_sample_at(2, [10.0, 0.0]),
                ],
            ),
        ];
        let curve = msd(&samples);
        // Both cells displaced by exactly 1 unit along x at tau=1.
        assert_eq!(curve.len(), 2);
        assert!((curve[0] - 1.0).abs() < 1e-12, "curve[0]={}", curve[0]);
        // tau=2: cell 1 moved (3,4) -> |.|^2 = 25; cell 2 unmoved -> 0.
        // Population average over 2 cells: 12.5.
        assert!((curve[1] - 12.5).abs() < 1e-12, "curve[1]={}", curve[1]);
    }

    #[test]
    fn msd_skips_cells_missing_from_a_later_sample_rather_than_dropping_the_lag() {
        let samples = vec![
            sample_at(
                0,
                vec![cell_sample_at(1, [0.0, 0.0]), cell_sample_at(2, [0.0, 0.0])],
            ),
            // Cell 2 lost by tau=1; only cell 1 contributes.
            sample_at(1, vec![cell_sample_at(1, [3.0, 4.0])]),
        ];
        let curve = msd(&samples);
        assert_eq!(curve.len(), 1);
        assert!((curve[0] - 25.0).abs() < 1e-12, "curve[0]={}", curve[0]);
    }

    #[test]
    fn msd_of_a_single_sample_is_empty() {
        let samples = vec![sample_at(0, vec![cell_sample_at(1, [0.0, 0.0])])];
        assert!(msd(&samples).is_empty());
    }

    #[test]
    fn run_end_to_end_produces_ok_status_and_the_expected_sample_count() {
        use crate::cell::CellType;
        use crate::config::UserConfig;
        use crate::initialization::initialize;
        use crate::lattice::Boundary;

        let config = UserConfig::<2> {
            grid: Some([24, 24]),
            boundary: Some(Boundary::Fixed),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 12,
                    target_interface: 24,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 0.0,
                    max_act: 0,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 12,
                    target_interface: 24,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 0.0,
                    max_act: 0,
                },
            ],
            cell_counts: vec![2, 2],
            adhesion: Some(vec![
                vec![0.0, 1.0, 1.0],
                vec![1.0, 0.5, 1.0],
                vec![1.0, 1.0, 0.5],
            ]),
            burn_in_mcs: Some(20),
            readout_mcs: Some(20),
            sampling_interval_mcs: Some(5),
            seed: Some(7),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut cpm = CPM::new(config).unwrap();
        initialize(&mut cpm).unwrap();

        let result = run(&mut cpm);

        assert_eq!(result.status, SimulationStatus::Ok);
        assert_eq!(result.samples.len(), 4);
        assert_eq!(result.samples[0].cells.len(), 4);
        assert_eq!(
            result.metadata.convention_hash,
            convention_hash(&cpm.config)
        );
        assert_eq!(
            result.metadata.model_hash,
            model_hash(&cpm.config, &cpm.state)
        );

        let curve = msd(&result.samples);
        assert_eq!(curve.len(), result.samples.len() - 1);

        // `run()` is `run_with(cpm, RunOptions::default())` — the default
        // must not pay for a lattice snapshot nobody asked for.
        assert!(result.samples[0].lattice.is_none());
    }

    #[test]
    fn run_with_include_lattice_populates_a_snapshot_matching_cell_volumes() {
        use crate::cell::CellType;
        use crate::config::UserConfig;
        use crate::initialization::initialize;
        use crate::lattice::Boundary;
        use std::collections::HashMap;

        let config = UserConfig::<2> {
            grid: Some([16, 16]),
            boundary: Some(Boundary::Fixed),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 10,
                target_interface: 20,
                lambda_volume: 1.0,
                lambda_interface: 0.2,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![2],
            adhesion: Some(vec![vec![0.0, 1.0], vec![1.0, 0.5]]),
            burn_in_mcs: Some(10),
            readout_mcs: Some(10),
            sampling_interval_mcs: Some(5),
            seed: Some(3),
            ..Default::default()
        }
        .resolve()
        .unwrap();
        let mut cpm = CPM::new(config).unwrap();
        initialize(&mut cpm).unwrap();
        let n_sites = cpm.n_sites();

        let result = run_with(
            &mut cpm,
            RunOptions {
                include_lattice: true,
            },
        );

        for sample in &result.samples {
            let lattice = sample.lattice.as_ref().expect("include_lattice was true");
            assert_eq!(lattice.len(), n_sites);

            let mut counts: HashMap<CellId, u32> = HashMap::new();
            for &id in lattice {
                if id != 0 {
                    *counts.entry(id).or_insert(0) += 1;
                }
            }
            for cell in &sample.cells {
                assert_eq!(
                    counts.get(&cell.id).copied().unwrap_or(0),
                    cell.volume,
                    "lattice snapshot site count for cell {} disagrees with its tracked volume",
                    cell.id
                );
            }
        }
    }
}
