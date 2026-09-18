//! Converts `cpm-core`'s dimension-generic readout types (`RunResult<D>`,
//! `Sample<D>`, ...) into Python-facing `#[pyclass]` types, backed by NumPy
//! arrays where the underlying data is itself array-shaped. Everything here
//! is built eagerly, with the GIL already held (`Simulation::run` only
//! releases the GIL around `output::run` itself; this conversion runs after
//! reacquiring it).
//!
//! `ContactGraph` crosses over as a dense `(n_cells+1) x (n_cells+1)`
//! `uint32` matrix (medium at index 0) via `ContactGraph::to_dense` — the
//! natural shape given it's already a dense row-major buffer in Rust, and
//! the raw form that per-pair statistics like sorting, mixing, and
//! clustering need (always computed in Python, not frozen here) without
//! this crate pre-committing to a reduction. It is **upper-triangular, not
//! mirrored**: `ContactGraph::index` always stores a pair at its canonical
//! `(min(a,b), max(a,b))` position, so the lower triangle (including the
//! diagonal) is always zero — a Python consumer that wants a symmetric view
//! needs `m + m.T` itself.
//!
//! `theta` and batch execution live in the Python orchestration layer, so
//! nothing here binds `theta::extract`/`inject`/`names`.

use cpm_core::config::SeedPolicy;
use cpm_core::output::{RunMetadata, RunResult, Sample, SimulationStatus};
use numpy::{PyArray1, PyArray2, PyArrayDyn, PyArrayMethods};
use pyo3::prelude::*;

#[pyclass(get_all)]
#[derive(Clone)]
pub struct PyRunMetadata {
    pub cpm_version: String,
    pub convention_version: u32,
    pub convention_hash: String,
    pub model_hash: String,
    pub seed: u64,
    pub seed_policy: String,
    pub derived_phi: f64,
    pub theta_layout_version: u32,
}

/// `kind` is one of `"Ok"`, `"CellLost"`, `"Fragmented"`, `"NotEquilibrated"`,
/// `"Degenerate"`. `cell_id`/`at_mcs` are populated only for
/// `CellLost`/`Fragmented`, `None` otherwise — a status is a reportable
/// outcome to inspect, never an exception: the simulator detects and
/// reports these conditions rather than trying to repair them.
#[pyclass(get_all)]
#[derive(Clone)]
pub struct PyStatus {
    pub kind: String,
    pub cell_id: Option<u32>,
    pub at_mcs: Option<u64>,
}

/// One readout-window sample. `centroid`/`eigenvalues` are `(n_cells, D)`;
/// every other per-cell array is `(n_cells,)`, same row order as `ids`.
/// `contact` is the dense `(n_cells+1, n_cells+1)` graph described above.
#[pyclass]
pub struct PySample {
    #[pyo3(get)]
    pub mcs: u64,
    /// A cross-language reproducibility fingerprint as much as a readout in
    /// its own right — see `output::Sample`'s doc comment.
    #[pyo3(get)]
    pub conservative_energy: f64,
    #[pyo3(get)]
    pub ids: Py<PyArray1<u32>>,
    #[pyo3(get)]
    pub volume: Py<PyArray1<u32>>,
    #[pyo3(get)]
    pub interface: Py<PyArray1<u32>>,
    #[pyo3(get)]
    pub centroid: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub eigenvalues: Py<PyArray2<f64>>,
    #[pyo3(get)]
    pub radius_of_gyration_sq: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub kappa_sq: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub contact: Py<PyArray2<u32>>,
    #[pyo3(get)]
    pub fragmented_cells: Vec<u32>,
    #[pyo3(get)]
    pub medium_component_volumes: Vec<u32>,
    #[pyo3(get)]
    pub medium_component_touches_boundary: Vec<bool>,
    /// The full cell-id lattice at this instant, shaped `grid` (`(Lx, Ly)`
    /// in 2D, `(Lx, Ly, Lz)` in 3D) — `None` unless `run(...,
    /// include_lattice=True)` asked for it (`cpm_core::output::RunOptions`).
    /// A debug/visualisation escape hatch, not a compact SBI readout: real
    /// cell shapes/topology aren't visible from the per-cell arrays above at
    /// all, only from this.
    #[pyo3(get)]
    pub lattice: Option<Py<PyArrayDyn<u32>>>,
}

#[pyclass]
pub struct PyRunResult {
    #[pyo3(get)]
    pub metadata: Py<PyRunMetadata>,
    #[pyo3(get)]
    pub status: Py<PyStatus>,
    #[pyo3(get)]
    pub samples: Vec<Py<PySample>>,
    /// `cell_type_index[id - 1]` (same indexing as `PySample`'s per-cell
    /// arrays) is that cell's index into `cell_type_names` — cells never
    /// change type in v1, so this one array (not carried per-`Sample`)
    /// covers the whole run. Not part of `Sample` itself since it would
    /// otherwise repeat the same values in every sample for no reason; a
    /// debug/visualisation convenience (e.g. colouring cells by type), not
    /// a `cpm-core` readout.
    #[pyo3(get)]
    pub cell_type_index: Py<PyArray1<u32>>,
    #[pyo3(get)]
    pub cell_type_names: Vec<String>,
}

fn seed_policy_str(policy: SeedPolicy) -> &'static str {
    match policy {
        SeedPolicy::Independent => "independent",
        SeedPolicy::CommonRandomNumbers => "common_random_numbers",
        SeedPolicy::Explicit => "explicit",
    }
}

fn convert_metadata(m: RunMetadata) -> PyRunMetadata {
    PyRunMetadata {
        cpm_version: m.cpm_version,
        convention_version: m.convention_version,
        convention_hash: m.convention_hash,
        model_hash: m.model_hash,
        seed: m.seed,
        seed_policy: seed_policy_str(m.seed_policy).to_string(),
        derived_phi: m.derived_phi,
        theta_layout_version: m.theta_layout_version,
    }
}

fn convert_status(status: SimulationStatus) -> PyStatus {
    match status {
        SimulationStatus::Ok => PyStatus {
            kind: "Ok".to_string(),
            cell_id: None,
            at_mcs: None,
        },
        SimulationStatus::CellLost { id, at_mcs } => PyStatus {
            kind: "CellLost".to_string(),
            cell_id: Some(id),
            at_mcs: Some(at_mcs),
        },
        SimulationStatus::Fragmented { id, at_mcs } => PyStatus {
            kind: "Fragmented".to_string(),
            cell_id: Some(id),
            at_mcs: Some(at_mcs),
        },
        SimulationStatus::NotEquilibrated => PyStatus {
            kind: "NotEquilibrated".to_string(),
            cell_id: None,
            at_mcs: None,
        },
        SimulationStatus::Degenerate => PyStatus {
            kind: "Degenerate".to_string(),
            cell_id: None,
            at_mcs: None,
        },
    }
}

fn convert_sample<const D: usize>(
    py: Python<'_>,
    sample: &Sample<D>,
    grid: &[usize],
) -> PyResult<Py<PySample>> {
    let n_cells = sample.cells.len();
    let mut ids = Vec::with_capacity(n_cells);
    let mut volume = Vec::with_capacity(n_cells);
    let mut interface = Vec::with_capacity(n_cells);
    let mut centroid_rows: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    let mut eigenvalue_rows: Vec<Vec<f64>> = Vec::with_capacity(n_cells);
    let mut radius_of_gyration_sq = Vec::with_capacity(n_cells);
    let mut kappa_sq = Vec::with_capacity(n_cells);
    for cell in &sample.cells {
        ids.push(cell.id);
        volume.push(cell.volume);
        interface.push(cell.interface);
        centroid_rows.push(cell.centroid.to_vec());
        eigenvalue_rows.push(cell.shape.eigenvalues.to_vec());
        radius_of_gyration_sq.push(cell.shape.radius_of_gyration_sq);
        kappa_sq.push(cell.shape.kappa_sq);
    }

    let (n, dense) = sample.contact.to_dense();
    let contact_rows: Vec<Vec<u32>> = dense.chunks(n).map(|row| row.to_vec()).collect();

    let medium_component_volumes = sample.medium_components.iter().map(|m| m.volume).collect();
    let medium_component_touches_boundary = sample
        .medium_components
        .iter()
        .map(|m| m.touches_boundary)
        .collect();

    let py_sample = PySample {
        mcs: sample.mcs,
        conservative_energy: sample.conservative_energy,
        ids: PyArray1::from_vec_bound(py, ids).unbind(),
        volume: PyArray1::from_vec_bound(py, volume).unbind(),
        interface: PyArray1::from_vec_bound(py, interface).unbind(),
        centroid: PyArray2::from_vec2_bound(py, &centroid_rows)
            .expect("every row has length D by construction")
            .unbind(),
        eigenvalues: PyArray2::from_vec2_bound(py, &eigenvalue_rows)
            .expect("every row has length D by construction")
            .unbind(),
        radius_of_gyration_sq: PyArray1::from_vec_bound(py, radius_of_gyration_sq).unbind(),
        kappa_sq: PyArray1::from_vec_bound(py, kappa_sq).unbind(),
        contact: PyArray2::from_vec2_bound(py, &contact_rows)
            .expect("every row has length n by construction")
            .unbind(),
        fragmented_cells: sample.fragmented_cells.clone(),
        medium_component_volumes,
        medium_component_touches_boundary,
        lattice: match &sample.lattice {
            Some(flat) => {
                let shaped = PyArray1::from_vec_bound(py, flat.clone()).reshape(grid.to_vec())?;
                Some(shaped.unbind())
            }
            None => None,
        },
    };
    Py::new(py, py_sample)
}

pub fn to_py_run_result<const D: usize>(
    py: Python<'_>,
    result: RunResult<D>,
    cell_type_index: Vec<u32>,
    cell_type_names: Vec<String>,
    grid: &[usize],
) -> PyResult<Py<PyRunResult>> {
    let metadata = Py::new(py, convert_metadata(result.metadata))?;
    let status = Py::new(py, convert_status(result.status))?;
    let mut samples = Vec::with_capacity(result.samples.len());
    for sample in &result.samples {
        samples.push(convert_sample::<D>(py, sample, grid)?);
    }
    let cell_type_index = PyArray1::from_vec_bound(py, cell_type_index).unbind();
    Py::new(
        py,
        PyRunResult {
            metadata,
            status,
            samples,
            cell_type_index,
            cell_type_names,
        },
    )
}
