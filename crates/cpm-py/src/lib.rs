//! The thin PyO3 binding layer. This is the only crate that may depend on
//! PyO3 and the only place `AnyCPM` dispatch lives.
//!
//! Python configures the simulator; Rust runs it. One call runs an entire
//! trajectory — no per-move FFI, no Python callbacks in the pixel-copy loop.
//! The builder API (`builder::Simulation`, registered as `CPM`) is the
//! actual Python entry point; `AnyCPM`/`DispatchError` below predate it and
//! remain as Rust-only dispatch types, exercised by their own tests but not
//! part of the Python surface.

// PyO3's `#[pymethods]` macro expands a `PyResult<T>`-returning method into
// callback machinery that runs the error type through `From<PyErr>` even
// though it's already `PyErr` — a known false positive of this lint against
// PyO3-generated code, not a real identity conversion anywhere in this
// crate's own source (see e.g. https://github.com/PyO3/pyo3/issues/2894 for
// the same interaction elsewhere).
#![allow(clippy::useless_conversion)]

mod builder;
mod convert;
mod errors;

use cpm_core::config::{ConfigError, ResolvedConfig};
use cpm_core::model::CPM;
use pyo3::prelude::*;

/// A generic type cannot cross the FFI boundary, so `cpm-py` needs exactly
/// one runtime dispatch point, constructed from the length of the `grid`
/// tuple. Dimensionality is inferred from `grid` and nowhere else — there
/// is no explicit `dimension` argument anywhere in the API.
pub enum AnyCPM {
    D2(CPM<2>),
    D3(CPM<3>),
}

#[derive(Debug)]
pub enum DispatchError {
    /// `grid` had a length other than 2 or 3.
    UnsupportedDimension(usize),
    Config(ConfigError),
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DispatchError::UnsupportedDimension(len) => write!(
                f,
                "grid must have length 2 or 3 (dimensionality is inferred from it), got length {len}"
            ),
            DispatchError::Config(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for DispatchError {}

impl From<ConfigError> for DispatchError {
    fn from(e: ConfigError) -> Self {
        DispatchError::Config(e)
    }
}

impl AnyCPM {
    /// Dispatch on `grid_len` alone. Only the closure matching the actual
    /// length is invoked, so a caller building a `ResolvedConfig<D>` from
    /// dynamic input never has to resolve the dimensionality it isn't
    /// using.
    pub fn new(
        grid_len: usize,
        config_2d: impl FnOnce() -> Result<ResolvedConfig<2>, ConfigError>,
        config_3d: impl FnOnce() -> Result<ResolvedConfig<3>, ConfigError>,
    ) -> Result<Self, DispatchError> {
        match grid_len {
            2 => Ok(AnyCPM::D2(CPM::new(config_2d()?)?)),
            3 => Ok(AnyCPM::D3(CPM::new(config_3d()?)?)),
            other => Err(DispatchError::UnsupportedDimension(other)),
        }
    }
}

/// Module registration for the real extension (`cpm._cpm`, per
/// `pyproject.toml`'s `python-source = "python"` layout). `python/cpm/`'s
/// pure-Python layer (`model.py`, `parameters.py`) wraps `CPM` and is what
/// users actually import; this module is the compiled surface underneath.
#[pymodule]
fn _cpm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<builder::Simulation>()?;
    m.add_class::<convert::PyRunResult>()?;
    m.add_class::<convert::PyRunMetadata>()?;
    m.add_class::<convert::PyStatus>()?;
    m.add_class::<convert::PySample>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpm_core::cell::CellType;
    use cpm_core::config::UserConfig;
    use cpm_core::lattice::Boundary;

    fn minimal_config_2d() -> ResolvedConfig<2> {
        UserConfig::<2> {
            grid: Some([10, 10]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 20,
                target_interface: 50,
                lambda_volume: 1.0,
                lambda_interface: 1.0,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![1],
            adhesion: Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]),
            burn_in_mcs: Some(100),
            readout_mcs: Some(100),
            sampling_interval_mcs: Some(10),
            seed: Some(1),
            ..Default::default()
        }
        .resolve()
        .unwrap()
    }

    fn minimal_config_3d() -> ResolvedConfig<3> {
        UserConfig::<3> {
            grid: Some([10, 10, 10]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![CellType {
                name: "a".into(),
                target_volume: 20,
                target_interface: 50,
                lambda_volume: 1.0,
                lambda_interface: 1.0,
                lambda_act: 0.0,
                max_act: 0,
            }],
            cell_counts: vec![1],
            adhesion: Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]),
            burn_in_mcs: Some(100),
            readout_mcs: Some(100),
            sampling_interval_mcs: Some(10),
            seed: Some(1),
            ..Default::default()
        }
        .resolve()
        .unwrap()
    }

    #[test]
    fn dispatches_to_d2_for_grid_len_2() {
        let result = AnyCPM::new(
            2,
            || Ok(minimal_config_2d()),
            || panic!("3D config builder must not be called for a 2-length grid"),
        );
        assert!(matches!(result, Ok(AnyCPM::D2(_))));
    }

    #[test]
    fn dispatches_to_d3_for_grid_len_3() {
        let result = AnyCPM::new(
            3,
            || panic!("2D config builder must not be called for a 3-length grid"),
            || Ok(minimal_config_3d()),
        );
        assert!(matches!(result, Ok(AnyCPM::D3(_))));
    }

    #[test]
    fn rejects_unsupported_length() {
        let result = AnyCPM::new(
            4,
            || panic!("no config builder should be called for an unsupported length"),
            || panic!("no config builder should be called for an unsupported length"),
        );
        assert!(matches!(
            result,
            Err(DispatchError::UnsupportedDimension(4))
        ));
    }
}
