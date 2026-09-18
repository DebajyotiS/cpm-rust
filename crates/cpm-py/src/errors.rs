//! Maps `cpm-core`'s `Result`-based configuration and initialisation errors
//! into Python exceptions: configuration errors are `Result` in Rust and
//! surface as Python exceptions with actionable messages, and the simulator
//! never panics on user input. All three error types here are newtype
//! wrappers around a `String` message, and since what matters is an
//! actionable message rather than a specific exception hierarchy, every one
//! maps to the same `PyValueError`.

use cpm_core::cell::CellTypeError;
use cpm_core::config::ConfigError;
use cpm_core::initialization::InitError;
use pyo3::exceptions::PyValueError;
use pyo3::PyErr;

pub fn config_err(e: ConfigError) -> PyErr {
    PyValueError::new_err(e.0)
}

pub fn cell_type_err(e: CellTypeError) -> PyErr {
    PyValueError::new_err(e.0)
}

pub fn init_err(e: InitError) -> PyErr {
    PyValueError::new_err(e.0)
}
