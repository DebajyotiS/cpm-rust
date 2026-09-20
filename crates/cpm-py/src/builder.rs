//! The Python-facing builder API.
//!
//! Named `Simulation` here, registered in the `_cpm` extension as `"CPM"`
//! (`cpm_core::model::CPM<D>` already owns the name `CPM` inside
//! `cpm-core`; giving the pyclass the same name in its own Rust source
//! would be a needless collision). `python/cpm/model.py`'s public `class
//! CPM` wraps this and is what users actually import.
//!
//! Every method here only mutates Python-side builder state
//! (`UserConfig<D>`, accumulated field by field) — nothing runs in Rust
//! until `run(...)` is called, keeping the guarantee that one Python call
//! runs an entire trajectory. In particular, `initialize(...)` only stashes
//! the chosen `InitializationSpec`; it does not itself place cells.
//! `resolve()` requires `burn_in_mcs`/`readout_mcs`/`sampling_interval_mcs`
//! already set, and those are only supplied to `run(...)` so `run(...)`
//! is where those three fields finally land on the (cloned) `UserConfig<D>`
//! before `resolve()`/`CPM::new` are called.

use crate::convert::{self, PyRunResult};
use crate::errors::{cell_type_err, config_err, init_err};
use cpm_core::cell::CellType;
use cpm_core::config::{
    AcceptanceMode, EnergyTermSet, InitializationSpec, ProposalMode, UserConfig,
};
use cpm_core::initialization;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::neighborhood::NeighborhoodKind;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

fn parse_boundary(s: &str) -> PyResult<Boundary> {
    match s {
        "fixed" => Ok(Boundary::Fixed),
        "periodic" => Ok(Boundary::Periodic),
        other => Err(PyValueError::new_err(format!(
            "unknown boundary {other:?}; expected \"fixed\" or \"periodic\""
        ))),
    }
}

fn parse_neighborhood(s: &str) -> PyResult<NeighborhoodKind> {
    match s {
        "von_neumann" => Ok(NeighborhoodKind::VonNeumann),
        "moore" => Ok(NeighborhoodKind::Moore),
        "eighteen" => Ok(NeighborhoodKind::Eighteen),
        "twenty_six" => Ok(NeighborhoodKind::TwentySix),
        "twenty_six_weighted" => Ok(NeighborhoodKind::TwentySixWeighted),
        other => Err(PyValueError::new_err(format!(
            "unknown neighbourhood {other:?}; expected one of \"von_neumann\", \"moore\", \
             \"eighteen\", \"twenty_six\", \"twenty_six_weighted\""
        ))),
    }
}

fn parse_acceptance(s: &str) -> PyResult<AcceptanceMode> {
    match s {
        "metropolis" => Ok(AcceptanceMode::Metropolis),
        "metropolis_hastings" => Ok(AcceptanceMode::MetropolisHastings),
        other => Err(PyValueError::new_err(format!(
            "unknown acceptance mode {other:?}; expected \"metropolis\" or \"metropolis_hastings\""
        ))),
    }
}

fn parse_proposal(s: &str) -> PyResult<ProposalMode> {
    match s {
        "uniform" => Ok(ProposalMode::Uniform),
        "edge_list" => Ok(ProposalMode::EdgeList),
        other => Err(PyValueError::new_err(format!(
            "unknown proposal mode {other:?}; expected \"uniform\" or \"edge_list\""
        ))),
    }
}

fn parse_active_terms(dict: &HashMap<String, bool>) -> PyResult<EnergyTermSet> {
    let mut terms = EnergyTermSet::default();
    for (key, &value) in dict {
        match key.as_str() {
            "volume" => terms.volume = value,
            "interface" => terms.interface = value,
            "adhesion" => terms.adhesion = value,
            "act" => terms.act = value,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown active_terms key {other:?}; expected one of \"volume\", \
                     \"interface\", \"adhesion\", \"act\""
                )))
            }
        }
    }
    Ok(terms)
}

#[derive(Debug)]
enum Builder {
    D2(UserConfig<2>),
    D3(UserConfig<3>),
}

#[derive(Debug)]
#[pyclass(name = "CPM")]
pub struct Simulation {
    builder: Builder,
    /// `UserConfig::cell_counts` is a positional `Vec<usize>` parallel to
    /// `cell_types`, with no name lookup of its own — `add_cells` needs one,
    /// so it's kept here instead.
    type_index: HashMap<String, usize>,
    active_terms: EnergyTermSet,
}

#[pymethods]
impl Simulation {
    #[new]
    #[pyo3(signature = (
        grid,
        boundary,
        seed,
        copy_neighborhood="von_neumann",
        energy_neighborhood=None,
        connectivity_neighborhood="von_neumann",
        acceptance=None,
        proposal=None,
        active_terms=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        grid: Vec<usize>,
        boundary: &str,
        seed: u64,
        copy_neighborhood: &str,
        energy_neighborhood: Option<&str>,
        connectivity_neighborhood: &str,
        acceptance: Option<&str>,
        proposal: Option<&str>,
        active_terms: Option<HashMap<String, bool>>,
    ) -> PyResult<Self> {
        let boundary = parse_boundary(boundary)?;
        let copy_kind = parse_neighborhood(copy_neighborhood)?;
        let energy_kind = energy_neighborhood.map(parse_neighborhood).transpose()?;
        let connectivity_kind = parse_neighborhood(connectivity_neighborhood)?;
        let acceptance_mode = acceptance.map(parse_acceptance).transpose()?;
        let proposal_mode = proposal.map(parse_proposal).transpose()?;
        let active_terms = match active_terms {
            Some(dict) => parse_active_terms(&dict)?,
            None => EnergyTermSet::default(),
        };

        let grid_len = grid.len();
        let builder = match grid_len {
            2 => {
                let g: [usize; 2] = grid.try_into().expect("grid_len == 2, checked above");
                Builder::D2(UserConfig::<2> {
                    grid: Some(g),
                    boundary: Some(boundary),
                    seed: Some(seed),
                    copy_neighborhood: Some(copy_kind),
                    energy_neighborhood: energy_kind,
                    connectivity_neighborhood: Some(connectivity_kind),
                    acceptance: acceptance_mode,
                    proposal: proposal_mode,
                    ..Default::default()
                })
            }
            3 => {
                let g: [usize; 3] = grid.try_into().expect("grid_len == 3, checked above");
                Builder::D3(UserConfig::<3> {
                    grid: Some(g),
                    boundary: Some(boundary),
                    seed: Some(seed),
                    copy_neighborhood: Some(copy_kind),
                    energy_neighborhood: energy_kind,
                    connectivity_neighborhood: Some(connectivity_kind),
                    acceptance: acceptance_mode,
                    proposal: proposal_mode,
                    ..Default::default()
                })
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "grid must have length 2 or 3 (dimensionality is inferred from it, \
                      not from an explicit `dimension` argument); got length {other}"
                )))
            }
        };

        Ok(Self {
            builder,
            type_index: HashMap::new(),
            active_terms,
        })
    }

    #[pyo3(signature = (
        name, target_volume, target_interface, lambda_volume, lambda_interface,
        lambda_act=0.0, max_act=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn register_cell_type(
        &mut self,
        name: String,
        target_volume: u32,
        target_interface: u32,
        lambda_volume: f64,
        lambda_interface: f64,
        lambda_act: f64,
        max_act: u32,
    ) -> PyResult<()> {
        if self.type_index.contains_key(&name) {
            return Err(PyValueError::new_err(format!(
                "cell type {name:?} has already been added"
            )));
        }
        let cell_type = CellType {
            name: name.clone(),
            target_volume,
            target_interface,
            lambda_volume,
            lambda_interface,
            lambda_act,
            max_act,
        };
        cell_type.validate().map_err(cell_type_err)?;

        let idx = self.type_index.len();
        self.type_index.insert(name, idx);
        match &mut self.builder {
            Builder::D2(cfg) => {
                cfg.cell_types.push(cell_type);
                cfg.cell_counts.push(0);
            }
            Builder::D3(cfg) => {
                cfg.cell_types.push(cell_type);
                cfg.cell_counts.push(0);
            }
        }
        Ok(())
    }

    fn add_cells(&mut self, cell_type: String, n: usize) -> PyResult<()> {
        let idx = *self.type_index.get(&cell_type).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown cell type {cell_type:?}; call register_cell_type first"
            ))
        })?;
        match &mut self.builder {
            Builder::D2(cfg) => cfg.cell_counts[idx] = n,
            Builder::D3(cfg) => cfg.cell_counts[idx] = n,
        }
        Ok(())
    }

    fn set_adhesion(&mut self, matrix: Vec<Vec<f64>>) -> PyResult<()> {
        match &mut self.builder {
            Builder::D2(cfg) => cfg.adhesion = Some(matrix),
            Builder::D3(cfg) => cfg.adhesion = Some(matrix),
        }
        Ok(())
    }

    /// Configures the initial placement; does not execute it. Supplying
    /// `lattice` selects `InitializationSpec::Explicit`; omitting it
    /// selects `Default` (scatter-and-grow). Only these two paths exist —
    /// there is no Rust-side "method"/"packing" choice to route a string
    /// enum to, so those kwargs are not implemented here rather than being
    /// faked.
    #[pyo3(signature = (lattice=None, cell_type_of=None, warn_on_default_init=true))]
    fn initialize(
        &mut self,
        lattice: Option<Vec<u32>>,
        cell_type_of: Option<Vec<usize>>,
        warn_on_default_init: bool,
    ) -> PyResult<()> {
        let spec = match lattice {
            Some(lattice) => {
                let cell_type_of = cell_type_of.ok_or_else(|| {
                    PyValueError::new_err("cell_type_of is required when lattice is supplied")
                })?;
                InitializationSpec::Explicit {
                    lattice,
                    cell_type_of,
                }
            }
            None => InitializationSpec::Default {
                warn_on_default_init,
            },
        };
        match &mut self.builder {
            Builder::D2(cfg) => cfg.initialization = Some(spec),
            Builder::D3(cfg) => cfg.initialization = Some(spec),
        }
        Ok(())
    }

    /// The one call that actually runs anything in Rust: resolve, build,
    /// place cells, then burn in and read out — with the GIL released for
    /// the `output::run` call itself. Returns `(used_default, result)`;
    /// `python/cpm/model.py`'s wrapper is what turns `used_default` into an
    /// actual `CPMInitializationWarning`, keeping that policy decision in
    /// Python.
    #[pyo3(signature = (burn_in_mcs, readout_mcs, sampling_interval_mcs, include_lattice=false))]
    fn run(
        &self,
        py: Python<'_>,
        burn_in_mcs: u64,
        readout_mcs: u64,
        sampling_interval_mcs: u64,
        include_lattice: bool,
    ) -> PyResult<(bool, Py<PyRunResult>)> {
        match &self.builder {
            Builder::D2(cfg) => run_generic::<2>(
                py,
                cfg.clone(),
                self.active_terms,
                burn_in_mcs,
                readout_mcs,
                sampling_interval_mcs,
                include_lattice,
            ),
            Builder::D3(cfg) => run_generic::<3>(
                py,
                cfg.clone(),
                self.active_terms,
                burn_in_mcs,
                readout_mcs,
                sampling_interval_mcs,
                include_lattice,
            ),
        }
    }
}

/// `UserConfig<D>::resolve` requires `burn_in_mcs`/`readout_mcs`/
/// `sampling_interval_mcs` up front, so this is where those three finally
/// land on the (cloned) builder config before resolving. `CPM::new`
/// validates internally (`ResolvedConfig::validate`), so there's no
/// separate `validate()` call needed here.
#[allow(clippy::too_many_arguments)]
fn run_generic<const D: usize>(
    py: Python<'_>,
    mut user_config: UserConfig<D>,
    active_terms: EnergyTermSet,
    burn_in_mcs: u64,
    readout_mcs: u64,
    sampling_interval_mcs: u64,
    include_lattice: bool,
) -> PyResult<(bool, Py<PyRunResult>)> {
    user_config.burn_in_mcs = Some(burn_in_mcs);
    user_config.readout_mcs = Some(readout_mcs);
    user_config.sampling_interval_mcs = Some(sampling_interval_mcs);

    let mut resolved = user_config.resolve().map_err(config_err)?;
    resolved.active_terms = active_terms;
    let grid = resolved.grid.to_vec();

    let mut cpm = CPM::new(resolved).map_err(config_err)?;
    let outcome = initialization::initialize(&mut cpm).map_err(init_err)?;
    let options = cpm_core::output::RunOptions { include_lattice };
    let result = py.allow_threads(|| cpm_core::output::run_with(&mut cpm, options));

    // Cells never change type in v1, so this is stable for the whole run;
    // read from `cpm` after `output::run` rather than needing `output::run`
    // itself to carry it. `cpm.state.cells` is already dense-indexed by
    // `id - 1` (`state::index_of`'s convention), matching every other
    // per-cell array here.
    let cell_type_index = cpm
        .state
        .cells
        .iter()
        .map(|c| c.type_index as u32)
        .collect();
    let cell_type_names = cpm
        .config
        .cell_types
        .iter()
        .map(|t| t.name.clone())
        .collect();

    let py_result = convert::to_py_run_result(py, result, cell_type_index, cell_type_names, &grid)?;
    Ok((outcome.used_default, py_result))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Any test that constructs a `PyErr` (via `?`/`.unwrap_err()` on a
    /// `PyResult`) needs the embedded interpreter to actually initialise
    /// successfully first — see `build.rs` for why this is needed at all
    /// under a `uv`-managed virtualenv. Idempotent and safe to call from
    /// every such test redundantly (`cargo test` runs them in one process).
    fn ensure_python_home() {
        std::env::set_var("PYTHONHOME", env!("PYO3_TEST_PYTHONHOME"));
    }

    #[test]
    fn dispatches_to_d2_for_grid_len_2() {
        let sim = Simulation::new(
            vec![10, 10],
            "periodic",
            1,
            "von_neumann",
            None,
            "von_neumann",
            None,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(sim.builder, Builder::D2(_)));
    }

    #[test]
    fn dispatches_to_d3_for_grid_len_3() {
        let sim = Simulation::new(
            vec![10, 10, 10],
            "periodic",
            1,
            "von_neumann",
            None,
            "von_neumann",
            None,
            None,
            None,
        )
        .unwrap();
        assert!(matches!(sim.builder, Builder::D3(_)));
    }

    #[test]
    fn rejects_unsupported_grid_length() {
        ensure_python_home();
        let err = Simulation::new(
            vec![10, 10, 10, 10],
            "periodic",
            1,
            "von_neumann",
            None,
            "von_neumann",
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("length 2 or 3"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn register_cell_type_rejects_invalid_type() {
        ensure_python_home();
        let mut sim = Simulation::new(
            vec![10, 10],
            "periodic",
            1,
            "von_neumann",
            None,
            "von_neumann",
            None,
            None,
            None,
        )
        .unwrap();
        let err = sim
            .register_cell_type("bad".into(), 20, 50, -1.0, 1.0, 0.0, 0)
            .unwrap_err();
        assert!(err.to_string().contains("non-negative"), "{err}");
    }

    #[test]
    fn add_cells_rejects_unknown_type() {
        ensure_python_home();
        let mut sim = Simulation::new(
            vec![10, 10],
            "periodic",
            1,
            "von_neumann",
            None,
            "von_neumann",
            None,
            None,
            None,
        )
        .unwrap();
        let err = sim.add_cells("ghost".into(), 5).unwrap_err();
        assert!(err.to_string().contains("unknown cell type"), "{err}");
    }

    #[test]
    fn parse_neighborhood_rejects_unknown_string() {
        assert!(parse_neighborhood("hexagonal").is_err());
    }

    #[test]
    fn parse_active_terms_rejects_unknown_key() {
        let mut dict = HashMap::new();
        dict.insert("gravity".to_string(), true);
        assert!(parse_active_terms(&dict).is_err());
    }
}
