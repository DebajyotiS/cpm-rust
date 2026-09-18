//! The resolved configuration schema — the centre of the design.
//! Everything scientifically meaningful is a field of [`ResolvedConfig`]; if
//! a behaviour depends on a value, that value lives here, because otherwise
//! it is invisible to [`convention_hash`] and a run is not reproducible from
//! its own record.
//!
//! [`UserConfig`] is the partial, user-supplied form; [`UserConfig::resolve`]
//! fills in every documented default and returns the fully-populated
//! [`ResolvedConfig`]. Only the resolved form is hashed and stored.

use crate::cell::CellType;
use crate::lattice::{Boundary, CellId};
use crate::neighborhood::NeighborhoodKind;
use crate::state::State;
use serde::Serialize;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl From<crate::cell::CellTypeError> for ConfigError {
    fn from(e: crate::cell::CellTypeError) -> Self {
        ConfigError(e.0)
    }
}

impl From<crate::neighborhood::NeighborhoodKindError> for ConfigError {
    fn from(e: crate::neighborhood::NeighborhoodKindError) -> Self {
        ConfigError(e.0)
    }
}

/// Standard CPM acceptance versus the Hastings-corrected test instrument.
/// `Metropolis` is production; `MetropolisHastings` exists solely to
/// enable the exact Boltzmann validation and must never be used for
/// production runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum AcceptanceMode {
    Metropolis,
    MetropolisHastings,
}

/// How a proposed copy's *target* site is drawn. Both modes propose the
/// same underlying chain — `EdgeList` only removes same-cell attempts that
/// `Uniform` would have early-outed on anyway, via binomial thinning
/// (`monte_carlo::run_mcs`), so the two are statistically equivalent, not
/// different models. They are not bit-identical for a given seed, though
/// (a different sequence of RNG draws), so `proposal` is still a
/// `convention_hash` field like every other simulation-control choice.
///
/// `Uniform` stays the default: every frozen trajectory
/// (`dimension_consistency.rs`) and the exact-Boltzmann `MetropolisHastings`
/// validation mode depend on it unchanged. `EdgeList` is what a 3D
/// inference run should select explicitly — a per-run performance choice
/// once 3D inference is the actual goal, not a change to what the
/// simulator means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum ProposalMode {
    Uniform,
    EdgeList,
}

/// The policy, not the seed value itself, is a convention; the seed
/// value is run-specific and belongs in run metadata, not `convention_hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub enum SeedPolicy {
    /// Default for SBI: each simulation's seed is derived deterministically
    /// from `(master_seed, simulation_index)`.
    Independent,
    /// Debugging and sensitivity studies only — never for SBI training data.
    CommonRandomNumbers,
    /// Caller supplies the exact seed.
    Explicit,
}

/// Which energy terms are active. Volume, interface and adhesion are
/// always on; `act` defaults to `false` and enables the (non-conservative)
/// Act term when set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct EnergyTermSet {
    pub volume: bool,
    pub interface: bool,
    pub adhesion: bool,
    pub act: bool,
}

impl Default for EnergyTermSet {
    fn default() -> Self {
        Self {
            volume: true,
            interface: true,
            adhesion: true,
            act: false,
        }
    }
}

/// Explicit placement or default scatter-and-grow, both implemented in
/// `initialization.rs`. `Explicit` scopes to a fully-specified initial
/// lattice (row-major, unpadded, length `N_sites`) rather than also
/// supporting bare seed positions — a positions-only convenience belongs
/// in the Python layer, built on top of this representation, not as a
/// second Rust-level representation.
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub enum InitializationSpec {
    Default {
        warn_on_default_init: bool,
    },
    Explicit {
        /// Row-major, unpadded, length `N_sites`. Cell ids must be
        /// 1-indexed and contiguous; 0 is medium.
        lattice: Vec<CellId>,
        /// `cell_type_of[id - 1]` is that cell's index into `cell_types`.
        /// Length gives the declared cell count directly — it must match
        /// the declared cell types' assignments.
        cell_type_of: Vec<usize>,
    },
}

impl Default for InitializationSpec {
    fn default() -> Self {
        InitializationSpec::Default {
            warn_on_default_init: true,
        }
    }
}

/// Partial, user-supplied configuration. Every field with a documented
/// default is `Option`; [`resolve`](UserConfig::resolve) fills it in.
/// Fields with no sensible default (grid, cell types, adhesion, timing,
/// seed) are required and [`resolve`](UserConfig::resolve) errors clearly if
/// they are missing, rather than guessing.
#[derive(Debug, Clone)]
pub struct UserConfig<const D: usize> {
    pub grid: Option<[usize; D]>,
    pub copy_neighborhood: Option<NeighborhoodKind>,
    pub energy_neighborhood: Option<NeighborhoodKind>,
    pub connectivity_neighborhood: Option<NeighborhoodKind>,
    /// No default: the boundary convention materially affects results, and
    /// an undecided convention must never be silently defaulted.
    pub boundary: Option<Boundary>,
    pub acceptance: Option<AcceptanceMode>,
    pub proposal: Option<ProposalMode>,
    pub cell_types: Vec<CellType>,
    /// How many cells of each type to create, same order as `cell_types`.
    /// Only consumed by the default scatter-and-grow path
    /// (`initialization.rs`) — `InitializationSpec::Explicit` derives its
    /// cell count from the supplied lattice instead.
    pub cell_counts: Vec<usize>,
    /// Symmetric, size `(n_types + 1) x (n_types + 1)`, index 0 = medium.
    pub adhesion: Option<Vec<Vec<f64>>>,
    pub initialization: Option<InitializationSpec>,
    pub burn_in_mcs: Option<u64>,
    pub readout_mcs: Option<u64>,
    pub sampling_interval_mcs: Option<u64>,
    pub seed: Option<u64>,
    pub seed_policy: Option<SeedPolicy>,
    /// Opaque strings for now; carried so the schema shape stays stable
    /// ahead of a future typed `ReadoutKind`.
    pub requested_readouts: Vec<String>,
}

impl<const D: usize> Default for UserConfig<D> {
    fn default() -> Self {
        Self {
            grid: None,
            copy_neighborhood: None,
            energy_neighborhood: None,
            connectivity_neighborhood: None,
            boundary: None,
            acceptance: None,
            proposal: None,
            cell_types: Vec::new(),
            cell_counts: Vec::new(),
            adhesion: None,
            initialization: None,
            burn_in_mcs: None,
            readout_mcs: None,
            sampling_interval_mcs: None,
            seed: None,
            seed_policy: None,
            requested_readouts: Vec::new(),
        }
    }
}

impl<const D: usize> UserConfig<D> {
    /// Fill in every documented default (copy, energy and connectivity
    /// neighbourhoods, acceptance mode, initialisation, seed policy);
    /// everything else must already be present. Does **not** validate
    /// the result — call [`ResolvedConfig::validate`] next.
    pub fn resolve(self) -> Result<ResolvedConfig<D>, ConfigError> {
        let grid = self
            .grid
            .ok_or_else(|| ConfigError("grid is required".into()))?;
        let adhesion = self
            .adhesion
            .ok_or_else(|| ConfigError("adhesion matrix is required".into()))?;
        let burn_in_mcs = self
            .burn_in_mcs
            .ok_or_else(|| ConfigError("burn_in_mcs is required".into()))?;
        let readout_mcs = self
            .readout_mcs
            .ok_or_else(|| ConfigError("readout_mcs is required".into()))?;
        let sampling_interval_mcs = self
            .sampling_interval_mcs
            .ok_or_else(|| ConfigError("sampling_interval_mcs is required".into()))?;
        let boundary = self
            .boundary
            .ok_or_else(|| ConfigError("boundary is required (no default value)".into()))?;
        let seed = self
            .seed
            .ok_or_else(|| ConfigError("seed is required".into()))?;

        // 2D default is Moore (8), 3D default is Eighteen (18) — never
        // blindly inherit Moore into 3D.
        let energy_default = if D == 2 {
            NeighborhoodKind::Moore
        } else {
            NeighborhoodKind::Eighteen
        };

        Ok(ResolvedConfig {
            grid,
            copy_neighborhood: self
                .copy_neighborhood
                .unwrap_or(NeighborhoodKind::VonNeumann),
            energy_neighborhood: self.energy_neighborhood.unwrap_or(energy_default),
            connectivity_neighborhood: self
                .connectivity_neighborhood
                .unwrap_or(NeighborhoodKind::VonNeumann),
            boundary,
            acceptance: self.acceptance.unwrap_or(AcceptanceMode::Metropolis),
            proposal: self.proposal.unwrap_or(ProposalMode::Uniform),
            cell_types: self.cell_types,
            cell_counts: self.cell_counts,
            adhesion,
            active_terms: EnergyTermSet::default(),
            initialization: self.initialization.unwrap_or_default(),
            burn_in_mcs,
            readout_mcs,
            sampling_interval_mcs,
            min_cell_volume: 1, // fixed in v1
            seed,
            seed_policy: self.seed_policy.unwrap_or(SeedPolicy::Independent),
            requested_readouts: self.requested_readouts,
        })
    }
}

/// The fully-resolved configuration: every default filled in. This is
/// the only form that is hashed ([`convention_hash`]) and stored.
#[derive(Debug, Clone)]
pub struct ResolvedConfig<const D: usize> {
    pub grid: [usize; D],
    pub copy_neighborhood: NeighborhoodKind,
    pub energy_neighborhood: NeighborhoodKind,
    pub connectivity_neighborhood: NeighborhoodKind,
    pub boundary: Boundary,
    pub acceptance: AcceptanceMode,
    pub proposal: ProposalMode,
    pub cell_types: Vec<CellType>,
    pub cell_counts: Vec<usize>,
    pub adhesion: Vec<Vec<f64>>,
    pub active_terms: EnergyTermSet,
    pub initialization: InitializationSpec,
    pub burn_in_mcs: u64,
    pub readout_mcs: u64,
    pub sampling_interval_mcs: u64,
    pub min_cell_volume: u32,
    pub seed: u64,
    pub seed_policy: SeedPolicy,
    pub requested_readouts: Vec<String>,
}

impl<const D: usize> ResolvedConfig<D> {
    /// Configuration errors return `Result` and surface in Python as
    /// exceptions with actionable messages — never panic on user input.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.grid.contains(&0) {
            return Err(ConfigError(format!(
                "every grid dimension must be nonzero, got {:?}",
                self.grid
            )));
        }

        // The simple-point tables (`simple_point.rs`) are built for the one
        // (6, 26)/(4, 8) pairing this crate actually uses — cells connected
        // via von Neumann adjacency, background via the complementary full
        // neighbourhood. `connectivity_neighborhood` exists in the schema
        // for forward extensibility, as explicit offset lists so future
        // custom stencils are possible, but supporting an arbitrary
        // cell-connectivity stencil would mean deriving *and verifying* a
        // different topological-number pairing for each one, which nobody
        // has asked for yet. Fail loudly rather than silently assume a
        // pairing that was never checked.
        if self.connectivity_neighborhood != NeighborhoodKind::VonNeumann {
            return Err(ConfigError(format!(
                "connectivity_neighborhood must be VonNeumann in this version \
                 (the simple-point tables only support the (4,8)/(6,26) pairing), \
                 got {:?}",
                self.connectivity_neighborhood
            )));
        }

        // copy_neighborhood must be a subset of connectivity_neighborhood.
        // The losing-cell-only connectivity check is invalid without this.
        let copy_stencil = self.copy_neighborhood.to_stencil::<D>()?;
        let connectivity_stencil = self.connectivity_neighborhood.to_stencil::<D>()?;
        if !copy_stencil.is_subset_of(&connectivity_stencil) {
            return Err(ConfigError(format!(
                "copy_neighborhood {:?} must be a subset of connectivity_neighborhood {:?}",
                self.copy_neighborhood, self.connectivity_neighborhood
            )));
        }
        // Energy neighbourhood only needs to resolve for this D; no subset
        // relationship is required of it.
        let energy_stencil = self.energy_neighborhood.to_stencil::<D>()?;

        // `State::interface` stores `I_c` as `u32`, which is only exact
        // under an unweighted energy neighbourhood. `TwentySixWeighted`
        // (and any `Custom` stencil with non-uniform weights) would make
        // `I_c` genuinely fractional, which the current bookkeeping can't
        // represent. Reject rather than silently truncate.
        if energy_stencil.weights.iter().any(|&w| w != 1.0) {
            return Err(ConfigError(
                "weighted energy neighbourhoods are not supported yet: interface \
                 bookkeeping is integer-typed; use an unweighted energy_neighborhood \
                 (Moore, Eighteen, or TwentySix)"
                    .into(),
            ));
        }

        if self.cell_types.is_empty() {
            return Err(ConfigError("at least one cell type is required".into()));
        }
        for t in &self.cell_types {
            t.validate()?;
        }

        if self.cell_counts.len() != self.cell_types.len() {
            return Err(ConfigError(format!(
                "cell_counts has {} entries, expected one per cell type ({})",
                self.cell_counts.len(),
                self.cell_types.len()
            )));
        }

        let n = self.cell_types.len();
        let expected_size = n + 1;
        if self.adhesion.len() != expected_size
            || self.adhesion.iter().any(|row| row.len() != expected_size)
        {
            return Err(ConfigError(format!(
                "adhesion matrix must be {expected_size}x{expected_size} (n_types + 1 for medium), got {}x{}",
                self.adhesion.len(),
                self.adhesion.first().map(|r| r.len()).unwrap_or(0)
            )));
        }
        for i in 0..expected_size {
            for j in 0..expected_size {
                let a = self.adhesion[i][j];
                let b = self.adhesion[j][i];
                if !a.is_finite() {
                    return Err(ConfigError(format!(
                        "adhesion[{i}][{j}] = {a} is not finite"
                    )));
                }
                if (a - b).abs() > 0.0 {
                    return Err(ConfigError(format!(
                        "adhesion matrix must be symmetric in v1: [{i}][{j}] = {a} but [{j}][{i}] = {b}"
                    )));
                }
            }
        }

        if self.readout_mcs == 0 {
            return Err(ConfigError("readout_mcs must be positive".into()));
        }
        if self.sampling_interval_mcs == 0 {
            return Err(ConfigError("sampling_interval_mcs must be positive".into()));
        }

        // The `MetropolisHastings` reversibility argument depends on the
        // *uniform* forward/reverse proposal symmetry (the forward move
        // tests cell A, the reverse tests cell B, drawn from the same
        // uniform-over-the-interior distribution). `EdgeList` changes how
        // the target site is drawn, so nothing currently establishes that
        // MH's reversibility argument still holds under it — and MH is a
        // test-only validation instrument, with no need to combine it with
        // a production-throughput optimisation anyway.
        if self.proposal == ProposalMode::EdgeList
            && self.acceptance == AcceptanceMode::MetropolisHastings
        {
            return Err(ConfigError(
                "proposal = EdgeList is not valid with acceptance = MetropolisHastings: \
                 the exact-Boltzmann validation's reversibility argument depends on the \
                 uniform proposal's forward/reverse symmetry, which EdgeList does not \
                 establish"
                    .into(),
            ));
        }

        Ok(())
    }
}

/// The subset of [`ResolvedConfig`] that [`convention_hash`] covers.
/// Deliberately excludes every field that `theta` can inject
/// (`lambda_volume`, `lambda_interface`, `lambda_act`, `max_act`, the
/// adhesion matrix values) — those vary across simulations that must still
/// share one `convention_hash` within a training set. `target_volume` and
/// `target_interface` are safe to include: they are calibration constants
/// theta never touches, so they are constant across such a set.
#[derive(Serialize)]
struct CellTypeConventionView<'a> {
    name: &'a str,
    target_volume: u32,
    target_interface: u32,
}

#[derive(Serialize)]
struct ConventionView<'a> {
    dimensionality: usize,
    // `serde` only implements `Serialize` for fixed-size arrays up to a
    // handful of literal lengths, not generically over a const parameter
    // `D`, so the grid travels through the hash as a `Vec` instead of
    // `[usize; D]`.
    grid: Vec<usize>,
    copy_neighborhood: &'a NeighborhoodKind,
    energy_neighborhood: &'a NeighborhoodKind,
    connectivity_neighborhood: &'a NeighborhoodKind,
    boundary: Boundary,
    acceptance: AcceptanceMode,
    proposal: ProposalMode,
    cell_types: Vec<CellTypeConventionView<'a>>,
    active_terms: EnergyTermSet,
    initialization: &'a InitializationSpec,
    burn_in_mcs: u64,
    readout_mcs: u64,
    sampling_interval_mcs: u64,
    min_cell_volume: u32,
    seed_policy: SeedPolicy,
    requested_readouts: &'a [String],
    /// [`CONVENTION_VERSION`]: not a config field (nothing here is
    /// user-set), but including it means every run's `convention_hash`
    /// changes automatically if a frozen rule this crate owns but doesn't
    /// expose as a `ResolvedConfig` field — most concretely, `output.rs`'s
    /// `NotEquilibrated` criterion's window fraction and slope threshold —
    /// is ever revised. This extends the same "the convention hash must
    /// cover every frozen rule" principle to that one other
    /// frozen-but-not-user-configurable rule.
    convention_version: u32,
}

/// Bumped whenever a frozen convention this crate owns but doesn't expose
/// as a `ResolvedConfig` field changes — currently just `output.rs`'s
/// `NotEquilibrated` window fraction and slope threshold. Lives here
/// (rather than in `output.rs`, which needs `convention_hash` itself) to
/// avoid a circular module dependency; `theta::THETA_LAYOUT_VERSION` is the
/// parallel case for `theta`'s own positional layout specifically.
pub const CONVENTION_VERSION: u32 = 1;

/// The convention hash: a fingerprint of everything that makes two runs
/// "the same simulator." Computed from the resolved config, not the
/// user's input syntax. Excludes `seed` (run-specific) and every
/// `theta`-eligible numeric value (see [`ConventionView`]) — two
/// simulations at different `theta` under the same conventions must hash
/// identically, since downstream tooling requires every simulation in a
/// training set to share one `convention_hash`.
pub fn convention_hash<const D: usize>(config: &ResolvedConfig<D>) -> String {
    let view = ConventionView {
        dimensionality: D,
        convention_version: CONVENTION_VERSION,
        grid: config.grid.to_vec(),
        copy_neighborhood: &config.copy_neighborhood,
        energy_neighborhood: &config.energy_neighborhood,
        connectivity_neighborhood: &config.connectivity_neighborhood,
        boundary: config.boundary,
        acceptance: config.acceptance,
        proposal: config.proposal,
        cell_types: config
            .cell_types
            .iter()
            .map(|t| CellTypeConventionView {
                name: &t.name,
                target_volume: t.target_volume,
                target_interface: t.target_interface,
            })
            .collect(),
        active_terms: config.active_terms,
        initialization: &config.initialization,
        burn_in_mcs: config.burn_in_mcs,
        readout_mcs: config.readout_mcs,
        sampling_interval_mcs: config.sampling_interval_mcs,
        min_cell_volume: config.min_cell_volume,
        seed_policy: config.seed_policy,
        requested_readouts: &config.requested_readouts,
    };
    // A struct (not a map) serializes fields in declaration order, so this
    // is deterministic without needing sorted-key canonicalisation.
    let bytes = serde_json::to_vec(&view).expect("ConventionView is always serializable");
    blake3::hash(&bytes).to_hex().to_string()
}

#[derive(Serialize)]
struct ModelHashView {
    convention_hash: String,
    /// Row-major, unpadded cell-id lattice — the same shape as
    /// `InitializationSpec::Explicit::lattice` — read via
    /// `Lattice::each_interior_coord` so the ordering is fixed regardless of
    /// internal halo padding. For `InitializationSpec::Explicit` this
    /// duplicates what `convention_hash` already covers (the spec *is* the
    /// placement); for `InitializationSpec::Default` this is the only
    /// record of which particular scatter-and-grow draw actually happened.
    lattice: Vec<CellId>,
    /// `cells[id - 1].type_index`, i.e. the resolved cell-id -> cell-type
    /// assignment, redundant with `lattice` plus `cell_types` for
    /// `Explicit` init but included for both paths so `model_hash` never
    /// depends on which path was taken to be a complete record.
    cell_type_of: Vec<usize>,
}

/// The model hash: `convention_hash` plus the *resolved* initial
/// placement. Two runs sharing a `convention_hash` (same conventions,
/// same `InitializationSpec`) can still differ here whenever
/// `InitializationSpec::Default` resolves a different placement — a
/// different seed, or the same seed after an unrelated upstream change to
/// the scatter-and-grow algorithm. A defaulted run is only reproducible
/// from its own output record if that record captures *which* placement
/// it actually got, not just that "the default was used": that is
/// exactly what this hash exists to pin down, on top of what
/// `convention_hash` already covers.
pub fn model_hash<const D: usize>(config: &ResolvedConfig<D>, state: &State<D>) -> String {
    let mut lattice = Vec::with_capacity(state.lattice.n_sites());
    state.lattice.each_interior_coord(|coord| {
        let flat = state.lattice.flat_index(coord);
        lattice.push(state.lattice.get(flat));
    });
    let cell_type_of = state.cells.iter().map(|c| c.type_index).collect();
    let view = ModelHashView {
        convention_hash: convention_hash(config),
        lattice,
        cell_type_of,
    };
    let bytes = serde_json::to_vec(&view).expect("ModelHashView is always serializable");
    blake3::hash(&bytes).to_hex().to_string()
}

/// Shared config fixtures for other modules' tests (`theta.rs`'s round-trip
/// test in particular). `pub(crate)` and `cfg(test)`-gated: test-only, never
/// part of the public API.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::cell::CellType;

    /// `target_interface` for both types is calibrated, not guessed:
    /// measured on 2026-09-15 via
    /// `crates/cpm-core/examples/calibrate_targets.rs`, relaxing a single
    /// cell of the given `target_volume` under this file's actual 2D
    /// conventions (von Neumann copy/connectivity, Moore energy
    /// neighbourhood, Fixed boundary, large grid) with `lambda_interface =
    /// 0` (not presupposing the answer) and cell-medium adhesion set to
    /// this fixture's own `J` value for that type (5.0 for epithelial, 6.0
    /// for stem) so the relaxation has real surface tension.
    ///
    /// That last point is a deliberate, measurement-driven departure from
    /// the usual calibration procedure of disabling adhesion entirely:
    /// tried literally (adhesion = 0 *and* `lambda_interface` = 0), nothing
    /// in `H` penalises surface area at all, and the equilibrium is a
    /// thin, branching, non-compact shape (confirmed stable across a 10x
    /// longer burn-in and confirmed visually, not a transient) — a genuine
    /// property of zero-surface-tension CPM dynamics (entropy over
    /// connected-shape space favours ramified configurations), not a bug,
    /// but not the compact shape "I*" is meant to describe either. See
    /// `calibrate_targets.rs`'s own doc comment for the full evidence
    /// trail.
    pub fn two_type_config() -> ResolvedConfig<2> {
        let epithelial = CellType {
            name: "epithelial".into(),
            target_volume: 50,
            target_interface: 75, // calibrated: mean 74.753, std 0.973
            lambda_volume: 10.0,
            lambda_interface: 2.0,
            lambda_act: 0.0,
            max_act: 0,
        };
        let stem = CellType {
            name: "stem".into(),
            target_volume: 40,
            target_interface: 66, // calibrated: mean 66.097, std 0.430
            lambda_volume: 15.0,
            lambda_interface: 3.0,
            lambda_act: 0.0,
            max_act: 0,
        };
        // Rows/cols: 0 = medium, 1 = epithelial, 2 = stem.
        let adhesion = vec![
            vec![0.0, 5.0, 6.0],
            vec![5.0, 2.0, 4.0],
            vec![6.0, 4.0, 3.0],
        ];
        UserConfig::<2> {
            grid: Some([30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![epithelial, stem],
            cell_counts: vec![5, 5],
            adhesion: Some(adhesion),
            burn_in_mcs: Some(1000),
            readout_mcs: Some(1000),
            sampling_interval_mcs: Some(100),
            seed: Some(42),
            ..Default::default()
        }
        .resolve()
        .expect("fixture config must resolve")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use test_support::two_type_config;

    fn minimal_user_config() -> UserConfig<2> {
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
            adhesion: Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]),
            burn_in_mcs: Some(100),
            readout_mcs: Some(100),
            sampling_interval_mcs: Some(10),
            seed: Some(1),
            ..Default::default()
        }
    }

    #[test]
    fn resolve_fills_documented_defaults() {
        let resolved = minimal_user_config().resolve().unwrap();
        assert_eq!(resolved.copy_neighborhood, NeighborhoodKind::VonNeumann);
        assert_eq!(resolved.energy_neighborhood, NeighborhoodKind::Moore); // D = 2
        assert_eq!(
            resolved.connectivity_neighborhood,
            NeighborhoodKind::VonNeumann
        );
        assert_eq!(resolved.acceptance, AcceptanceMode::Metropolis);
        assert_eq!(resolved.min_cell_volume, 1);
        assert_eq!(resolved.seed_policy, SeedPolicy::Independent);
        assert_eq!(
            resolved.initialization,
            InitializationSpec::Default {
                warn_on_default_init: true
            }
        );
    }

    #[test]
    fn energy_default_is_eighteen_in_3d() {
        let user = UserConfig::<3> {
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
            adhesion: Some(vec![vec![0.0, 1.0], vec![1.0, 0.0]]),
            burn_in_mcs: Some(100),
            readout_mcs: Some(100),
            sampling_interval_mcs: Some(10),
            seed: Some(1),
            ..Default::default()
        };
        let resolved = user.resolve().unwrap();
        assert_eq!(resolved.energy_neighborhood, NeighborhoodKind::Eighteen);
    }

    #[test]
    fn resolve_errors_on_missing_required_fields() {
        assert!(UserConfig::<2>::default().resolve().is_err());

        let mut missing_boundary = minimal_user_config();
        missing_boundary.boundary = None;
        assert!(missing_boundary.resolve().is_err());
    }

    #[test]
    fn validate_accepts_default_neighbourhoods() {
        assert!(two_type_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_copy_wider_than_connectivity() {
        let mut config = two_type_config();
        config.copy_neighborhood = NeighborhoodKind::Moore;
        config.connectivity_neighborhood = NeighborhoodKind::VonNeumann;
        let err = config.validate().unwrap_err();
        assert!(err.0.contains("subset"), "unexpected error: {err}");
    }

    #[test]
    fn validate_rejects_edge_list_proposal_with_metropolis_hastings() {
        let mut config = two_type_config();
        config.proposal = ProposalMode::EdgeList;
        config.acceptance = AcceptanceMode::MetropolisHastings;
        let err = config.validate().unwrap_err();
        assert!(err.0.contains("EdgeList"), "unexpected error: {err}");
    }

    #[test]
    fn validate_accepts_edge_list_proposal_with_metropolis() {
        let mut config = two_type_config();
        config.proposal = ProposalMode::EdgeList;
        config.acceptance = AcceptanceMode::Metropolis;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_wrong_sized_adhesion_matrix() {
        let mut config = two_type_config();
        config.adhesion = vec![vec![0.0, 1.0], vec![1.0, 0.0]]; // 2x2, needs 3x3
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_asymmetric_adhesion_matrix() {
        let mut config = two_type_config();
        config.adhesion[0][1] = 99.0; // now differs from adhesion[1][0]
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_empty_cell_types() {
        let mut config = two_type_config();
        config.cell_types.clear();
        assert!(config.validate().is_err());
    }

    #[test]
    fn convention_hash_is_deterministic() {
        let config = two_type_config();
        assert_eq!(convention_hash(&config), convention_hash(&config));
    }

    #[test]
    fn convention_hash_changes_with_grid() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.grid = [40, 40];
        assert_ne!(convention_hash(&a), convention_hash(&b));
    }

    /// The convention hash must reflect which energy terms are active, so
    /// a run that used a term is distinguishable from one that didn't.
    /// `ConventionView` already carries `active_terms: EnergyTermSet` (the
    /// field predates Act's runtime implementation), so this just confirms
    /// toggling `act` specifically is visible in the hash now that
    /// toggling it means something.
    #[test]
    fn convention_hash_changes_with_act_enabled() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.active_terms.act = true;
        assert_ne!(convention_hash(&a), convention_hash(&b));
    }

    #[test]
    fn convention_hash_changes_with_proposal_mode() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.proposal = ProposalMode::EdgeList;
        assert_ne!(convention_hash(&a), convention_hash(&b));
    }

    #[test]
    fn convention_hash_ignores_seed() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.seed = a.seed + 1;
        assert_eq!(convention_hash(&a), convention_hash(&b));
    }

    /// The critical property: theta-eligible numeric values must never
    /// affect the hash, because every simulation in an SBI training set
    /// shares one convention_hash despite differing theta.
    #[test]
    fn convention_hash_ignores_theta_eligible_values() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.cell_types[0].lambda_volume += 100.0;
        b.cell_types[0].lambda_interface += 100.0;
        b.cell_types[0].lambda_act += 100.0;
        b.cell_types[0].max_act += 100;
        b.adhesion[0][1] += 100.0;
        b.adhesion[1][0] += 100.0;
        assert_eq!(convention_hash(&a), convention_hash(&b));
    }

    #[test]
    fn convention_hash_reflects_type_count_and_names() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.cell_types[0].name = "renamed".into();
        assert_ne!(convention_hash(&a), convention_hash(&b));
    }

    fn placed_state(grid: usize, sites: &[(usize, usize, CellId)]) -> State<2> {
        use crate::cell::Cell;
        use crate::lattice::Lattice;
        let mut lattice = Lattice::new([grid, grid], Boundary::Fixed);
        for &(r, c, id) in sites {
            let flat = lattice.flat_index([r, c]);
            lattice.set(flat, id);
        }
        let max_id = sites.iter().map(|&(_, _, id)| id).max().unwrap_or(0);
        let cells = (1..=max_id).map(|id| Cell { id, type_index: 0 }).collect();
        State::with_cells(lattice, cells)
    }

    #[test]
    fn model_hash_is_deterministic() {
        let config = two_type_config();
        let state = placed_state(30, &[(5, 5, 1)]);
        assert_eq!(model_hash(&config, &state), model_hash(&config, &state));
    }

    /// The property `convention_hash` alone cannot give: two runs under
    /// identical conventions (so identical `convention_hash`, e.g. both
    /// `InitializationSpec::Default`) that nonetheless resolved to different
    /// initial placements — different seeds, or the same seed after an
    /// unrelated scatter-and-grow change — are not the same run, and
    /// `model_hash` must say so even though `convention_hash` cannot.
    #[test]
    fn model_hash_differs_on_placement_alone() {
        let config = two_type_config();
        let a = placed_state(30, &[(5, 5, 1)]);
        let b = placed_state(30, &[(6, 6, 1)]);
        assert_eq!(convention_hash(&config), convention_hash(&config));
        assert_ne!(model_hash(&config, &a), model_hash(&config, &b));
    }

    #[test]
    fn model_hash_changes_with_convention_hash_too() {
        let a = two_type_config();
        let mut b = two_type_config();
        b.grid = [40, 40];
        let state_a = placed_state(30, &[(5, 5, 1)]);
        let state_b = placed_state(40, &[(5, 5, 1)]);
        assert_ne!(model_hash(&a, &state_a), model_hash(&b, &state_b));
    }
}
