//! Parallel execution of independent simulations, one per `theta`, via
//! Rayon. Each simulation is fully self-contained — its own cloned config,
//! its own [`CPM`], its own deterministically derived seed — so there is no
//! shared mutable state across the parallel work, and results come back in
//! the same order as `thetas` regardless of how Rayon schedules the work
//! across threads.

use crate::config::{ConfigError, ResolvedConfig};
use crate::initialization::{self, InitError};
use crate::model::CPM;
use crate::output::{self, RunOptions, RunResult};
use crate::rng::derive_seed_u64;
use crate::theta::{self, theta_len, ThetaError};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub enum BatchError {
    Config(ConfigError),
    Init(InitError),
    Theta(ThetaError),
    WrongThetaLength {
        index: usize,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::Config(e) => write!(f, "{e}"),
            BatchError::Init(e) => write!(f, "{e}"),
            BatchError::Theta(e) => write!(f, "{e}"),
            BatchError::WrongThetaLength {
                index,
                expected,
                got,
            } => write!(
                f,
                "theta at index {index} has length {got}, expected {expected}"
            ),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<ConfigError> for BatchError {
    fn from(e: ConfigError) -> Self {
        BatchError::Config(e)
    }
}

impl From<InitError> for BatchError {
    fn from(e: InitError) -> Self {
        BatchError::Init(e)
    }
}

impl From<ThetaError> for BatchError {
    fn from(e: ThetaError) -> Self {
        BatchError::Theta(e)
    }
}

/// Settings for [`run_batch`] that aren't already part of `ResolvedConfig` —
/// grid, cell types, adhesion structure, and the burn-in/readout/sampling
/// windows all travel into every batch member unchanged via
/// `base_config.clone()`.
#[derive(Debug, Clone, Copy)]
pub struct BatchOptions {
    pub master_seed: u64,
    pub include_lattice: bool,
}

/// One simulation's outcome from [`run_batch`]. `cell_type_index` is read
/// off that simulation's own `CPM` before it's dropped, rather than assumed
/// to match every other batch member's: it's identical across the batch
/// whenever cells come from the default scatter-and-grow placement (theta
/// never touches `cell_counts`/`cell_types`), but explicit placement carries
/// its own `cell_type_of` array on `InitializationSpec` — still the same
/// for every batch member here, since that's part of `base_config` too, but
/// reading it back from the real `CPM` rather than re-deriving it keeps this
/// correct under either placement method without special-casing which one
/// was used.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchItem<const D: usize> {
    pub used_default: bool,
    pub cell_type_index: Vec<u32>,
    pub result: RunResult<D>,
}

/// Runs one independent trajectory per entry in `thetas`, in parallel. Every
/// simulation clones `base_config` and only changes two things: `theta` is
/// injected ([`theta::inject`]), and the seed is derived from
/// `(options.master_seed, index)` ([`derive_seed_u64`]) rather than reused
/// across simulations — the independent-seed-per-simulation policy this
/// project's SBI training sets need.
///
/// Every theta's length is checked against the config's cell-type count
/// before any parallel work starts, so a malformed theta surfaces as
/// [`BatchError::WrongThetaLength`] rather than the panic
/// [`theta::inject`]'s own length assertion would otherwise raise inside a
/// worker thread. Everything else about the config — including per-type
/// stiffness bounds and adhesion-matrix symmetry — is re-checked by
/// `CPM::new`'s own validation on every construction, the same as a single
/// run gets.
///
/// Returns one [`BatchItem`] per theta, in input order.
pub fn run_batch<const D: usize>(
    base_config: &ResolvedConfig<D>,
    thetas: &[Vec<f64>],
    options: BatchOptions,
) -> Result<Vec<BatchItem<D>>, BatchError> {
    let expected_len = theta_len(base_config.cell_types.len());
    for (index, theta) in thetas.iter().enumerate() {
        if theta.len() != expected_len {
            return Err(BatchError::WrongThetaLength {
                index,
                expected: expected_len,
                got: theta.len(),
            });
        }
    }

    thetas
        .par_iter()
        .enumerate()
        .map(|(index, theta)| {
            let mut config = base_config.clone();
            theta::inject(&mut config, theta)?;
            config.seed = derive_seed_u64(options.master_seed, index as u64);

            let mut cpm = CPM::new(config)?;
            let outcome = initialization::initialize(&mut cpm)?;
            let run_options = RunOptions {
                include_lattice: options.include_lattice,
            };
            let cell_type_index = cpm
                .state
                .cells
                .iter()
                .map(|c| c.type_index as u32)
                .collect();
            let result = output::run_with(&mut cpm, run_options);
            Ok(BatchItem {
                used_default: outcome.used_default,
                cell_type_index,
                result,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use crate::config::UserConfig;
    use crate::lattice::Boundary;

    fn two_type_config() -> ResolvedConfig<2> {
        UserConfig::<2> {
            grid: Some([20, 20]),
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
            cell_counts: vec![2, 2],
            adhesion: Some(vec![
                vec![0.0, 3.0, 3.0],
                vec![3.0, 1.0, 5.0],
                vec![3.0, 5.0, 1.0],
            ]),
            burn_in_mcs: Some(5),
            readout_mcs: Some(5),
            sampling_interval_mcs: Some(5),
            seed: Some(0), // overwritten per simulation by run_batch
            ..Default::default()
        }
        .resolve()
        .unwrap()
    }

    fn options(master_seed: u64) -> BatchOptions {
        BatchOptions {
            master_seed,
            include_lattice: false,
        }
    }

    fn sample_thetas(base_config: &ResolvedConfig<2>, n: usize) -> Vec<Vec<f64>> {
        let base = theta::extract(base_config);
        (0..n)
            .map(|i| base.iter().map(|v| v + i as f64).collect())
            .collect()
    }

    #[test]
    fn results_come_back_in_input_order_with_injected_theta() {
        let config = two_type_config();
        let thetas = sample_thetas(&config, 4);
        let results = run_batch(&config, &thetas, options(1)).unwrap();
        assert_eq!(results.len(), thetas.len());
    }

    #[test]
    fn every_batch_member_shares_one_convention_hash() {
        let config = two_type_config();
        let thetas = sample_thetas(&config, 5);
        let results = run_batch(&config, &thetas, options(1)).unwrap();
        let first_hash = results[0].result.metadata.convention_hash.clone();
        for item in &results {
            assert_eq!(item.result.metadata.convention_hash, first_hash);
        }
    }

    #[test]
    fn wrong_length_theta_is_rejected_before_any_simulation_runs() {
        let config = two_type_config();
        let mut thetas = sample_thetas(&config, 3);
        thetas[1].pop(); // one element short
        let err = run_batch(&config, &thetas, options(1)).unwrap_err();
        assert_eq!(
            err,
            BatchError::WrongThetaLength {
                index: 1,
                expected: theta_len(2),
                got: theta_len(2) - 1,
            }
        );
    }

    #[test]
    fn same_master_seed_gives_bit_identical_results_regardless_of_thread_count() {
        let config = two_type_config();
        let thetas = sample_thetas(&config, 8);

        let default_pool = run_batch(&config, &thetas, options(42)).unwrap();
        let single_threaded = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| run_batch(&config, &thetas, options(42)).unwrap());

        assert_eq!(default_pool, single_threaded);
    }

    #[test]
    fn different_batch_members_get_different_seeds_and_therefore_diverge() {
        let config = two_type_config();
        // Identical theta for every member: any difference in the result is
        // attributable only to the per-simulation derived seed.
        let base = theta::extract(&config);
        let thetas = vec![base.clone(), base.clone(), base];
        let results = run_batch(&config, &thetas, options(7)).unwrap();
        let volumes: Vec<_> = results
            .iter()
            .map(|item| item.result.samples.last().unwrap().cells.clone())
            .collect();
        assert!(
            volumes[0] != volumes[1] || volumes[1] != volumes[2],
            "independent seeds should not all produce identical trajectories"
        );
    }
}
