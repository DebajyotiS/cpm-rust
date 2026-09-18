//! `CPM<D>`: the dimension-generic engine. Wires together the resolved
//! config, the lattice, the precomputed neighbour offset tables, the energy
//! terms, and the RNG. `CPM::new` validates a config and builds an empty
//! (all-medium) lattice with no cells; `initialization.rs` places cells,
//! `monte_carlo.rs` runs the dynamics via [`CPM::run_mcs`].

use crate::config::{ConfigError, ResolvedConfig};
use crate::connectivity::topology_offset_deltas;
use crate::edge_list::EdgeList;
use crate::energy::Terms;
use crate::lattice::{offset_table, Lattice};
use crate::rng::{rng_from_seed, Rng};
use crate::state::State;

pub struct CPM<const D: usize> {
    pub config: ResolvedConfig<D>,
    pub state: State<D>,
    pub terms: Terms,
    pub rng: Rng,
    /// Elapsed Monte Carlo steps.
    pub mcs: u64,
    /// Precomputed flat-index deltas, one per stencil, built once at
    /// construction from the resolved neighbourhood choices.
    pub copy_offsets: Vec<isize>,
    pub energy_offsets: Vec<isize>,
    pub energy_weights: Vec<f64>,
    pub connectivity_offsets: Vec<isize>,
    /// The fixed canonical full-neighbourhood offsets the simple-point
    /// pattern is built over (`connectivity.rs`) — independent of
    /// `connectivity_neighborhood`'s own stencil.
    pub topology_offsets: Vec<isize>,
    /// `Some` only under `ProposalMode::EdgeList`, `None` under the default
    /// `Uniform` (so `Uniform` runs pay no maintenance cost for a set they
    /// never consult). `CPM::new` always starts it at `None` — there are no
    /// cells yet, so no edges either; `initialization.rs` builds the real
    /// set once cells are placed, only if the config asked for `EdgeList`.
    pub edge_list: Option<EdgeList>,
}

impl<const D: usize> CPM<D> {
    /// Validate the config, build an empty lattice sized from `config.grid`,
    /// and precompute every offset table and energy term. Returns `Result`;
    /// never panics on user input.
    pub fn new(config: ResolvedConfig<D>) -> Result<Self, ConfigError> {
        config.validate()?;

        let lattice = Lattice::new(config.grid, config.boundary);
        let strides = lattice.strides();

        let copy_stencil = config.copy_neighborhood.to_stencil::<D>()?;
        let energy_stencil = config.energy_neighborhood.to_stencil::<D>()?;
        let connectivity_stencil = config.connectivity_neighborhood.to_stencil::<D>()?;

        let copy_offsets = offset_table(&copy_stencil, strides);
        let energy_offsets = offset_table(&energy_stencil, strides);
        let energy_weights = energy_stencil.weights.clone();
        let connectivity_offsets = offset_table(&connectivity_stencil, strides);
        let topology_offsets = topology_offset_deltas(strides);

        let terms = Terms::from_config(&config, &energy_offsets, &energy_weights, &copy_offsets);
        let rng = rng_from_seed(config.seed);

        Ok(Self {
            state: State::empty(lattice),
            terms,
            rng,
            mcs: 0,
            copy_offsets,
            energy_offsets,
            energy_weights,
            connectivity_offsets,
            edge_list: None,
            topology_offsets,
            config,
        })
    }

    /// `N_sites` in the MCS definition: the unpadded site count.
    pub fn n_sites(&self) -> usize {
        self.state.lattice.n_sites()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::test_support::two_type_config;
    use crate::neighborhood::NeighborhoodKind;

    #[test]
    fn new_succeeds_on_a_valid_config() {
        let config = two_type_config();
        let cpm = CPM::new(config).unwrap();
        assert_eq!(cpm.n_sites(), 30 * 30);
        assert_eq!(cpm.state.n_cells(), 0); // CPM::new builds the lattice only; cells are placed separately
    }

    #[test]
    fn offset_tables_match_stencil_sizes() {
        let cpm = CPM::new(two_type_config()).unwrap();
        assert_eq!(cpm.copy_offsets.len(), 4); // von Neumann, 2D
        assert_eq!(cpm.energy_offsets.len(), 8); // Moore, 2D default
        assert_eq!(cpm.energy_weights.len(), 8);
        assert_eq!(cpm.connectivity_offsets.len(), 4); // von Neumann, 2D
    }

    #[test]
    fn new_propagates_config_validation_errors() {
        let mut config = two_type_config();
        config.copy_neighborhood = NeighborhoodKind::Moore;
        config.connectivity_neighborhood = NeighborhoodKind::VonNeumann;
        assert!(CPM::new(config).is_err());
    }
}
