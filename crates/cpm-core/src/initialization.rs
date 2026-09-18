//! Cell placement: explicit (a fully-specified initial lattice) or the
//! default scatter-and-grow. Both are deterministic given the seed and both
//! leave `cpm.state` with a validated, connected set of cells.
//!
//! Emitting the actual `CPMInitializationWarning` is Python-side
//! (`python/cpm/warnings.py`) — there is no Python-callable `CPM(...)`
//! constructor yet to warn from; that arrives with the future builder API.
//! What this module provides is [`Outcome::used_default`], the flag that
//! constructor will eventually check before calling `warnings.warn`.

use crate::cell::Cell;
use crate::config::InitializationSpec;
use crate::lattice::{Boundary, CellId};
use crate::model::CPM;
use crate::state::{index_of, State};
use rand::Rng as _;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct InitError(pub String);

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for InitError {}

pub struct Outcome {
    /// Whether the default scatter-and-grow path was taken (as opposed to
    /// an explicitly-supplied lattice) — the caller's signal for whether
    /// `CPMInitializationWarning` should fire.
    pub used_default: bool,
}

/// Place cells into `cpm.state` per `cpm.config.initialization`, replacing
/// whatever (empty) state `CPM::new` left behind.
pub fn initialize<const D: usize>(cpm: &mut CPM<D>) -> Result<Outcome, InitError> {
    match cpm.config.initialization.clone() {
        InitializationSpec::Default { .. } => {
            scatter_and_grow(cpm)?;
            Ok(Outcome { used_default: true })
        }
        InitializationSpec::Explicit {
            lattice,
            cell_type_of,
        } => {
            place_explicit(cpm, &lattice, &cell_type_of)?;
            Ok(Outcome {
                used_default: false,
            })
        }
    }
}

// ---------------------------------------------------------------------
// Explicit placement
// ---------------------------------------------------------------------

fn place_explicit<const D: usize>(
    cpm: &mut CPM<D>,
    lattice_values: &[CellId],
    cell_type_of: &[usize],
) -> Result<(), InitError> {
    let n_sites = cpm.n_sites();
    if lattice_values.len() != n_sites {
        return Err(InitError(format!(
            "explicit lattice has {} entries, expected N_sites = {n_sites}",
            lattice_values.len()
        )));
    }
    let n_cells = cell_type_of.len();
    let n_types = cpm.config.cell_types.len();
    for (i, &type_index) in cell_type_of.iter().enumerate() {
        if type_index >= n_types {
            return Err(InitError(format!(
                "cell {} has type_index {type_index}, but only {n_types} types are declared",
                i + 1
            )));
        }
    }
    for &id in lattice_values {
        if id as usize > n_cells {
            return Err(InitError(format!(
                "lattice references cell id {id}, but cell_type_of only covers {n_cells} cells"
            )));
        }
    }

    let mut lattice = crate::lattice::Lattice::<D>::new(cpm.config.grid, cpm.config.boundary);
    let mut coords = Vec::with_capacity(n_sites);
    lattice.each_interior_coord(|c| coords.push(c));
    debug_assert_eq!(coords.len(), n_sites);
    let mut volume = vec![0u32; n_cells];
    for (&value, &coord) in lattice_values.iter().zip(coords.iter()) {
        let flat = lattice.flat_index(coord);
        lattice.set(flat, value);
        if value != 0 {
            volume[index_of(value)] += 1;
        }
    }
    if lattice.boundary() == Boundary::Periodic {
        lattice.refresh_periodic_halo();
    }

    for (i, &v) in volume.iter().enumerate() {
        if v == 0 {
            return Err(InitError(format!(
                "cell {} has zero volume in the supplied lattice",
                i + 1
            )));
        }
    }

    let cells: Vec<Cell> = (1..=n_cells as CellId)
        .map(|id| Cell {
            id,
            type_index: cell_type_of[index_of(id)],
        })
        .collect();

    for id in 1..=n_cells as CellId {
        if !is_connected(&lattice, id, &cpm.connectivity_offsets) {
            return Err(InitError(format!(
                "cell {id} is not connected under connectivity_neighborhood — \
                 every explicitly-placed cell must already be connected"
            )));
        }
    }

    cpm.state = State::with_cells(lattice, cells);
    cpm.state.volume = volume;
    recompute_interface_and_positions(cpm);
    cpm.state.conservative_energy = cpm.terms.global_energy(&cpm.state);
    Ok(())
}

/// BFS connectivity check under `connectivity_neighborhood` (always von
/// Neumann, per `config.rs`) — used only here, at initialisation; the hot
/// loop never needs a global connectivity check, since a global flood fill
/// has no place in the Monte Carlo hot loop.
fn is_connected<const D: usize>(
    lattice: &crate::lattice::Lattice<D>,
    id: CellId,
    connectivity_offsets: &[isize],
) -> bool {
    let n_sites = lattice.n_sites();
    let mut sites = Vec::new();
    lattice.each_interior_coord(|coord| {
        let flat = lattice.flat_index(coord);
        if lattice.get(flat) == id {
            sites.push(flat);
        }
    });
    if sites.is_empty() {
        return false;
    }
    let mut visited = std::collections::HashSet::with_capacity(sites.len().min(n_sites));
    let mut stack = vec![sites[0]];
    visited.insert(sites[0]);
    while let Some(flat) = stack.pop() {
        for &offset in connectivity_offsets {
            // Canonicalise to the interior representative before visiting:
            // under a periodic boundary, a halo neighbour just *mirrors*
            // some other interior site (refreshed by
            // `refresh_periodic_halo`), it isn't a distinct BFS node. If
            // treated as one, expanding from it one more step (offset
            // applied a second time) walks straight through the one-cell-
            // thick halo and out of the padded array entirely.
            let neighbour = canonical_neighbour(lattice, flat, offset);
            if lattice.get(neighbour) == id && visited.insert(neighbour) {
                stack.push(neighbour);
            }
        }
    }
    visited.len() == sites.len()
}

/// The canonical *interior* flat index of the neighbour reached via
/// `offset` from `flat`. Under `Fixed`, a halo neighbour genuinely is a
/// distinct (permanently-medium) position, so it's returned as-is. Under
/// `Periodic`, it's mapped to the interior site it mirrors — see
/// [`crate::lattice::Lattice::wrapped_interior_flat`].
fn canonical_neighbour<const D: usize>(
    lattice: &crate::lattice::Lattice<D>,
    flat: usize,
    offset: isize,
) -> usize {
    let neighbour = lattice.neighbour(flat, offset);
    if lattice.is_interior(neighbour) || lattice.boundary() != Boundary::Periodic {
        neighbour
    } else {
        lattice.wrapped_interior_flat(neighbour)
    }
}

// ---------------------------------------------------------------------
// Default: scatter and grow
// ---------------------------------------------------------------------

fn scatter_and_grow<const D: usize>(cpm: &mut CPM<D>) -> Result<(), InitError> {
    let mut type_of_cell: Vec<usize> = Vec::new();
    for (type_index, &count) in cpm.config.cell_counts.iter().enumerate() {
        type_of_cell.extend(std::iter::repeat_n(type_index, count));
    }
    let n_cells = type_of_cell.len();
    if n_cells == 0 {
        return Err(InitError(
            "cell_counts sums to zero cells — nothing to place".into(),
        ));
    }

    let dims = cpm.config.grid;
    let boundary = cpm.config.boundary;

    // Minimum seed separation: a rough packing radius from the average
    // target volume, halved so modestly dense configurations still have
    // room to succeed — this is a placement heuristic, not a modelling
    // convention, and does not need to be exact.
    let avg_target_volume: f64 = cpm
        .config
        .cell_types
        .iter()
        .map(|t| t.target_volume as f64)
        .sum::<f64>()
        / cpm.config.cell_types.len() as f64;
    let min_separation = (avg_target_volume / std::f64::consts::PI).sqrt();

    let mut seeds: Vec<[usize; D]> = Vec::with_capacity(n_cells);
    const MAX_ATTEMPTS_PER_SEED: usize = 20_000;
    for _ in 0..n_cells {
        let mut placed = false;
        for _ in 0..MAX_ATTEMPTS_PER_SEED {
            let candidate: [usize; D] = std::array::from_fn(|d| cpm.rng.gen_range(0..dims[d]));
            let far_enough = seeds
                .iter()
                .all(|s| periodic_distance(s, &candidate, dims, boundary) >= min_separation);
            if far_enough {
                seeds.push(candidate);
                placed = true;
                break;
            }
        }
        if !placed {
            return Err(InitError(format!(
                "could not place all {n_cells} seed points with minimum separation \
                 {min_separation:.2} on a grid of {:?} — too crowded for this configuration",
                dims
            )));
        }
    }

    let mut lattice = crate::lattice::Lattice::<D>::new(dims, boundary);
    let mut volume = vec![0u32; n_cells];
    for (i, &coord) in seeds.iter().enumerate() {
        let id = (i + 1) as CellId;
        let flat = lattice.flat_index(coord);
        lattice.set(flat, id);
        volume[i] += 1;
    }
    if boundary == Boundary::Periodic {
        lattice.refresh_periodic_halo();
    }

    // Grow each cell outward, claiming an adjacent medium site at a time,
    // round-robin across cells so no single cell races ahead. A cell grown
    // by repeatedly annexing a copy-neighbourhood-adjacent medium site
    // stays connected by construction — no simple-point check is needed
    // here (medium, the losing side, is exempt regardless).
    let target_volume: Vec<u32> = type_of_cell
        .iter()
        .map(|&t| cpm.config.cell_types[t].target_volume)
        .collect();
    let mut frontier: Vec<Vec<usize>> = (0..n_cells)
        .map(|i| frontier_of(&lattice, lattice.flat_index(seeds[i]), &cpm.copy_offsets))
        .collect();

    let mut order: Vec<usize> = (0..n_cells).collect();
    loop {
        let mut grew_any = false;
        for &i in &order {
            if volume[i] >= target_volume[i] {
                continue;
            }
            frontier[i].retain(|&flat| lattice.get(flat) == 0);
            if frontier[i].is_empty() {
                continue; // collided or boxed in; this cell stops growing
            }
            let pick = cpm.rng.gen_range(0..frontier[i].len());
            let claimed = frontier[i].swap_remove(pick);
            let id = (i + 1) as CellId;
            lattice.set(claimed, id);
            volume[i] += 1;
            if lattice.boundary() == Boundary::Periodic {
                lattice.refresh_periodic_halo();
            }
            for &offset in &cpm.copy_offsets {
                if let Some(neighbour) = canonical_growable_neighbour(&lattice, claimed, offset) {
                    if lattice.get(neighbour) == 0 {
                        frontier[i].push(neighbour);
                    }
                }
            }
            grew_any = true;
        }
        if !grew_any {
            break;
        }
        // Deterministic shuffle each round so growth direction doesn't
        // systematically favour low cell indices.
        for i in (1..order.len()).rev() {
            let j = cpm.rng.gen_range(0..=i);
            order.swap(i, j);
        }
    }

    let cells: Vec<Cell> = (1..=n_cells as CellId)
        .map(|id| Cell {
            id,
            type_index: type_of_cell[index_of(id)],
        })
        .collect();

    cpm.state = State::with_cells(lattice, cells);
    cpm.state.volume = volume;
    recompute_interface_and_positions(cpm);
    cpm.state.conservative_energy = cpm.terms.global_energy(&cpm.state);
    Ok(())
}

fn frontier_of<const D: usize>(
    lattice: &crate::lattice::Lattice<D>,
    seed_flat: usize,
    copy_offsets: &[isize],
) -> Vec<usize> {
    copy_offsets
        .iter()
        .filter_map(|&offset| canonical_growable_neighbour(lattice, seed_flat, offset))
        .filter(|&flat| lattice.get(flat) == 0)
        .collect()
}

/// The canonical *interior* flat index of the neighbour reached via
/// `offset` from `flat`, or `None` if that neighbour isn't a legitimate
/// growable site. Under `Fixed`, a halo neighbour is the permanent wall —
/// never growable. Under `Periodic`, a halo neighbour is only a *view* onto
/// some other interior site (refreshed by
/// [`crate::lattice::Lattice::refresh_periodic_halo`]); growth must claim
/// that interior site directly, never the raw halo index — the halo is
/// only one cell thick, so treating it as an independent claimable site
/// would walk straight out of the padded array on the very next step.
fn canonical_growable_neighbour<const D: usize>(
    lattice: &crate::lattice::Lattice<D>,
    flat: usize,
    offset: isize,
) -> Option<usize> {
    let raw = lattice.neighbour(flat, offset);
    if lattice.is_interior(raw) || lattice.boundary() == Boundary::Periodic {
        Some(canonical_neighbour(lattice, flat, offset))
    } else {
        None // Fixed boundary halo: the permanent wall, never growable.
    }
}

/// Nearest-image Euclidean distance between two interior coordinates —
/// wrapping under `Periodic`, plain otherwise. Only used for the
/// scatter-seed separation heuristic above.
fn periodic_distance<const D: usize>(
    a: &[usize; D],
    b: &[usize; D],
    dims: [usize; D],
    boundary: Boundary,
) -> f64 {
    let mut sum_sq = 0.0;
    for d in 0..D {
        let raw = (a[d] as f64 - b[d] as f64).abs();
        let delta = if boundary == Boundary::Periodic {
            raw.min(dims[d] as f64 - raw)
        } else {
            raw
        };
        sum_sq += delta * delta;
    }
    sum_sq.sqrt()
}

/// Establishes every derived-from-the-lattice piece of `State` a fresh
/// placement needs: `interface`, unwrapped `sum_position`, the
/// gyration-tensor accumulator `sum_position_outer`, and the contact graph.
/// The contact graph specifically needs establishing here because a fresh
/// placement never went through `dynamics::commit`'s incremental updates, so
/// nothing else would ever populate it — `crate::state::recompute_contact`
/// is the same from-scratch definition the brute-force checker also uses to
/// verify incremental maintenance hasn't drifted.
fn recompute_interface_and_positions<const D: usize>(cpm: &mut CPM<D>) {
    let n_cells = cpm.state.cells.len();
    let mut interface = vec![0u32; n_cells];
    let mut sum_position = vec![[0i64; D]; n_cells];
    let mut sum_position_outer = vec![[[0i64; D]; D]; n_cells];
    let lattice = cpm.state.lattice.clone();
    lattice.each_interior_coord(|coord| {
        let flat = lattice.flat_index(coord);
        let cell = lattice.get(flat);
        if cell == 0 {
            return;
        }
        let i = index_of(cell);
        for d in 0..D {
            sum_position[i][d] += coord[d] as i64;
            for e in 0..D {
                sum_position_outer[i][d][e] += coord[d] as i64 * coord[e] as i64;
            }
        }
        for &offset in &cpm.energy_offsets {
            let neighbour = lattice.get(lattice.neighbour(flat, offset));
            if neighbour != cell {
                interface[i] += 1;
            }
        }
    });
    cpm.state.interface = interface;
    cpm.state.sum_position = sum_position;
    cpm.state.sum_position_outer = sum_position_outer;
    cpm.state.contact =
        crate::state::recompute_contact(&lattice, &cpm.energy_offsets, cpm.state.cells.len());
    // Only built under `ProposalMode::EdgeList` — a `Uniform` run never
    // consults it, so it stays `None` rather than paying for a set nothing
    // reads.
    if cpm.config.proposal == crate::config::ProposalMode::EdgeList {
        cpm.edge_list = Some(crate::edge_list::EdgeList::build(
            &lattice,
            &cpm.copy_offsets,
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CellType;
    use crate::checker;
    use crate::config::UserConfig;

    fn config(seed: u64) -> crate::config::ResolvedConfig<2> {
        UserConfig::<2> {
            grid: Some([30, 30]),
            boundary: Some(Boundary::Periodic),
            cell_types: vec![
                CellType {
                    name: "a".into(),
                    target_volume: 10,
                    target_interface: 24,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 0.0,
                    max_act: 0,
                },
                CellType {
                    name: "b".into(),
                    target_volume: 10,
                    target_interface: 24,
                    lambda_volume: 1.0,
                    lambda_interface: 0.2,
                    lambda_act: 0.0,
                    max_act: 0,
                },
            ],
            cell_counts: vec![4, 4],
            adhesion: Some(vec![
                vec![0.0, 2.0, 2.0],
                vec![2.0, 1.0, 3.0],
                vec![2.0, 3.0, 1.0],
            ]),
            burn_in_mcs: Some(0),
            readout_mcs: Some(1),
            sampling_interval_mcs: Some(1),
            seed: Some(seed),
            ..Default::default()
        }
        .resolve()
        .unwrap()
    }

    #[test]
    fn scatter_and_grow_reaches_target_volumes_and_stays_consistent() {
        let mut cpm = CPM::new(config(11)).unwrap();
        let outcome = initialize(&mut cpm).unwrap();
        assert!(outcome.used_default);
        assert_eq!(cpm.state.n_cells(), 8);
        for (i, cell) in cpm.state.cells.iter().enumerate() {
            let target = cpm.config.cell_types[cell.type_index].target_volume;
            assert_eq!(
                cpm.state.volume[i], target,
                "cell {} should have reached its target volume on an uncrowded grid",
                cell.id
            );
            assert!(is_connected(
                &cpm.state.lattice,
                cell.id,
                &cpm.connectivity_offsets
            ));
        }
        checker::check(&cpm).expect("freshly-initialised state must be internally consistent");
    }

    #[test]
    fn scatter_and_grow_is_deterministic_given_the_seed() {
        let mut a = CPM::new(config(99)).unwrap();
        let mut b = CPM::new(config(99)).unwrap();
        initialize(&mut a).unwrap();
        initialize(&mut b).unwrap();
        assert_eq!(a.state.volume, b.state.volume);
        for coord0 in 0..30usize {
            for coord1 in 0..30usize {
                let flat_a = a.state.lattice.flat_index([coord0, coord1]);
                let flat_b = b.state.lattice.flat_index([coord0, coord1]);
                assert_eq!(a.state.lattice.get(flat_a), b.state.lattice.get(flat_b));
            }
        }
    }

    #[test]
    fn different_seeds_scatter_differently() {
        let mut a = CPM::new(config(1)).unwrap();
        let mut b = CPM::new(config(2)).unwrap();
        initialize(&mut a).unwrap();
        initialize(&mut b).unwrap();
        assert_ne!(a.state.sum_position, b.state.sum_position);
    }

    #[test]
    fn explicit_placement_round_trips() {
        let mut cpm = CPM::new(config(5)).unwrap();
        // Build a tiny hand-made lattice: two 2x2 blocks, well separated.
        let mut lattice_values = vec![0 as CellId; cpm.n_sites()];
        let mut set = |x: usize, y: usize, id: CellId| {
            lattice_values[x * 30 + y] = id;
        };
        for x in 1..3 {
            for y in 1..3 {
                set(x, y, 1);
            }
        }
        for x in 10..12 {
            for y in 10..12 {
                set(x, y, 2);
            }
        }
        let cell_type_of = vec![0, 1];
        place_explicit(&mut cpm, &lattice_values, &cell_type_of).unwrap();

        assert_eq!(cpm.state.n_cells(), 2);
        assert_eq!(cpm.state.volume, vec![4, 4]);
        checker::check(&cpm).expect("explicitly-placed state must be internally consistent");
    }

    #[test]
    fn explicit_placement_rejects_disconnected_cell() {
        let mut cpm = CPM::new(config(5)).unwrap();
        let mut lattice_values = vec![0 as CellId; cpm.n_sites()];
        // Two disjoint single sites sharing the same id: not connected.
        lattice_values[30 + 1] = 1;
        lattice_values[20 * 30 + 20] = 1;
        lattice_values[5 * 30 + 5] = 2; // second declared cell, present and connected
        let cell_type_of = vec![0, 1];
        assert!(place_explicit(&mut cpm, &lattice_values, &cell_type_of).is_err());
    }

    #[test]
    fn explicit_placement_rejects_zero_volume_cell() {
        let mut cpm = CPM::new(config(5)).unwrap();
        let mut lattice_values = vec![0 as CellId; cpm.n_sites()];
        lattice_values[5 * 30 + 5] = 1;
        // cell 2 declared but never placed.
        let cell_type_of = vec![0, 1];
        assert!(place_explicit(&mut cpm, &lattice_values, &cell_type_of).is_err());
    }

    #[test]
    fn explicit_placement_rejects_wrong_length() {
        let mut cpm = CPM::new(config(5)).unwrap();
        let cell_type_of = vec![0];
        assert!(place_explicit(&mut cpm, &[0, 1, 0], &cell_type_of).is_err());
    }
}
