//! Diagnostic binary, run manually rather than as part of the test suite,
//! that investigates a mismatch in the tail of the exact-Boltzmann two-cell
//! validation.

use cpm_core::cell::{Cell, CellType};
use cpm_core::config::{AcceptanceMode, UserConfig};
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::attempt;
use cpm_core::state::State;
use std::collections::HashSet;

const GRID2: usize = 4;
const SITES2: usize = GRID2 * GRID2;

struct TwoCellParams {
    lambda_v: [f64; 2],
    v_star: [f64; 2],
    lambda_i: [f64; 2],
    i_star: [f64; 2],
    adhesion: [[f64; 3]; 3],
}

fn moore_neighbours_4x4() -> [[Option<usize>; 8]; SITES2] {
    const OFFSETS: [(i32, i32); 8] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ];
    let mut table = [[None; 8]; SITES2];
    for r in 0..GRID2 as i32 {
        for c in 0..GRID2 as i32 {
            let idx = (r * GRID2 as i32 + c) as usize;
            for (k, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let (nr, nc) = (r + dr, c + dc);
                if (0..GRID2 as i32).contains(&nr) && (0..GRID2 as i32).contains(&nc) {
                    table[idx][k] = Some((nr * GRID2 as i32 + nc) as usize);
                }
            }
        }
    }
    table
}

fn von_neumann_neighbours_4x4() -> [[Option<usize>; 4]; SITES2] {
    const OFFSETS: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    let mut table = [[None; 4]; SITES2];
    for r in 0..GRID2 as i32 {
        for c in 0..GRID2 as i32 {
            let idx = (r * GRID2 as i32 + c) as usize;
            for (k, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let (nr, nc) = (r + dr, c + dc);
                if (0..GRID2 as i32).contains(&nr) && (0..GRID2 as i32).contains(&nc) {
                    table[idx][k] = Some((nr * GRID2 as i32 + nc) as usize);
                }
            }
        }
    }
    table
}

fn h_two_cell(
    digits: &[u8; SITES2],
    neighbours: &[[Option<usize>; 8]; SITES2],
    p: &TwoCellParams,
) -> f64 {
    let mut volume = [0u32; 2];
    let mut interface = [0u32; 2];
    let mut adhesion_energy = 0.0;
    for site in 0..SITES2 {
        let owner = digits[site];
        if owner != 0 {
            volume[owner as usize - 1] += 1;
        }
        for slot in neighbours[site] {
            let neighbour_owner = match slot {
                Some(n) => digits[n],
                None => 0,
            };
            if neighbour_owner != owner {
                if owner != 0 {
                    interface[owner as usize - 1] += 1;
                }
                adhesion_energy += p.adhesion[owner as usize][neighbour_owner as usize];
            }
        }
    }
    let mut out_of_grid = 0.0;
    for site in 0..SITES2 {
        let owner = digits[site];
        for slot in neighbours[site] {
            if slot.is_none() && owner != 0 {
                out_of_grid += p.adhesion[owner as usize][0];
            }
        }
    }
    let adhesion_energy = adhesion_energy / 2.0 + out_of_grid / 2.0;
    let mut h = adhesion_energy;
    for c in 0..2 {
        h += p.lambda_v[c] * (volume[c] as f64 - p.v_star[c]).powi(2);
        h += p.lambda_i[c] * (interface[c] as f64 - p.i_star[c]).powi(2);
    }
    h
}

fn digits_of(mut pattern: u32) -> [u8; SITES2] {
    let mut digits = [0u8; SITES2];
    for slot in digits.iter_mut() {
        *slot = (pattern % 3) as u8;
        pattern /= 3;
    }
    digits
}

fn pattern_of(digits: &[u8; SITES2]) -> u32 {
    let mut pattern = 0u32;
    for &d in digits.iter().rev() {
        pattern = pattern * 3 + d as u32;
    }
    pattern
}

fn print_grid(digits: &[u8; SITES2]) {
    for r in 0..GRID2 {
        let row: String = (0..GRID2)
            .map(|c| match digits[r * GRID2 + c] {
                0 => '.',
                1 => '1',
                2 => '2',
                _ => '?',
            })
            .collect();
        println!("  {row}");
    }
}

fn build_cpm(params: &TwoCellParams, seed: u64) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([GRID2, GRID2]),
        boundary: Some(Boundary::Fixed),
        acceptance: Some(AcceptanceMode::MetropolisHastings),
        cell_types: vec![
            CellType {
                name: "a".into(),
                target_volume: params.v_star[0].round() as u32,
                target_interface: params.i_star[0].round() as u32,
                lambda_volume: params.lambda_v[0],
                lambda_interface: params.lambda_i[0],
                lambda_act: 0.0,
                max_act: 0,
            },
            CellType {
                name: "b".into(),
                target_volume: params.v_star[1].round() as u32,
                target_interface: params.i_star[1].round() as u32,
                lambda_volume: params.lambda_v[1],
                lambda_interface: params.lambda_i[1],
                lambda_act: 0.0,
                max_act: 0,
            },
        ],
        cell_counts: vec![1, 1],
        adhesion: Some(params.adhesion.iter().map(|row| row.to_vec()).collect()),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    CPM::new(config).unwrap()
}

fn load_pattern(cpm: &mut CPM<2>, digits: &[u8; SITES2]) {
    let mut lattice = cpm.state.lattice.clone();
    let mut volume = [0u32; 2];
    for (site, &owner) in digits.iter().enumerate() {
        if owner != 0 {
            let flat = lattice.flat_index([site / GRID2, site % GRID2]);
            lattice.set(flat, owner as cpm_core::lattice::CellId);
            volume[owner as usize - 1] += 1;
        }
    }
    cpm.state = State::with_cells(
        lattice,
        vec![
            Cell {
                id: 1,
                type_index: 0,
            },
            Cell {
                id: 2,
                type_index: 1,
            },
        ],
    );
    cpm.state.volume = volume.to_vec();
    let mut interface = [0u32; 2];
    let neighbours = moore_neighbours_4x4();
    for (site, &owner) in digits.iter().enumerate() {
        if owner == 0 {
            continue;
        }
        for slot in neighbours[site] {
            let neighbour_owner = match slot {
                Some(n) => digits[n],
                None => 0,
            };
            if neighbour_owner != owner {
                interface[owner as usize - 1] += 1;
            }
        }
    }
    cpm.state.interface = interface.to_vec();
}

fn main() {
    let params = TwoCellParams {
        lambda_v: [0.05, 0.05],
        v_star: [5.0, 5.0],
        lambda_i: [0.01, 0.01],
        i_star: [14.0, 14.0],
        adhesion: [[0.0, 0.5, 0.5], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2]],
    };
    let neighbours = moore_neighbours_4x4();
    let vn = von_neumann_neighbours_4x4();

    // ---- Check #4: does production H match h_two_cell on states actually
    // in the mismatched bins (not just random patterns)? ----
    let target_hs = [42.5f64, 43.0, 42.0, 38.8, 39.3];
    let total = 3u32.pow(SITES2 as u32);
    println!("=== Check #4: production vs independent H on target-bin states ===");
    for &target_h in &target_hs {
        let mut examples = Vec::new();
        for p in 0..total {
            let digits = digits_of(p);
            if !digits.contains(&1) || !digits.contains(&2) {
                continue;
            }
            let h = h_two_cell(&digits, &neighbours, &params);
            if (h - target_h).abs() < 1e-9 {
                examples.push(digits);
                if examples.len() >= 3 {
                    break;
                }
            }
        }
        println!("-- H = {target_h} ({} example(s) found) --", examples.len());
        for digits in &examples {
            let mut cpm = build_cpm(&params, 1);
            load_pattern(&mut cpm, digits);
            let prod_h = cpm.terms.global_energy(&cpm.state);
            let indep_h = h_two_cell(digits, &neighbours, &params);
            println!(
                "  pattern {}: production={:.6} independent={:.6} diff={:.2e}",
                pattern_of(digits),
                prod_h,
                indep_h,
                prod_h - indep_h
            );
        }
    }

    // ---- Check #1/#2: does the REAL sampler (attempt()) ever visit a
    // target-bin state, starting from the test's actual seed configuration? ----
    println!("\n=== Check #1/#2: does the real sampler visit target-bin states? ===");
    let mut cpm = build_cpm(&params, 2);
    let mut start_digits = [0u8; SITES2];
    start_digits[GRID2 + 1] = 1;
    start_digits[2 * GRID2 + 2] = 2;
    load_pattern(&mut cpm, &start_digits);
    cpm.state.conservative_energy = cpm.terms.global_energy(&cpm.state);

    let mut visited_patterns: HashSet<u32> = HashSet::new();
    let mut visited_at_target_h: [u64; 5] = [0; 5];
    const N: u64 = 20_000_000;
    for _ in 0..N {
        attempt(&mut cpm, false);
        let mut digits = [0u8; SITES2];
        for (site, slot) in digits.iter_mut().enumerate() {
            let flat = cpm.state.lattice.flat_index([site / GRID2, site % GRID2]);
            *slot = cpm.state.lattice.get(flat) as u8;
        }
        let pattern = pattern_of(&digits);
        if visited_patterns.insert(pattern) {
            let h = h_two_cell(&digits, &neighbours, &params);
            for (i, &target_h) in target_hs.iter().enumerate() {
                if (h - target_h).abs() < 1e-9 {
                    visited_at_target_h[i] += 1;
                }
            }
        }
    }
    println!(
        "After {N} attempts: {} distinct patterns visited by the real sampler",
        visited_patterns.len()
    );
    for (i, &target_h) in target_hs.iter().enumerate() {
        println!(
            "  distinct patterns visited at H={target_h}: {}",
            visited_at_target_h[i]
        );
    }

    // ---- Cross-check: BFS reachability (independent move generator) vs
    // sampler-visited set, from the SAME start state. ----
    println!("\n=== BFS reachability from the same start state ===");
    let start_pattern = pattern_of(&start_digits);
    let mut bfs_visited = vec![false; total as usize];
    bfs_visited[start_pattern as usize] = true;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start_pattern);
    let mut bfs_count = 1u64;
    while let Some(state) = queue.pop_front() {
        let digits = digits_of(state);
        for site in 0..SITES2 {
            let current = digits[site];
            let mut seen_owners = [false; 3];
            for slot in vn[site] {
                let neighbour_owner = match slot {
                    Some(n) => digits[n],
                    None => 0,
                };
                if neighbour_owner != current {
                    seen_owners[neighbour_owner as usize] = true;
                }
            }
            for new_owner in 0u8..3 {
                if !seen_owners[new_owner as usize] {
                    continue;
                }
                let mut new_digits = digits;
                new_digits[site] = new_owner;
                if !new_digits.contains(&1) || !new_digits.contains(&2) {
                    continue;
                }
                let new_state = pattern_of(&new_digits);
                if !bfs_visited[new_state as usize] {
                    bfs_visited[new_state as usize] = true;
                    bfs_count += 1;
                    queue.push_back(new_state);
                }
            }
        }
    }
    println!("BFS reachable from start: {bfs_count}");
    let sampler_not_in_bfs = visited_patterns
        .iter()
        .filter(|&&p| !bfs_visited[p as usize])
        .count();
    println!("sampler-visited patterns NOT in BFS-reachable set: {sampler_not_in_bfs}");

    // Are the target-H example patterns found earlier actually in the BFS set?
    for &target_h in &target_hs {
        for p in 0..total {
            let digits = digits_of(p);
            if !digits.contains(&1) || !digits.contains(&2) {
                continue;
            }
            let h = h_two_cell(&digits, &neighbours, &params);
            if (h - target_h).abs() < 1e-9 {
                println!(
                    "  H={target_h} pattern {p}: bfs_reachable={} sampler_visited={}",
                    bfs_visited[p as usize],
                    visited_patterns.contains(&p)
                );
                break;
            }
        }
    }

    // ---- Check #3: print example grids for the target bins. ----
    println!("\n=== Check #3: example grids for target bins ===");
    for &target_h in &target_hs {
        for p in 0..total {
            let digits = digits_of(p);
            if !digits.contains(&1) || !digits.contains(&2) {
                continue;
            }
            let h = h_two_cell(&digits, &neighbours, &params);
            if (h - target_h).abs() < 1e-9 {
                println!("H={target_h}:");
                print_grid(&digits);
                break;
            }
        }
    }
}
