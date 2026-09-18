//! Runs the generic engine at `D = 2` and asserts it reproduces a frozen
//! reference trajectory bit-identically. This is not a "does the physics
//! look right" test — `exact_boltzmann.rs` and the brute-force checker
//! already own that — it is a tripwire against a *later* 3D-motivated
//! refactor (to `lattice.rs`'s offset tables, `dynamics.rs`'s incremental
//! bookkeeping, or anything else touched to make 3D work) silently
//! perturbing the already-validated 2D dynamics. Both dimensions run
//! through the same generic code, so a 3D-only change is not guaranteed to
//! leave 2D alone unless something actually checks.
//!
//! If this test's literals ever need updating, that is worth pausing on
//! before just re-pasting new numbers: it means *something* about 2D
//! behaviour changed, and the question is always whether that was the
//! change actually intended.
//!
//! Deterministic RNG plus fixed floating-point instruction ordering on a
//! given build makes bit-for-bit reproduction the right bar here, not an
//! approximate one — the same property `monte_carlo::tests::
//! same_seed_gives_a_bit_identical_trajectory` checks between two *live*
//! runs, just pinned against a value captured once and frozen in source
//! rather than only checked against a second run in the same test process.

use cpm_core::cell::CellType;
use cpm_core::config::{model_hash, AcceptanceMode, InitializationSpec, UserConfig};
use cpm_core::initialization::initialize;
use cpm_core::lattice::Boundary;
use cpm_core::model::CPM;
use cpm_core::monte_carlo::run_mcs;

// Large relative to target_volume, matching `monte_carlo.rs`'s own
// dynamics-smoke-test fixture: keeps ordinary 500-MCS growth/jiggle well
// clear of the half-box unwrapped-position-tracking guard, which a small
// periodic grid can trip on nothing more than ordinary drift.
const GRID: usize = 30;

/// Two 2x2 blocks, off-centre from each other and from the grid centre, so
/// nothing here is accidentally symmetric — a bug that only shows up when
/// e.g. `x` and `y` are treated inconsistently would otherwise have a
/// chance to hide behind a symmetric fixture.
fn explicit_lattice() -> Vec<cpm_core::lattice::CellId> {
    let mut lattice = vec![0 as cpm_core::lattice::CellId; GRID * GRID];
    let at = |r: usize, c: usize| r * GRID + c;
    for (r, c) in [(6, 8), (6, 9), (7, 8), (7, 9)] {
        lattice[at(r, c)] = 1;
    }
    for (r, c) in [(18, 20), (18, 21), (19, 20), (19, 21)] {
        lattice[at(r, c)] = 2;
    }
    lattice
}

fn reference_cpm(seed: u64) -> CPM<2> {
    let config = UserConfig::<2> {
        grid: Some([GRID, GRID]),
        boundary: Some(Boundary::Periodic),
        acceptance: Some(AcceptanceMode::Metropolis),
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
                lambda_volume: 1.2,
                lambda_interface: 0.15,
                lambda_act: 0.0,
                max_act: 0,
            },
        ],
        cell_counts: vec![1, 1],
        adhesion: Some(vec![
            vec![0.0, 1.0, 1.5],
            vec![1.0, 0.5, 2.0],
            vec![1.5, 2.0, 0.5],
        ]),
        initialization: Some(InitializationSpec::Explicit {
            lattice: explicit_lattice(),
            cell_type_of: vec![0, 1],
        }),
        burn_in_mcs: Some(0),
        readout_mcs: Some(1),
        sampling_interval_mcs: Some(1),
        seed: Some(seed),
        ..Default::default()
    }
    .resolve()
    .unwrap();
    let mut cpm = CPM::new(config).unwrap();
    initialize(&mut cpm).unwrap();
    cpm
}

/// Frozen on 2026-09-15 from a real run of this exact fixture, current at
/// that commit; `model_hash` re-frozen on 2026-09-16 when
/// `config::CONVENTION_VERSION` was added to `convention_hash`'s inputs,
/// and again on 2026-09-17 when `config::ProposalMode` was added as a
/// `ResolvedConfig`/`convention_hash` field — in both cases the simulation
/// itself is untouched (the other four assertions are unchanged, and this
/// fixture resolves `proposal` to its `Uniform` default either way), only
/// the hash of "which conventions produced this" moved, exactly as
/// intended by adding those fields. Regenerate by temporarily replacing
/// the right-hand side of each assertion below with a clearly-wrong
/// placeholder (e.g. `vec![0, 0]` or `0u64`) and copying the actual values
/// out of the resulting assertion failure — `cargo test`'s `assert_eq!`
/// prints both sides in full.
#[test]
fn frozen_2d_trajectory_reproduces_bit_identically() {
    let mut cpm = reference_cpm(20260915);
    run_mcs(&mut cpm, 500);

    assert_eq!(cpm.mcs, 500);
    assert_eq!(cpm.state.volume, vec![7, 9]);
    assert_eq!(cpm.state.interface, vec![28, 32]);
    assert_eq!(
        cpm.state.conservative_energy.to_bits(),
        4638468362461406824u64, // 124.60000000000002274
        "conservative_energy = {:.17}",
        cpm.state.conservative_energy
    );
    assert_eq!(
        model_hash(&cpm.config, &cpm.state),
        "60ec713de9d698ca55697ff41339081739dfbd506b6117049acaeb7ccbc1d390"
    );
}
