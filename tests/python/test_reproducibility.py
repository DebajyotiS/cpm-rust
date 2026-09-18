"""Verifies that the same configuration and seed through the Python API
reproduce the Rust-side reference trajectory.

Mirrors `crates/cpm-core/tests/dimension_consistency.rs`'s exact fixture
(same grid, cell types, adhesion, explicit placement, seed) so the frozen
`conservative_energy` bit pattern captured there can be reused directly as
the oracle here. `burn_in_mcs=0, readout_mcs=500, sampling_interval_mcs=500`
makes `output::run`'s single readout chunk call `run_mcs(cpm, 500)` exactly
once on a freshly-placed `CPM` — the same call the Rust test makes directly
— so the resulting sample lands on the identical trajectory point.
`result.metadata.model_hash` is not compared: it also hashes
`burn_in_mcs`/`readout_mcs`/`sampling_interval_mcs`, which legitimately
differ here (0/500/500) from the Rust fixture's own resolved values (0/1/1,
placeholders the Rust test never actually runs through `output::run`) — that
divergence is expected, not a reproducibility failure.
"""

import struct

import cpm

GRID = 30


def _explicit_lattice():
    lattice = [0] * (GRID * GRID)

    def at(r, c):
        return r * GRID + c

    for r, c in [(6, 8), (6, 9), (7, 8), (7, 9)]:
        lattice[at(r, c)] = 1
    for r, c in [(18, 20), (18, 21), (19, 20), (19, 21)]:
        lattice[at(r, c)] = 2
    return lattice


def _run(seed: int):
    sim = cpm.CPM(
        grid=(GRID, GRID),
        boundary="periodic",
        seed=seed,
        copy_neighborhood="von_neumann",
        connectivity_neighborhood="von_neumann",
        acceptance="metropolis",
    )
    sim.add_cell_type(
        "a", target_volume=12, target_interface=24, lambda_volume=1.0, lambda_interface=0.2
    )
    sim.add_cell_type(
        "b", target_volume=12, target_interface=24, lambda_volume=1.2, lambda_interface=0.15
    )
    sim.add_cells("a", 1)
    sim.add_cells("b", 1)
    sim.set_adhesion([[0.0, 1.0, 1.5], [1.0, 0.5, 2.0], [1.5, 2.0, 0.5]])
    sim.initialize(lattice=_explicit_lattice(), cell_type_of=[0, 1])
    return sim.run(burn_in_mcs=0, readout_mcs=500, sampling_interval_mcs=500)


def _bits_to_float(bits: int) -> float:
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


# Frozen alongside `dimension_consistency.rs`'s own frozen values (same
# fixture, same seed, same total MCS count) — see that file's doc comment
# for how to regenerate if the dynamics ever legitimately change.
FROZEN_CONSERVATIVE_ENERGY_BITS = 4638468362461406824
FROZEN_VOLUME = [7, 9]
FROZEN_INTERFACE = [28, 32]


def test_python_driven_run_matches_the_frozen_rust_trajectory():
    result = _run(seed=20260915)
    assert len(result.samples) == 1
    sample = result.samples[0]

    assert sample.mcs == 500
    assert list(sample.ids) == [1, 2]
    assert list(sample.volume) == FROZEN_VOLUME
    assert list(sample.interface) == FROZEN_INTERFACE
    assert sample.conservative_energy == _bits_to_float(FROZEN_CONSERVATIVE_ENERGY_BITS), (
        f"conservative_energy = {sample.conservative_energy!r}"
    )


def test_different_seed_diverges():
    a = _run(seed=20260915)
    b = _run(seed=1)
    assert a.samples[0].conservative_energy != b.samples[0].conservative_energy
