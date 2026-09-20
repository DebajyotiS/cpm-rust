"""End-to-end smoke test through the canonical builder shape:
`CPM(...)` -> `register_cell_type` -> `add_cells` -> `set_adhesion` -> `run`.
"""

import numpy as np
import pytest

import cpm
from cpm.warnings import CPMInitializationWarning


def _small_confluent_sim(seed=1):
    sim = cpm.CPM(
        grid=(30, 30),
        boundary="periodic",
        seed=seed,
        copy_neighborhood="von_neumann",
        connectivity_neighborhood="von_neumann",
    )
    sim.register_cell_type(
        name="epithelial",
        target_volume=50,
        target_interface=75,
        lambda_volume=10.0,
        lambda_interface=2.0,
    )
    sim.add_cells(cell_type="epithelial", n=5)
    sim.set_adhesion([[0.0, 5.0], [5.0, 2.0]])
    return sim


def test_run_end_to_end_produces_expected_sample_count_and_shapes():
    sim = _small_confluent_sim()
    # No explicit `initialize()` call, so the default scatter-and-grow path
    # fires `CPMInitializationWarning` — expected here, not incidental noise.
    with pytest.warns(CPMInitializationWarning):
        result = sim.run(burn_in_mcs=20, readout_mcs=50, sampling_interval_mcs=10)

    assert len(result.samples) == 5  # readout_mcs // sampling_interval_mcs
    assert result.status.kind in {"Ok", "CellLost", "Fragmented", "NotEquilibrated", "Degenerate"}
    assert 0.0 < result.metadata.derived_phi <= 1.0
    assert isinstance(result.metadata.convention_hash, str) and result.metadata.convention_hash

    sample = result.samples[-1]
    n_cells = len(sample.ids)
    assert n_cells <= 5  # CellLost could shrink this in principle, not expected here
    assert sample.centroid.shape == (n_cells, 2)
    assert sample.eigenvalues.shape == (n_cells, 2)
    assert sample.volume.dtype == np.uint32
    assert sample.contact.shape == (n_cells + 1, n_cells + 1)
    assert sample.contact.dtype == np.uint32
    # Each unordered pair is stored once at its canonical (min, max) index
    # (`ContactGraph::index`), so the matrix is upper-triangular, not
    # mirrored — the lower triangle (including the diagonal) is always zero.
    assert np.array_equal(np.tril(sample.contact), np.zeros_like(sample.contact))


def test_include_lattice_defaults_to_none_and_can_be_requested():
    sim = _small_confluent_sim()
    with pytest.warns(CPMInitializationWarning):
        without = sim.run(burn_in_mcs=5, readout_mcs=10, sampling_interval_mcs=5)
    assert without.samples[0].lattice is None

    sim2 = _small_confluent_sim()
    with pytest.warns(CPMInitializationWarning):
        with_lattice = sim2.run(
            burn_in_mcs=5, readout_mcs=10, sampling_interval_mcs=5, include_lattice=True
        )
    sample = with_lattice.samples[-1]
    lattice = sample.lattice
    assert lattice is not None
    assert lattice.shape == (30, 30)
    assert lattice.dtype == np.uint32
    # The lattice snapshot and the per-cell volume readout are two
    # independent codepaths reporting the same underlying state -- they
    # must agree exactly.
    for cell_id, volume in zip(sample.ids, sample.volume):
        assert int((lattice == cell_id).sum()) == volume


def test_two_cell_types_are_labelled_correctly():
    sim = cpm.CPM(grid=(30, 30), boundary="periodic", seed=3)
    sim.register_cell_type(
        name="epithelial", target_volume=30, target_interface=60, lambda_volume=1.0,
        lambda_interface=1.0,
    )
    sim.register_cell_type(
        name="stem", target_volume=30, target_interface=60, lambda_volume=1.0,
        lambda_interface=1.0,
    )
    sim.add_cells(cell_type="epithelial", n=3)
    sim.add_cells(cell_type="stem", n=2)
    sim.set_adhesion([[0.0, 1.0, 1.0], [1.0, 0.5, 1.0], [1.0, 1.0, 0.5]])
    with pytest.warns(CPMInitializationWarning):
        result = sim.run(burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5)

    assert result.cell_type_names == ["epithelial", "stem"]
    assert len(result.cell_type_index) == 5
    assert sorted(result.cell_type_index.tolist()) == [0, 0, 0, 1, 1]


def test_run_is_deterministic_given_the_same_seed():
    with pytest.warns(CPMInitializationWarning):
        a = _small_confluent_sim(seed=7).run(
            burn_in_mcs=10, readout_mcs=10, sampling_interval_mcs=10
        )
    with pytest.warns(CPMInitializationWarning):
        b = _small_confluent_sim(seed=7).run(
            burn_in_mcs=10, readout_mcs=10, sampling_interval_mcs=10
        )
    assert a.samples[0].conservative_energy == b.samples[0].conservative_energy
    assert np.array_equal(a.samples[0].volume, b.samples[0].volume)
