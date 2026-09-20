"""`run_batch`, `theta_names`, and `extract_theta`: parallel execution of
independent simulations across `theta`, one call at a time.
"""

import warnings

import cpm
import pytest
from cpm.warnings import CPMInitializationWarning


def _two_type_sim(seed=1):
    sim = cpm.CPM(grid=(20, 20), boundary="periodic", seed=seed)
    sim.register_cell_type(
        name="a", target_volume=9, target_interface=20, lambda_volume=1.0, lambda_interface=0.2
    )
    sim.register_cell_type(
        name="b", target_volume=9, target_interface=20, lambda_volume=1.5, lambda_interface=0.3
    )
    sim.add_cells("a", 2)
    sim.add_cells("b", 2)
    sim.set_adhesion([[0.0, 3.0, 3.0], [3.0, 1.0, 5.0], [3.0, 5.0, 1.0]])
    return sim


def test_theta_names_and_extract_theta_match_canonical_layout():
    sim = _two_type_sim()
    names = sim.theta_names()
    theta = sim.extract_theta()
    assert len(names) == len(theta)
    assert names[:4] == [
        "lambda_volume[a]",
        "lambda_volume[b]",
        "lambda_interface[a]",
        "lambda_interface[b]",
    ]
    assert theta[:4] == [1.0, 1.5, 0.2, 0.3]


def test_run_batch_returns_results_in_input_order():
    sim = _two_type_sim()
    base = sim.extract_theta()
    thetas = [base, [v * 1.1 for v in base], [v * 0.9 for v in base]]
    with pytest.warns(CPMInitializationWarning):
        results = sim.run_batch(
            thetas, burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=42
        )
    assert len(results) == len(thetas)
    expected_statuses = {"Ok", "CellLost", "Fragmented", "NotEquilibrated", "Degenerate"}
    for result in results:
        assert result.status.kind in expected_statuses


def test_run_batch_fires_exactly_one_warning_for_the_whole_batch():
    sim = _two_type_sim()
    base = sim.extract_theta()
    thetas = [base] * 5
    with pytest.warns(CPMInitializationWarning) as record:
        sim.run_batch(thetas, burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=1)
    assert len(record) == 1


def test_run_batch_members_share_one_convention_hash():
    sim = _two_type_sim()
    base = sim.extract_theta()
    thetas = [base, [v * 2 for v in base]]
    with pytest.warns(CPMInitializationWarning):
        results = sim.run_batch(
            thetas, burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=1
        )
    hashes = {r.metadata.convention_hash for r in results}
    assert len(hashes) == 1


def test_run_batch_gives_each_member_a_different_seed():
    sim = _two_type_sim()
    base = sim.extract_theta()
    thetas = [base, base, base]  # identical theta: any divergence is from the seed alone
    with pytest.warns(CPMInitializationWarning):
        results = sim.run_batch(
            thetas, burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=7
        )
    seeds = {r.metadata.seed for r in results}
    assert len(seeds) == len(thetas)


def test_wrong_length_theta_raises_a_clean_error():
    sim = _two_type_sim()
    base = sim.extract_theta()
    bad_theta = base[:-1]  # one element short
    with pytest.raises(ValueError):
        sim.run_batch(
            [base, bad_theta], burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=1
        )


def test_run_batch_can_silence_the_default_init_warning():
    sim = _two_type_sim()
    sim.initialize(warn_on_default_init=False)
    base = sim.extract_theta()
    with warnings.catch_warnings():
        warnings.simplefilter("error", CPMInitializationWarning)
        sim.run_batch(
            [base, base], burn_in_mcs=5, readout_mcs=5, sampling_interval_mcs=5, master_seed=1
        )
