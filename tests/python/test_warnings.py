import warnings

import cpm
import pytest
from cpm.warnings import CPMInitializationWarning


def test_is_a_user_warning():
    assert issubclass(CPMInitializationWarning, UserWarning)


def test_can_be_raised_and_caught():
    with pytest.warns(CPMInitializationWarning):
        warnings.warn("default initialisation used", CPMInitializationWarning, stacklevel=2)


def test_suppressible_via_simplefilter():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", CPMInitializationWarning)
        warnings.warn("default initialisation used", CPMInitializationWarning, stacklevel=2)
        assert len(caught) == 0


def _sim():
    sim = cpm.CPM(grid=(20, 20), boundary="periodic", seed=1)
    sim.register_cell_type(
        name="a", target_volume=20, target_interface=50, lambda_volume=1.0, lambda_interface=1.0
    )
    sim.add_cells(cell_type="a", n=2)
    sim.set_adhesion([[0.0, 1.0], [1.0, 0.5]])
    return sim


def test_run_without_calling_initialize_warns_on_default_placement():
    sim = _sim()
    with pytest.warns(CPMInitializationWarning):
        sim.run(burn_in_mcs=1, readout_mcs=1, sampling_interval_mcs=1)


def test_warn_on_default_init_false_suppresses_it():
    sim = _sim()
    sim.initialize(warn_on_default_init=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(burn_in_mcs=1, readout_mcs=1, sampling_interval_mcs=1)
        assert not any(issubclass(w.category, CPMInitializationWarning) for w in caught)


def test_explicit_placement_never_warns_regardless_of_the_flag():
    sim = cpm.CPM(grid=(4, 4), boundary="fixed", seed=1)
    sim.register_cell_type(
        name="a", target_volume=1, target_interface=4, lambda_volume=1.0, lambda_interface=1.0
    )
    sim.set_adhesion([[0.0, 1.0], [1.0, 0.0]])
    lattice = [0] * 16
    lattice[5] = 1
    sim.initialize(lattice=lattice, cell_type_of=[0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(burn_in_mcs=0, readout_mcs=1, sampling_interval_mcs=1)
        assert not any(issubclass(w.category, CPMInitializationWarning) for w in caught)
