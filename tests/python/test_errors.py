"""Configuration errors surface as Python exceptions with actionable
messages, never as a crash."""

import pytest

import cpm


def test_unsupported_grid_length_raises_value_error():
    with pytest.raises(ValueError, match="length 2 or 3"):
        cpm.CPM(grid=(1, 2, 3, 4), boundary="periodic", seed=1)


def test_unknown_neighborhood_string_raises_value_error():
    with pytest.raises(ValueError, match="unknown neighbourhood"):
        cpm.CPM(grid=(10, 10), boundary="periodic", seed=1, copy_neighborhood="hexagonal")


def test_unknown_boundary_string_raises_value_error():
    with pytest.raises(ValueError, match="unknown boundary"):
        cpm.CPM(grid=(10, 10), boundary="squishy", seed=1)


def test_non_square_adhesion_matrix_raises_at_run_time():
    sim = cpm.CPM(grid=(10, 10), boundary="periodic", seed=1)
    sim.register_cell_type(
        name="a", target_volume=20, target_interface=50, lambda_volume=1.0, lambda_interface=1.0
    )
    sim.add_cells(cell_type="a", n=1)
    sim.set_adhesion([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])  # 3x2, needs 2x2
    with pytest.raises(ValueError, match="adhesion matrix"):
        sim.run(burn_in_mcs=1, readout_mcs=1, sampling_interval_mcs=1)


def test_positive_lambda_act_with_zero_max_act_raises_immediately():
    sim = cpm.CPM(grid=(10, 10), boundary="periodic", seed=1)
    with pytest.raises(ValueError, match="max_act"):
        sim.register_cell_type(
            name="a",
            target_volume=20,
            target_interface=50,
            lambda_volume=1.0,
            lambda_interface=1.0,
            lambda_act=1.0,
            max_act=0,
        )


def test_unknown_cell_type_in_add_cells_raises_value_error():
    sim = cpm.CPM(grid=(10, 10), boundary="periodic", seed=1)
    with pytest.raises(ValueError, match="unknown cell type"):
        sim.add_cells(cell_type="ghost", n=1)
