"""Tests for the density-design helper: `n = cells_for_density(grid,
target_volume, phi)`, inverting `phi = sum(V_c) / N_sites`."""

import warnings

import cpm
import pytest
from cpm.warnings import CPMLowResolutionWarning


def test_round_trips_against_the_phi_formula():
    grid = (64, 64)
    target_volume = 100  # radius ~5.6 sites, above the floor: no warning noise
    n = cpm.cells_for_density(grid=grid, target_volume=target_volume, phi=0.8)
    n_sites = grid[0] * grid[1]
    implied_phi = n * target_volume / n_sites
    assert implied_phi == pytest.approx(0.8, abs=0.02)


def test_warns_below_the_radius_floor():
    with pytest.warns(CPMLowResolutionWarning):
        cpm.cells_for_density(grid=(64, 64), target_volume=4, phi=0.8)


def test_does_not_warn_above_the_radius_floor():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cpm.cells_for_density(grid=(64, 64), target_volume=1000, phi=0.8)
        assert not any(issubclass(w.category, CPMLowResolutionWarning) for w in caught)
