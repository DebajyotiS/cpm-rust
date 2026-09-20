"""The user-facing builder API.

Wraps `_cpm.CPM` (the PyO3 pyclass, `Simulation` in Rust source — named
`CPM` only in the compiled extension to avoid colliding with
`cpm_core::model::CPM<D>`'s own name). Everything here delegates straight
through to the Rust builder except `run`, which is where the
`used_default`/`warn_on_default_init` signal `_cpm.CPM.run` returns turns
into an actual `warnings.warn(...)` call — kept in Python rather than Rust
so the warning *policy* lives next to the stdlib machinery
(`warnings.simplefilter`) that governs it.
"""

from __future__ import annotations

import warnings
from typing import Any

from . import _cpm
from .warnings import CPMInitializationWarning


class CPM:
    """One CPM trajectory. Configure with the constructor and the
    ``register_cell_type``/``add_cells``/``set_adhesion``/``initialize``
    builder calls, then call :meth:`run`. Nothing executes in Rust until
    :meth:`run` is called: Python only configures the simulation, it does
    not step it.
    """

    def __init__(
        self,
        grid: tuple[int, ...],
        boundary: str,
        seed: int,
        copy_neighborhood: str = "von_neumann",
        energy_neighborhood: str | None = None,
        connectivity_neighborhood: str = "von_neumann",
        acceptance: str | None = None,
        proposal: str | None = None,
        active_terms: dict[str, bool] | None = None,
    ) -> None:
        self._raw = _cpm.CPM(
            grid=list(grid),
            boundary=boundary,
            seed=seed,
            copy_neighborhood=copy_neighborhood,
            energy_neighborhood=energy_neighborhood,
            connectivity_neighborhood=connectivity_neighborhood,
            acceptance=acceptance,
            proposal=proposal,
            active_terms=active_terms,
        )
        self._warn_on_default_init = True

    def register_cell_type(
        self,
        name: str,
        target_volume: int,
        target_interface: int,
        lambda_volume: float,
        lambda_interface: float,
        lambda_act: float = 0.0,
        max_act: int = 0,
    ) -> None:
        self._raw.register_cell_type(
            name,
            target_volume,
            target_interface,
            lambda_volume,
            lambda_interface,
            lambda_act,
            max_act,
        )

    def add_cells(self, cell_type: str, n: int) -> None:
        self._raw.add_cells(cell_type, n)

    def set_adhesion(self, matrix: Any) -> None:
        """`matrix` is `(n_types + 1) x (n_types + 1)` (index 0 = medium),
        symmetric. Accepts a nested list or anything convertible to one
        (e.g. a NumPy array via `.tolist()`-equivalent iteration)."""
        rows = [list(row) for row in matrix]
        self._raw.set_adhesion(rows)

    def initialize(
        self,
        lattice: Any | None = None,
        cell_type_of: Any | None = None,
        warn_on_default_init: bool = True,
    ) -> None:
        """Configures the initial placement; does not execute it. Supplying
        `lattice` selects explicit placement; omitting it selects the
        default scatter-and-grow, which `run()` will warn about unless
        `warn_on_default_init` is `False`."""
        self._warn_on_default_init = warn_on_default_init
        lattice_list = list(lattice) if lattice is not None else None
        cell_type_of_list = list(cell_type_of) if cell_type_of is not None else None
        self._raw.initialize(lattice_list, cell_type_of_list, warn_on_default_init)

    def run(
        self,
        burn_in_mcs: int,
        readout_mcs: int,
        sampling_interval_mcs: int,
        include_lattice: bool = False,
    ):
        """Runs the entire trajectory in one call (no per-move FFI back into
        Python) and returns a `RunResult` (`.metadata`, `.status`,
        `.samples`).

        `include_lattice=True` additionally populates each sample's
        `.lattice` (a `grid`-shaped NumPy array of cell ids) — a debug and
        visualisation escape hatch, off by default since a real inference
        run taking many samples would otherwise pay for a full lattice copy
        per sample it never looks at."""
        used_default, result = self._raw.run(
            burn_in_mcs, readout_mcs, sampling_interval_mcs, include_lattice
        )
        if used_default and self._warn_on_default_init:
            warnings.warn(
                "default (scatter-and-grow) initial placement used; call "
                "initialize(lattice=...) for an explicit placement, or "
                "initialize(warn_on_default_init=False) to silence this",
                CPMInitializationWarning,
                stacklevel=2,
            )
        return result

    def theta_names(self) -> list[str]:
        """The canonical `theta` parameter names, in the order every `theta`
        vector this simulation accepts must use. Requires
        `register_cell_type`/`set_adhesion` to already be called; everything
        `theta` covers comes from cell-type and adhesion structure, not from
        `burn_in_mcs`/`readout_mcs`/`sampling_interval_mcs`, which don't need
        to be set yet to call this."""
        return self._raw.theta_names()

    def extract_theta(self) -> list[float]:
        """The current configuration's `theta`, in canonical order — a
        starting point to perturb before calling `run_batch`."""
        return self._raw.extract_theta()

    def run_batch(
        self,
        thetas: Any,
        burn_in_mcs: int,
        readout_mcs: int,
        sampling_interval_mcs: int,
        master_seed: int,
        include_lattice: bool = False,
    ) -> list[Any]:
        """Runs one independent trajectory per entry in `thetas`, in
        parallel across available CPU cores (Rayon, with the GIL released
        for the whole batch). Each simulation's seed is derived
        deterministically from `(master_seed, index)`, so the results are
        reproducible regardless of how the work happens to get scheduled
        across threads.

        Returns a list of `RunResult`, one per theta, in the same order as
        `thetas`. At most one `CPMInitializationWarning` fires for the whole
        call if any simulation used default placement — not one per
        simulation, since a training-set-sized batch would otherwise be
        unusably noisy."""
        thetas_list = [list(theta) for theta in thetas]
        any_used_default, results = self._raw.run_batch(
            thetas_list,
            burn_in_mcs,
            readout_mcs,
            sampling_interval_mcs,
            master_seed,
            include_lattice,
        )
        if any_used_default and self._warn_on_default_init:
            warnings.warn(
                "default (scatter-and-grow) initial placement used in at "
                "least one batch member; call initialize(lattice=...) for "
                "an explicit placement, or "
                "initialize(warn_on_default_init=False) to silence this",
                CPMInitializationWarning,
                stacklevel=2,
            )
        return results
