"""Configuration-design helpers.

Pure arithmetic — `phi = sum(V_c) / N_sites` is derived, never a settable
field, so this module only ever helps a user *choose* a cell count before
building a `CPM`; nothing here touches Rust, and nothing here is part of
`convention_hash` or any other reproducibility record.
"""

from __future__ import annotations

import math
import warnings

from .warnings import CPMLowResolutionWarning

#: Below this many lattice sites of radius, discretisation starts
#: dominating the morphology readouts. This is a design-time sanity check
#: rather than a frozen scientific convention: it doesn't affect Rust's
#: output at all.
_MIN_CELL_RADIUS_SITES = 5.0


def _implied_cell_radius(target_volume: float, dimensionality: int) -> float:
    """Inverts the volume of a `dimensionality`-ball of the given volume —
    plain geometry, not a modelling choice: a 2D "volume" is really an area
    (`V = pi r^2`), a 3D volume is `V = (4/3) pi r^3`."""
    if dimensionality == 2:
        return math.sqrt(target_volume / math.pi)
    if dimensionality == 3:
        return (3.0 * target_volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    raise ValueError(
        f"grid must have length 2 or 3 (dimensionality is inferred from it); "
        f"got length {dimensionality}"
    )


def cells_for_density(grid: tuple[int, ...], target_volume: float, phi: float) -> int:
    """Inverts `phi = sum(V_c) / N_sites` for cell count `n`, given a grid
    and a per-cell target volume: `n = round(phi * N_sites / target_volume)`.

    Warns with :class:`CPMLowResolutionWarning` if the implied cell radius
    (from `target_volume` alone) is below ~5 lattice sites.
    """
    n_sites = math.prod(grid)
    radius = _implied_cell_radius(target_volume, len(grid))
    if radius < _MIN_CELL_RADIUS_SITES:
        warnings.warn(
            f"implied cell radius {radius:.2f} sites (from target_volume="
            f"{target_volume}) is below {_MIN_CELL_RADIUS_SITES:.0f} sites; "
            "lattice discretisation will start dominating the morphology "
            "readouts at this density",
            CPMLowResolutionWarning,
            stacklevel=2,
        )
    return round(phi * n_sites / target_volume)
