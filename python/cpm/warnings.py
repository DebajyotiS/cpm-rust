"""Warnings raised by the Python-facing API.

`CPMInitializationWarning` is driven by a real signal: `cpm.model.CPM.run`
checks the `used_default` flag `_cpm.CPM.run` returns (itself driven by
`cpm_core::initialization::Outcome::used_default`) and raises this from
Python, not from Rust — keeping the warning *policy* (whether to actually
warn, given `warn_on_default_init`) in the same place `warnings.simplefilter`
and that flag naturally live.
"""


class CPMInitializationWarning(UserWarning):
    """Raised when a simulation falls back to default (scatter-and-grow)
    initial placement instead of an explicitly supplied one.

    Initial contact topology is not a neutral default: it affects how long
    burn-in needs to be before readouts are meaningful, so silently picking
    it for the user would be an undocumented modelling choice.

    Suppressible via the standard library (``warnings.simplefilter("ignore",
    CPMInitializationWarning)``) or via the ``warn_on_default_init=False``
    flag on ``CPM.initialize(...)``.
    """


class CPMLowResolutionWarning(UserWarning):
    """Raised by :func:`cpm.parameters.cells_for_density` when the implied
    cell radius (from the requested ``target_volume`` and ``phi``) falls
    below about 5 lattice sites.

    Below that radius, lattice discretisation starts dominating the
    morphology readouts — the shape descriptors computed at that density
    would say more about the grid than the tissue. It is a separate,
    narrowly-scoped warning rather than an overload of
    :class:`CPMInitializationWarning` for an unrelated condition.
    """
