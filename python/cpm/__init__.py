"""Cellular Potts Model tissue simulator: Python configuration and analysis
layer over the Rust `cpm-core` engine.

`CPM` (`.model`) is the builder API for configuring and running a
simulation; `cells_for_density` (`.parameters`) is a helper for choosing a
cell count that targets a given confluence.
"""

from .model import CPM
from .parameters import cells_for_density
from .warnings import CPMInitializationWarning, CPMLowResolutionWarning

__all__ = [
    "CPM",
    "cells_for_density",
    "CPMInitializationWarning",
    "CPMLowResolutionWarning",
]
