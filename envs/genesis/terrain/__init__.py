"""Local terrain module.

This module extends Genesis' built-in terrain generator with custom
terrain types and a heightfield-builder API, **without modifying any
Genesis source code**.  All new functions live in this folder so the
project remains portable across machines (no patches to
``genesis/ext/...`` needed).

Public API
==========

- :class:`SubTerrain` — minimal stand-in for the Genesis-internal
  ``SubTerrain`` (used by the terrain functions).
- :func:`make_subterrain` — build a single sub-terrain heightfield
  array.
- :func:`build_heightfield` — stitch a 2D grid of sub-terrains into
  a single heightfield array, ready to be passed as
  ``gs.morphs.Terrain(height_field=...)``.

Adding a new terrain type
=========================

1. Implement a function with the same signature as the other entries
   in :data:`TERRAIN_FUNCTIONS`:

       def my_terrain(sub, **params) -> SubTerrain:
           ...

2. Register it in :data:`TERRAIN_FUNCTIONS`.
3. Reference it by string in the 2D layout passed to
   :func:`build_heightfield`.
"""

from .subterrain import SubTerrain
from .functions import (
    TERRAIN_FUNCTIONS,
    make_subterrain,
    stairs_terrain_y,
    down_stairs_terrain_y,
)
from .builder import build_heightfield, build_go2_rough_layout

__all__ = [
    "SubTerrain",
    "TERRAIN_FUNCTIONS",
    "make_subterrain",
    "stairs_terrain_y",
    "down_stairs_terrain_y",
    "build_heightfield",
    "build_go2_rough_layout",
]
