"""Local stand-in for Genesis' internal ``SubTerrain``.

Genesis' ``SubTerrain`` is a tiny class with only 5 fields: the
geometry (width, length), the resolution (horizontal_scale,
vertical_scale) and a mutable ``height_field_raw`` array that the
terrain functions modify in place.  The class itself is defined in
``genesis/ext/isaacgym/terrain_utils.py`` but importing it from there
would couple us to a private Genesis path, so we replicate the
interface locally.
"""

from __future__ import annotations

import numpy as np


class SubTerrain:
    """Mutable heightfield for a single sub-terrain tile.

    Attributes:
        terrain_name:    A name (purely informational).
        width:           Number of cells along the **first** axis
                         (the x-axis in the world frame).  In
                         Genesis' heightfield format, ``height_field_raw``
                         is ``(width, length)``.
        length:          Number of cells along the **second** axis
                         (the y-axis).
        vertical_scale:  Meters per cell in the height direction.
        horizontal_scale: Meters per cell in the horizontal
                          (x/y) directions.
        height_field_raw: 2D float array of shape ``(width, length)``
                          holding the **discrete** heights (in
                          ``vertical_scale`` units, not meters).
    """

    def __init__(
        self,
        terrain_name: str = "terrain",
        width: int = 256,
        length: int = 256,
        vertical_scale: float = 1.0,
        horizontal_scale: float = 1.0,
    ) -> None:
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.float64)
