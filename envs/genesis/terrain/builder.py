"""Stitch a 2D layout of sub-terrains into a single heightfield.

The layout is described by a 2D list of strings — one entry per cell.
For each cell we build a small :class:`SubTerrain`, apply the
appropriate terrain function, and paste the result into the master
heightfield array.  Cells **overlap by 1 cell** at the boundary
(Genesis' convention, see ``terrain.py`` line 65-67 in the Genesis
source) so adjacent cells share a continuous edge.

The returned array is what you pass to
``gs.morphs.Terrain(height_field=hf)``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .functions import make_subterrain


def build_heightfield(
    layout: Sequence[Sequence[str]],
    subterrain_size: tuple[float, float],
    horizontal_scale: float,
    vertical_scale: float,
    subterrain_parameters: Dict[str, Dict] | None = None,
) -> np.ndarray:
    """Stitch a 2D layout into one big heightfield.

    Parameters:
        layout:               2D list of terrain-type names, indexed
                              ``layout[row][col]``.  ``row`` grows in
                              +y (world), ``col`` grows in +x
                              (matches Genesis' ``subterrain_types``
                              convention).
        subterrain_size:      ``(sx, sy)`` size of each cell in meters
                              (e.g. ``(6.0, 6.0)``).
        horizontal_scale:     meters per cell in x/y.
        vertical_scale:       meters per cell in z.
        subterrain_parameters: per-type parameter dict.  For example::

                {
                    "stairs_terrain_y": {"step_width": 0.20,
                                          "step_height": 0.10},
                    "gap_terrain":      {"gap_size": 0.5, "depth": 0.5},
                }

    Returns:
        A 2D ``np.ndarray`` of shape ``(N_rows * (H - 1) + 1,
        N_cols * (W - 1) + 1)`` where ``H``, ``W`` are the cell
        resolution of one sub-terrain.  Units are **discrete**
        (``vertical_scale`` units, not meters).  This is the format
        Genesis expects for ``height_field=...``.
    """
    if subterrain_parameters is None:
        subterrain_parameters = {}

    n_rows = len(layout)
    n_cols = len(layout[0])
    if any(len(row) != n_cols for row in layout):
        raise ValueError(
            f"All rows of `layout` must have the same length; got "
            f"{[len(r) for r in layout]}"
        )

    sx, sy = subterrain_size
    # Number of cells per sub-terrain along each axis.  Genesis adds 1
    # cell so the edges overlap by 1 sample.  See Genesis' terrain.py:
    #     subterrain_rows = int(morph.subterrain_size[0] /
    #                           morph.horizontal_scale) + 1
    cells_per_x = int(round(sx / horizontal_scale)) + 1
    cells_per_y = int(round(sy / horizontal_scale)) + 1

    # Master heightfield, init to -inf so cell boundaries are
    # "no data" (we then take the max to merge cells).
    heightfield = np.full(
        (
            n_rows * (cells_per_x - 1) + 1,
            n_cols * (cells_per_y - 1) + 1,
        ),
        fill_value=-np.inf,
        dtype=np.float64,
    )

    for i in range(n_rows):  # i = row index = y axis in world
        for j in range(n_cols):  # j = col index = x axis in world
            ttype = layout[i][j]
            params = subterrain_parameters.get(ttype, {})

            sub = make_subterrain(
                ttype,
                width=cells_per_x,
                length=cells_per_y,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                **params,
            )

            # Paste this cell into the master heightfield.  Cells
            # overlap by 1 sample at the boundary (Genesis convention).
            y_lo = i * (cells_per_x - 1)
            y_hi = y_lo + cells_per_x
            x_lo = j * (cells_per_y - 1)
            x_hi = x_lo + cells_per_y
            heightfield[y_lo:y_hi, x_lo:x_hi] = np.maximum(
                heightfield[y_lo:y_hi, x_lo:x_hi],
                sub.height_field_raw,
            )

    # Any remaining -inf samples (shouldn't be any after a full
    # layout, but be safe) get clamped to 0.
    heightfield[np.isneginf(heightfield)] = 0.0
    return heightfield


# ---------------------------------------------------------------------------
# go2-rough specific layout (kept here so it lives next to the builder)
# ---------------------------------------------------------------------------

def build_go2_rough_layout(
    n_subterrains: tuple[int, int] = (5, 5),
    subterrain_size: tuple[float, float] = (6.0, 6.0),
    horizontal_scale: float = 0.1,
    vertical_scale: float = 0.005,
    step_width: float = 0.20,
    step_height: float = 0.10,
    num_steps: int = 15,
    **kwargs,
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Build the canonical 5x5 go2-rough heightfield.

    **Design principle**: Go2 spawns at the centre flat cell and
    walks outward in any cardinal direction.  Each direction leads
    to a different obstacle family — no direction is "dead ground".

    The staircase occupies the **south** column (col 2, rows 3-4)
    and forms a complete **up → plateau → down → flat** sequence:

    * **row 3, col 2** — ``stairs_terrain_y`` with 15 steps
      (0 → 1.5 m in the first 3 m, then 3 m flat plateau at 1.5 m).
    * **row 4, col 2** — ``down_stairs_terrain_y`` with 15 steps
      (1.5 m → 0 m in the first 3 m, then 3 m flat run-out at 0 m).

    Go2 walking +y from centre enters the stairs at h=0 (no wall),
    climbs to 1.5 m, crosses the plateau, descends back to 0 m, and
    has a generous flat run-out so it never "falls off" the stairs.

    Layout (5 rows x 5 columns, 6 m x 6 m cells -> 30 m x 30 m)::

        row 0  f   f   f   f   f    <- outermost flat margin
        row 1  f  rU  oS  sT  gP    <- NORTH perimeter
        row 2  f  oS   f  pT   f       centre (flat + box bumps/pit)
        row 3  f  gP  sU  sT  oS    <- SOUTH (gap, stairs-up+plateau,
        row 4  f   f  sD   f   f          stairs-down+flat run-out)

    Legend:
        - ``f``   = flat_terrain
        - ``rU``  = random_uniform_terrain (irregular bumps)
        - ``oS``  = discrete_obstacles_terrain (box bumps)
        - ``sT``  = stepping_stones_terrain (meihua piles)
        - ``gP``  = gap_terrain (gap/jump)
        - ``pT``  = pit_terrain (depression pit)
        - ``sU``  = stairs_terrain_y (ascending stairs +y + plateau, 0→1.5 m)
        - ``sD``  = down_stairs_terrain_y (descending stairs +y + flat, 1.5→0 m)

    Step dimensions:
        step_width=0.20 m, step_height=0.10 m, 15 steps -> 3 m ramp,
        1.50 m total rise, 26.57 deg angle.

    Returns:
        (heightfield, subterrain_parameters)
    """
    n_rows, n_cols = n_subterrains
    if n_rows != 5 or n_cols != 5:
        raise ValueError(
            f"go2-rough layout assumes 5x5 subterrains, got "
            f"{n_subterrains}."
        )

    top_height = step_height * num_steps  # 1.5 m

    layout: List[List[str]] = [
        ["flat_terrain"] * 5,                                                  # row 0
        [                                                                         # row 1 -- NORTH perimeter
            "flat_terrain",
            "random_uniform_terrain",
            "discrete_obstacles_terrain",
            "stepping_stones_terrain",
            "gap_terrain",
        ],
        [                                                                         # row 2 -- CENTRE
            "flat_terrain",
            "discrete_obstacles_terrain",
            "flat_terrain",
            "pit_terrain",
            "flat_terrain",
        ],
        [                                                                         # row 3 -- SOUTH perimeter
            "flat_terrain",
            "gap_terrain",
            "stairs_terrain_y",
            "stepping_stones_terrain",
            "discrete_obstacles_terrain",
        ],
        ["flat_terrain"] * 5,                                                  # row 4 (col 2 -> stairs down)
    ]
    # Row 4, col 2: descending stairs with flat run-out so Go2 never
    # "falls off" the staircase and learns to descend safely.
    layout[4][2] = "down_stairs_terrain_y"

    subterrain_parameters: Dict[str, Dict] = {
        "stairs_terrain": {
            "step_width": step_width,
            "step_height": step_height,
        },
        "stairs_terrain_y": {
            "step_width": step_width,
            "step_height": step_height,
            "num_steps": num_steps,
        },
        "down_stairs_terrain": {
            "step_width": step_width,
            "step_height": step_height,
            "base_height": top_height,
        },
        "down_stairs_terrain_y": {
            "step_width": step_width,
            "step_height": step_height,
            "base_height": top_height,
            "num_steps": num_steps,
        },
        "flat_terrain_at_height": {
            "height": top_height,
        },
        "discrete_obstacles_terrain": {
            "max_height": 0.12,
            "min_size": 0.3,
            "max_size": 1.0,
            "num_rects": 4,
        },
        "stepping_stones_terrain": {
            "stone_size": 0.4,
            "stone_distance": 0.1,
            "max_height": 0.10,
            "platform_size": 1.0,
        },
        "gap_terrain": {
            "gap_size": 0.5,
            "platform_size": 2.0,
            "depth": 0.5,
        },
        "pit_terrain": {
            "depth": 0.5,
            "platform_size": 2.0,
        },
        "random_uniform_terrain": {
            "min_height": -0.05,
            "max_height": 0.10,
            "step": 0.005,
            "downsampled_scale": 0.2,
        },
    }

    hf = build_heightfield(
        layout=layout,
        subterrain_size=subterrain_size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        subterrain_parameters=subterrain_parameters,
    )

    # Per-cell difficulty rating in [1.0, 5.0].  Used by
    # ``Go2RoughEnv`` to (1) bias the spawn distribution toward
    # harder cells and (2) weight the forward-progress reward so the
    # policy is rewarded more for advancing on the difficult cells
    # than for cruising on the centre flat cell.  These ratings
    # match the canonical layout:
    #   1.0 = flat / centre (no obstacle, base reward)
    #   2.0 = long flat run-out (low-cost, mostly free)
    #   2.5 = discrete_obstacles / random_uniform (small bumps,
    #         nuisance only)
    #   3.0 = pit_terrain / gap_terrain (one dangerous sinkhole)
    #   3.5 = stepping_stones_terrain (must place feet carefully)
    #   4.5 = up-stairs (must lift feet, balance over slope)
    #   4.5 = down-stairs (must brake, balance over slope)
    #   5.0 = flat-at-top (1.5 m elevation, walking on the plate)
    cell_difficulty = {
        "flat_terrain": 1.0,
        "flat_terrain_at_height": 5.0,
        "random_uniform_terrain": 2.5,
        "discrete_obstacles_terrain": 2.5,
        "stepping_stones_terrain": 3.5,
        "gap_terrain": 3.0,
        "pit_terrain": 3.0,
        "stairs_terrain": 4.5,
        "down_stairs_terrain": 4.5,
        "stairs_terrain_y": 4.5,
        "down_stairs_terrain_y": 4.5,
    }
    difficulty_map = np.array(
        [[cell_difficulty.get(cell, 1.0) for cell in row] for row in layout],
        dtype=np.float32,
    )

    return hf, subterrain_parameters, difficulty_map
