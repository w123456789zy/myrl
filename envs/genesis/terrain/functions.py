"""Custom terrain functions and a unified factory :func:`make_subterrain`.

This module is **fully self-contained** — we do NOT import from
``genesis.ext.isaacgym.terrain_utils`` so the project remains portable
across Genesis versions (the upstream module's API has shifted: the
official Genesis release does not export ``gap_terrain``,
``pit_terrain``, ``flat_terrain_at_height`` or ``down_stairs_terrain``,
and they vary between point releases).

All terrain functions used by go2-rough are re-implemented here, in
the upstream "legged-gym" style: each mutates a :class:`SubTerrain`
in place.  The signatures match the upstream API where it exists, so
swapping the import back later is a one-line change.

Local-only additions vs upstream:
  * ``stairs_terrain_y``    — staircase that ramps in **+y** (world).
  * ``down_stairs_terrain_y``— descending staircase in +y.
  * Optional ``num_steps`` for both: stops the staircase partway and
    fills the rest of the cell with a plateau/flat.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from .subterrain import SubTerrain


# ===========================================================================
# 1. Locally-added +y staircase (the only pieces of terrain logic we own)
# ===========================================================================

def stairs_terrain_y(
    terrain: SubTerrain,
    step_width: float,
    step_height: float,
    num_steps: int | None = None,
) -> SubTerrain:
    """Ascending staircase that ramps in the **+y** (world) direction.

    In :func:`build_heightfield`, the first axis of ``height_field_raw``
    maps to world +y, so this staircase is correctly oriented for a dog
    walking +y from the centre cell.

    When ``num_steps`` is given and less than the cell capacity, the
    remaining rows are filled with a flat **plateau** at the
    top-of-stairs height.
    """
    step_width_d = int(step_width / terrain.horizontal_scale)
    step_height_d = int(step_height / terrain.vertical_scale)

    if num_steps is None:
        num_steps = terrain.width // step_width_d

    height = 0
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width_d : (i + 1) * step_width_d, :] += height
        height += step_height_d

    # Fill unwritten rows with the top-of-stairs plateau.
    plateau_start = num_steps * step_width_d
    if plateau_start < terrain.width:
        terrain.height_field_raw[plateau_start:, :] += height

    return terrain


def down_stairs_terrain_y(
    terrain: SubTerrain,
    step_width: float,
    step_height: float,
    platform_size: float = 0.0,
    base_height: float = 0.0,
    num_steps: int | None = None,
) -> SubTerrain:
    """Descending staircase that ramps in the **+y** (world) direction.

    Pairs with :func:`stairs_terrain_y` to form a continuous
    ``up → plateau → down → flat`` sequence.

    When ``num_steps`` is given, only that many steps are created and
    the rest of the cell stays at ground level (0), providing a safe
    flat run-out after the descent.
    """
    step_width_d = int(step_width / terrain.horizontal_scale)
    step_height_d = int(step_height / terrain.vertical_scale)  # positive
    platform_steps = int(platform_size / terrain.horizontal_scale)
    base = int(base_height / terrain.vertical_scale)

    if num_steps is None:
        num_steps = terrain.width // step_width_d

    height = base
    for i in range(num_steps):
        y_lo = i * step_width_d
        y_hi = (i + 1) * step_width_d
        if platform_steps > 0 and i < platform_steps // step_width_d:
            terrain.height_field_raw[y_lo:y_hi, :] += base
            continue
        terrain.height_field_raw[y_lo:y_hi, :] += height
        height -= step_height_d

    return terrain


# ===========================================================================
# 2. Self-contained re-implementations of the rest of the legged-gym
#    terrain family.  These are simple enough to be inlined, and
#    keeping them local means we don't depend on whatever subset of
#    them happens to ship with the installed Genesis version.
# ===========================================================================

def _flat_terrain_impl(terrain: SubTerrain, **_) -> SubTerrain:
    """Identity: height_field_raw is already all zeros."""
    return terrain


def _flat_terrain_at_height_impl(terrain: SubTerrain, height: float = 0.0, **_) -> SubTerrain:
    """Flat terrain raised/lowered to a specific world height."""
    h = int(height / terrain.vertical_scale)
    terrain.height_field_raw[:] = h
    return terrain


def _stairs_terrain_impl(
    terrain: SubTerrain,
    step_width: float,
    step_height: float,
    num_steps: int | None = None,
    **_,
) -> SubTerrain:
    """Staircase ramping along the first axis of height_field_raw (+y world).

    Identical algorithm to :func:`stairs_terrain_y` (the +y version
    lives in its own function so the layout can reference it by
    name without ambiguity).
    """
    return stairs_terrain_y(terrain, step_width, step_height, num_steps=num_steps)


def _down_stairs_terrain_impl(
    terrain: SubTerrain,
    step_width: float,
    step_height: float,
    platform_size: float = 0.0,
    base_height: float = 0.0,
    num_steps: int | None = None,
    **_,
) -> SubTerrain:
    """Descending staircase ramping along the first axis (+y world)."""
    return down_stairs_terrain_y(
        terrain,
        step_width,
        step_height,
        platform_size=platform_size,
        base_height=base_height,
        num_steps=num_steps,
    )


def _discrete_obstacles_terrain_impl(
    terrain: SubTerrain,
    max_height: float = 0.05,
    min_size: float = 1.0,
    max_size: float = 5.0,
    num_rects: int = 20,
    platform_size: float = 1.0,
    **_,
) -> SubTerrain:
    """Terrain with randomly placed rectangular box obstacles.

    A flat square platform of size ``platform_size`` is preserved at
    the centre of the cell so the dog has a clear entry/exit zone.
    """
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    grid_size_x, grid_size_y = terrain.height_field_raw.shape
    width_choices = np.arange(min(min_size, grid_size_x - 1), min(max_size, grid_size_x - 1) + 1, 4)
    length_choices = np.arange(min(min_size, grid_size_y - 1), min(max_size, grid_size_y - 1) + 1, 4)
    height_choices = [-max_height, -max_height // 2, max_height // 2, max_height]

    for _ in range(num_rects):
        width = np.random.choice(width_choices)
        length = np.random.choice(length_choices)
        height = np.random.choice(height_choices)
        start_x = np.random.choice(range(0, grid_size_x - width, 4))
        start_y = np.random.choice(range(0, grid_size_y - length, 4))
        terrain.height_field_raw[start_x : start_x + width, start_y : start_y + length] = height

    # Re-zero the central platform (the random obstacles may have
    # written over it).
    start_x = (terrain.width - platform_size) // 2
    end_x = (terrain.width + platform_size) // 2
    start_y = (terrain.length - platform_size) // 2
    end_y = (terrain.length + platform_size) // 2
    terrain.height_field_raw[start_x:end_x, start_y:end_y] = 0
    return terrain


def _stepping_stones_terrain_impl(
    terrain: SubTerrain,
    stone_size: float = 1.0,
    stone_distance: float = 0.25,
    max_height: float = 0.2,
    platform_size: float = 1.0,
    depth: float = -10.0,
    **_,
) -> SubTerrain:
    """Stepping-stones / "meihua-pile" terrain with random-height stones.

    The cell is filled with deep "holes" and a regular grid of stone
    tiles is dropped on top with random heights in
    ``[-max_height, +max_height]``.  A flat platform is kept at the
    centre.
    """
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height - 1, max_height, step=1)

    # Fill the whole cell with the deep-hole floor.
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)

    if terrain.length >= terrain.width:
        start_y = 0
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # Fill the first (partial) hole with a random stone.
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0:stop_x, start_y:stop_y] = np.random.choice(height_range)
            # Then drop stones across the rest of the row.
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    else:  # width > length — symmetric case
        start_x = 0
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x:stop_x, 0:stop_y] = np.random.choice(height_range)
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    # Central platform.
    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain


def _gap_terrain_impl(
    terrain: SubTerrain,
    gap_size: float = 0.5,
    platform_size: float = 1.0,
    depth: float = 0.5,
    **_,
) -> SubTerrain:
    """Terrain with a square gap (hole below ground) surrounded by platform.

    The robot must JUMP over the gap to cross the tile.
    """
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    depth = int(depth / terrain.vertical_scale)  # positive

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size

    gap_depth = -depth
    # Carve the full square first…
    terrain.height_field_raw[center_x - x2 : center_x + x2, center_y - y2 : center_y + y2] = gap_depth
    # …then re-zero the platform ring inside that square.
    terrain.height_field_raw[center_x - x1 : center_x + x1, center_y - y1 : center_y + y1] = 0
    return terrain


def _pit_terrain_impl(
    terrain: SubTerrain,
    depth: float = 0.5,
    platform_size: float = 1.0,
    **_,
) -> SubTerrain:
    """Terrain with a square deep pit in the middle.

    The robot FALLS into the pit if it steps in.
    """
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)

    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth
    return terrain


def _random_uniform_terrain_impl(
    terrain: SubTerrain,
    min_height: float = -0.1,
    max_height: float = 0.1,
    step: float = 0.1,
    downsampled_scale: float | None = None,
    **_,
) -> SubTerrain:
    """Smooth random-bump terrain (uniform noise on a coarse grid, then
    bicubic-spline upsampled to the full resolution).
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale
    # 1) sample random uniform heights on a coarse grid
    scaled_width = terrain.width * terrain.horizontal_scale
    scaled_length = terrain.length * terrain.horizontal_scale
    min_h = int(min_height / terrain.vertical_scale)
    max_h = int(max_height / terrain.vertical_scale)
    step_h = int(step / terrain.vertical_scale)
    heights_range = np.arange(min_h, max_h + step_h, step=step_h)
    hf_downsampled = np.random.choice(
        heights_range,
        (
            max(1, int(scaled_width / downsampled_scale)),
            max(1, int(scaled_length / downsampled_scale)),
        ),
    )

    # 2) bicubic upsample to the full resolution.  We don't import
    #    genesis.utils.geom here — we just use simple numpy bilinear
    #    which is good enough for a heightmap that's later read at
    #    0.1 m horizontal resolution.  For most cases this produces
    #    visually-identical terrain to the upstream cubic spline.
    out = np.zeros((terrain.width, terrain.length), dtype=terrain.height_field_raw.dtype)
    src_rows, src_cols = hf_downsampled.shape
    for r in range(terrain.width):
        sr = r * (src_rows - 1) / max(1, terrain.width - 1)
        r0, r1 = int(np.floor(sr)), min(int(np.ceil(sr)), src_rows - 1)
        fr = sr - r0
        for c in range(terrain.length):
            sc = c * (src_cols - 1) / max(1, terrain.length - 1)
            c0, c1 = int(np.floor(sc)), min(int(np.ceil(sc)), src_cols - 1)
            fc = sc - c0
            v00 = hf_downsampled[r0, c0]
            v01 = hf_downsampled[r0, c1]
            v10 = hf_downsampled[r1, c0]
            v11 = hf_downsampled[r1, c1]
            out[r, c] = (
                (1 - fr) * (1 - fc) * v00
                + (1 - fr) * fc * v01
                + fr * (1 - fc) * v10
                + fr * fc * v11
            )

    terrain.height_field_raw += np.rint(out).astype(terrain.height_field_raw.dtype)
    return terrain


# ===========================================================================
# 3. Public registry: name -> factory callable
# ===========================================================================

TERRAIN_FUNCTIONS: Dict[str, Callable[..., SubTerrain]] = {
    "flat_terrain": _flat_terrain_impl,
    "flat_terrain_at_height": _flat_terrain_at_height_impl,
    "stairs_terrain": _stairs_terrain_impl,
    "down_stairs_terrain": _down_stairs_terrain_impl,
    "stairs_terrain_y": stairs_terrain_y,
    "down_stairs_terrain_y": down_stairs_terrain_y,
    "discrete_obstacles_terrain": _discrete_obstacles_terrain_impl,
    "stepping_stones_terrain": _stepping_stones_terrain_impl,
    "gap_terrain": _gap_terrain_impl,
    "pit_terrain": _pit_terrain_impl,
    "random_uniform_terrain": _random_uniform_terrain_impl,
}


def make_subterrain(
    terrain_type: str,
    width: int,
    length: int,
    horizontal_scale: float,
    vertical_scale: float,
    **params,
) -> SubTerrain:
    """Build a single sub-terrain by name.

    Parameters:
        terrain_type:     one of the keys in :data:`TERRAIN_FUNCTIONS`.
        width / length:   cell counts (width = first axis = world x,
                          length = second axis = world y).
        horizontal_scale: meters per cell in x/y.
        vertical_scale:   meters per cell in z.
        **params:         per-terrain-type parameters.
    """
    fn = TERRAIN_FUNCTIONS.get(terrain_type)
    if fn is None:
        raise ValueError(
            f"Unknown terrain type '{terrain_type}'. "
            f"Available: {sorted(TERRAIN_FUNCTIONS)}."
        )
    sub = SubTerrain(
        width=width,
        length=length,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
    )
    return fn(sub, **params)
