# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
#
# This file is part of AugMPCEnvs and distributed under the General Public License version 2 license.
#
# AugMPCEnvs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# AugMPCEnvs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AugMPCEnvs.  If not, see <http://www.gnu.org/licenses/>.
#
# Backend-neutral terrain generators. Each ``generate_*`` builds a heightfield (+ transform) and
# returns a :class:`TerrainData`; no simulator/USD code here. The heightfield math mirrors the Isaac
# ``RlTerrains.create_*`` methods (kept byte-faithful where they already produce a heightfield), but
# the USD/box authoring is dropped: every backend can build geometry from the heightfield (Genesis
# feeds it straight to ``gs.morphs.Terrain``). ``build_terrain_data`` dispatches by ``ground_type``.
#
# Convention: the terrain is centered on ``center`` (xy); the returned ``TerrainData.position`` is the
# world location of heightfield cell (0,0) (the corner), so it lines up with both Genesis terrain
# placement and the height-grid sensor (see TerrainData docstring).

import numpy as np

from aug_mpc_envs.utils.terrain_primitives import (
    SubTerrain,
    random_uniform_terrain,
    sloped_terrain,
    stairs_terrain,
    pyramid_stairs_terrain,
)
from aug_mpc_envs.utils.terrain_data import TerrainData

_DEFAULT_ORIENT = np.array([1.0, 0.0, 0.0, 0.0])


def _corner(center, terrain_width, terrain_length, base_z=0.0):
    """World position of cell (0,0) for a terrain centered (in xy) on ``center``."""
    center = np.asarray(center, dtype=np.float64)
    return np.array([center[0] - 0.5 * terrain_width,
                     center[1] - 0.5 * terrain_length,
                     base_z + center[2]])


def _slab_and_walls(center, tiled_width, tiled_length, base_z=0.0, with_walls=True,
                    wall_height=2.0, ground_thickness=0.1, wall_thickness=0.1):
    """World-frame boxes [cx,cy,cz,sx,sy,sz] for a base slab (top at ``base_z``) and optional
    perimeter walls, shared by the primitive (box) generators. Mirrors the Isaac *_prim slab/walls."""
    cx, cy = float(center[0]), float(center[1])
    boxes = [[cx, cy, base_z - 0.5 * ground_thickness, tiled_width, tiled_length, ground_thickness]]
    if with_walls:
        wz = base_z + 0.5 * wall_height
        boxes += [
            [cx + 0.5 * tiled_width + 0.5 * wall_thickness, cy, wz, wall_thickness, tiled_length, wall_height],
            [cx - 0.5 * tiled_width - 0.5 * wall_thickness, cy, wz, wall_thickness, tiled_length, wall_height],
            [cx, cy + 0.5 * tiled_length + 0.5 * wall_thickness, wz, tiled_width, wall_thickness, wall_height],
            [cx, cy - 0.5 * tiled_length - 0.5 * wall_thickness, wz, tiled_width, wall_thickness, wall_height],
        ]
    return boxes


# ---------------------------------------------------------------------------------------------------
#  heightfield-trimesh terrains (random / random_patches / slopes / stairs / stepup)
# ---------------------------------------------------------------------------------------------------

def generate_random(terrain_size=8.0, min_height=-0.05, max_height=0.05, step=0.05,
                    downsampled_scale=0.5, horizontal_scale=0.2, vertical_scale=0.005,
                    with_walls=True, wall_height=2.0, center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Random uniform-noise terrain."""
    num_rows = max(2, int(terrain_size / horizontal_scale))
    num_cols = max(2, int(terrain_size / horizontal_scale))
    heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)
    sub = SubTerrain(width=num_rows, length=num_cols,
                     vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    sub = random_uniform_terrain(sub, min_height=min_height, max_height=max_height,
                                 step=step, downsampled_scale=downsampled_scale)
    heightfield[:, :] = sub.height_field_raw
    if with_walls:
        wu = int(wall_height / vertical_scale)
        heightfield[0, :] = wu; heightfield[-1, :] = wu
        heightfield[:, 0] = wu; heightfield[:, -1] = wu
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=_corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale),
                       orientation=_DEFAULT_ORIENT)


def generate_random_patches(terrain_size=8.0, min_height=-0.05, max_height=0.05, step=0.05,
                            downsampled_scale=0.5, patch_ratio=0.3, patch_size=2.0,
                            horizontal_scale=0.08, vertical_scale=0.005,
                            with_walls=True, wall_height=2.0, center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Random uniform terrain with flat square plateaus stamped on top."""
    num_rows = max(2, int(terrain_size / horizontal_scale))
    num_cols = max(2, int(terrain_size / horizontal_scale))
    heightfield = np.zeros((num_rows, num_cols), dtype=np.int16)
    sub = SubTerrain(width=num_rows, length=num_cols,
                     vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    sub = random_uniform_terrain(sub, min_height=min_height, max_height=max_height,
                                 step=step, downsampled_scale=downsampled_scale)
    heightfield[:, :] = sub.height_field_raw
    num_patches_x = max(1, int(terrain_size // patch_size))
    num_patches_y = max(1, int(terrain_size // patch_size))
    patch_cells = max(1, int(patch_size / horizontal_scale))
    for _ in range(int(num_patches_x * num_patches_y * patch_ratio)):
        px = int(np.random.randint(0, num_patches_x) * patch_cells)
        py = int(np.random.randint(0, num_patches_y) * patch_cells)
        ph = np.random.uniform(min_height, max_height)
        heightfield[px:px + patch_cells, py:py + patch_cells] = int(ph / vertical_scale)
    if with_walls:
        wu = int(wall_height / vertical_scale)
        heightfield[0, :] = wu; heightfield[-1, :] = wu
        heightfield[:, 0] = wu; heightfield[:, -1] = wu
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=_corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale),
                       orientation=_DEFAULT_ORIENT)


def generate_slopes(terrain_size=8.0, slope=-0.5, horizontal_scale=0.08, vertical_scale=0.005,
                    center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Single sloped plane."""
    num_rows = max(2, int(terrain_size / horizontal_scale))
    num_cols = max(2, int(terrain_size / horizontal_scale))
    sub = SubTerrain(width=num_rows, length=num_cols,
                     vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    sub = sloped_terrain(sub, slope=slope)
    heightfield = sub.height_field_raw.astype(np.int32)
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=_corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale),
                       orientation=_DEFAULT_ORIENT)


def generate_stairs(terrain_size=8.0, step_width=0.3, step_height=-0.1, platform_size=1.0,
                    horizontal_scale=0.05, vertical_scale=0.005, center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Pyramid stairs."""
    num_rows = max(2, int(terrain_size / horizontal_scale))
    num_cols = max(2, int(terrain_size / horizontal_scale))
    sub = SubTerrain(width=num_rows, length=num_cols,
                     vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
    sub.height_field_raw = np.zeros((num_rows, num_cols), dtype=np.int32)
    sub = pyramid_stairs_terrain(sub, step_width=step_width, step_height=step_height,
                                 platform_size=platform_size)
    heightfield = sub.height_field_raw.astype(np.int32)
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=_corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale),
                       orientation=_DEFAULT_ORIENT)


def generate_stepup(terrain_size=8.0, stairs_ratio=0.3, min_steps=1, max_steps=1,
                    pyramid_platform_size=3.0, step_height=0.15, patch_size=3.0,
                    res_low=0.1, res_high=0.03, vertical_scale=0.005,
                    perimeter_wall_height=2.0, center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Tiled patches of straight/pyramid stairs (heightfield port of create_stepup_terrain)."""
    use_high_res = (stairs_ratio > 0.0) and (abs(int(min_steps)) > 0 or abs(int(max_steps)) > 0)
    horizontal_scale = res_high if use_high_res else res_low
    num_rows = max(1, int(round(terrain_size / horizontal_scale)))
    num_cols = max(1, int(round(terrain_size / horizontal_scale)))
    heightfield = np.zeros((num_rows, num_cols), dtype=np.int32)
    patch_rows = max(1, int(round(patch_size / horizontal_scale)))
    patch_cols = patch_rows
    n_patches_x = max(1, int(np.ceil(num_rows / patch_rows)))
    n_patches_y = max(1, int(np.ceil(num_cols / patch_cols)))
    for px in range(n_patches_x):
        for py in range(n_patches_y):
            ix = px * patch_rows; iy = py * patch_cols
            cur_rows = min(patch_rows, num_rows - ix)
            cur_cols = min(patch_cols, num_cols - iy)
            sub = SubTerrain(width=cur_rows, length=cur_cols,
                             vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
            sub.height_field_raw = np.zeros((sub.width, sub.length), dtype=np.int32)
            abs_min = max(1, abs(int(min_steps)))
            abs_max = max(abs_min, abs(int(max_steps)))
            steps_count = int(np.random.randint(abs_min, abs_max + 1))
            if np.random.rand() < stairs_ratio:
                step_width_cells = max(1, cur_rows // max(1, steps_count))
                max_fit_steps = max(1, cur_rows // step_width_cells)
                steps_count = max(1, min(steps_count, max_fit_steps))
                step_width_m = step_width_cells * horizontal_scale
                step_h_units = max(1, int(round(abs(step_height) / vertical_scale)))
                if min_steps < 0:
                    for i in range(steps_count):
                        start = i * step_width_cells
                        stop = min(sub.width, (i + 1) * step_width_cells)
                        h_units = -(steps_count - i) * step_h_units
                        sub.height_field_raw[start:stop, :] += h_units
                else:
                    sub = stairs_terrain(sub, step_width=step_width_m, step_height=step_height)
            else:
                platform_cells = max(1, int(round(pyramid_platform_size / horizontal_scale)))
                denom = max(1, 2 * steps_count)
                step_width_cells = max(1, (min(cur_rows, cur_cols) - platform_cells) // denom)
                step_width_m = step_width_cells * horizontal_scale
                min_dim = min(cur_rows, cur_cols)
                max_possible_steps = (min_dim - platform_cells) // (2 * max(1, step_width_cells))
                if max_possible_steps > 0:
                    steps_count = min(steps_count, max_possible_steps)
                    step_h_units = max(1, int(round(abs(step_height) / vertical_scale)))
                    if min_steps < 0:
                        start_x = 0; stop_x = sub.width; start_y = 0; stop_y = sub.length
                        for k in range(steps_count):
                            start_x += step_width_cells
                            stop_x = max(start_x, stop_x - step_width_cells)
                            start_y += step_width_cells
                            stop_y = max(start_y, stop_y - step_width_cells)
                            if start_x >= stop_x or start_y >= stop_y:
                                break
                            h_units = -(steps_count - k) * step_h_units
                            sub.height_field_raw[start_x:stop_x, start_y:stop_y] = h_units
                    else:
                        sub = pyramid_stairs_terrain(sub, step_width=step_width_m,
                                                     step_height=step_height,
                                                     platform_size=pyramid_platform_size)
            heightfield[ix:ix + sub.height_field_raw.shape[0],
                        iy:iy + sub.height_field_raw.shape[1]] = sub.height_field_raw
    wall_units = max(1, int(round(float(perimeter_wall_height) / vertical_scale)))
    heightfield[0, :] = np.maximum(heightfield[0, :], wall_units)
    heightfield[-1, :] = np.maximum(heightfield[-1, :], wall_units)
    heightfield[:, 0] = np.maximum(heightfield[:, 0], wall_units)
    heightfield[:, -1] = np.maximum(heightfield[:, -1], wall_units)
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=_corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale),
                       orientation=_DEFAULT_ORIENT)


# ---------------------------------------------------------------------------------------------------
#  primitive-style terrains (heightfield port of the *_prim Isaac terrains)
# ---------------------------------------------------------------------------------------------------

def generate_stepup_prim(terrain_size=8.0, stairs_ratio=0.9, platform_size=3.0,
                         step_height_lb=0.08, step_height_ub=0.15, n_steps=1, area_factor=0.7,
                         min_step_width=None, max_step_width=None, wall_height=2.0,
                         res_low=0.1, res_high=0.03, vertical_scale=0.005,
                         random_n_steps=False, center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Heightfield port of create_stepup_prim_terrain (raised tiles / pyramids, no per-box authoring)."""
    import math
    use_high_res = (stairs_ratio > 0.0) or (n_steps > 1)
    horizontal_scale = res_high if use_high_res else res_low
    terrain_width = float(terrain_size); terrain_length = float(terrain_size)
    num_rows = max(1, int(round(terrain_width / horizontal_scale)))
    num_cols = max(1, int(round(terrain_length / horizontal_scale)))
    heightfield = np.zeros((num_rows, num_cols), dtype=np.int32)
    n_tiles_x = max(1, int(np.ceil(terrain_width / platform_size)))
    n_tiles_y = max(1, int(np.ceil(terrain_length / platform_size)))
    step_height_lb = max(step_height_lb, 0.0)
    step_height_ub = max(step_height_ub, step_height_lb)
    wall_h_units = max(1, int(round(wall_height / vertical_scale)))
    area_factor = max(min(area_factor, 0.9999), 1e-3)
    corner = _corner(center, num_rows * horizontal_scale, num_cols * horizontal_scale)
    low_x, low_y, base_z = float(corner[0]), float(corner[1]), float(corner[2])
    boxes = []  # raised tiles/levels as primitive box colliders (world coords)
    for ix in range(n_tiles_x):
        for iy in range(n_tiles_y):
            start_x_m = ix * platform_size
            end_x_m = min(terrain_width, (ix + 1) * platform_size)
            start_y_m = iy * platform_size
            end_y_m = min(terrain_length, (iy + 1) * platform_size)
            start_row = int(round(start_x_m / horizontal_scale))
            end_row = min(num_rows, int(round(end_x_m / horizontal_scale)))
            start_col = int(round(start_y_m / horizontal_scale))
            end_col = min(num_cols, int(round(end_y_m / horizontal_scale)))
            if np.random.rand() >= stairs_ratio:
                continue
            size_x = end_x_m - start_x_m; size_y = end_y_m - start_y_m
            if size_x <= 0.0 or size_y <= 0.0:
                continue
            if n_steps <= 1:
                step_height = np.random.uniform(step_height_lb, step_height_ub)
                step_h_units = max(1, int(round(step_height / vertical_scale)))
                heightfield[start_row:end_row, start_col:end_col] = step_h_units
                boxes.append([low_x + start_x_m + 0.5 * size_x, low_y + start_y_m + 0.5 * size_y,
                              base_z + 0.5 * step_height, size_x, size_y, step_height])
            else:
                shrink_factor = math.sqrt(area_factor)
                steps_for_tile = np.random.randint(1, n_steps + 1) if random_n_steps else n_steps
                accumulated_units = 0
                accumulated_height = 0.0
                curr_start_x = start_x_m; curr_start_y = start_y_m
                curr_size_x = size_x; curr_size_y = size_y
                use_step_width = (min_step_width is not None and max_step_width is not None)
                if use_step_width:
                    min_w = max(0.0, float(min_step_width)); max_w = max(min_w, float(max_step_width))
                for level in range(steps_for_tile):
                    if use_step_width:
                        stair_w = np.random.uniform(min_w, max_w)
                        level_size_x = curr_size_x - 2.0 * stair_w
                        level_size_y = curr_size_y - 2.0 * stair_w
                    else:
                        level_size_x = size_x * (shrink_factor ** level)
                        level_size_y = size_y * (shrink_factor ** level)
                    if level_size_x < horizontal_scale or level_size_y < horizontal_scale:
                        break
                    level_height = np.random.uniform(step_height_lb, step_height_ub)
                    level_units = max(1, int(round(level_height / vertical_scale)))
                    if use_step_width:
                        level_start_x = curr_start_x + stair_w
                        level_start_y = curr_start_y + stair_w
                    else:
                        level_start_x = start_x_m + 0.5 * (size_x - level_size_x)
                        level_start_y = start_y_m + 0.5 * (size_y - level_size_y)
                    level_end_x = level_start_x + level_size_x
                    level_end_y = level_start_y + level_size_y
                    lsr = int(round(level_start_x / horizontal_scale))
                    ler = min(num_rows, int(round(level_end_x / horizontal_scale)))
                    lsc = int(round(level_start_y / horizontal_scale))
                    lec = min(num_cols, int(round(level_end_y / horizontal_scale)))
                    if ler <= lsr or lec <= lsc:
                        continue
                    accumulated_units += level_units
                    accumulated_height += level_height
                    heightfield[lsr:ler, lsc:lec] = accumulated_units
                    # each level is a box from base_z up to the accumulated height, centered on the level
                    boxes.append([low_x + level_start_x + 0.5 * level_size_x,
                                  low_y + level_start_y + 0.5 * level_size_y,
                                  base_z + accumulated_height - 0.5 * level_height,
                                  level_size_x, level_size_y, level_height])
                    if use_step_width:
                        curr_start_x = level_start_x; curr_start_y = level_start_y
                        curr_size_x = level_size_x; curr_size_y = level_size_y
    heightfield[0, :] = wall_h_units; heightfield[-1, :] = wall_h_units
    heightfield[:, 0] = wall_h_units; heightfield[:, -1] = wall_h_units
    boxes = _slab_and_walls(center, num_rows * horizontal_scale, num_cols * horizontal_scale,
                            base_z=base_z, with_walls=True, wall_height=wall_height) + boxes
    return TerrainData(heightfield, horizontal_scale, vertical_scale,
                       position=corner, orientation=_DEFAULT_ORIENT, boxes=np.asarray(boxes))


def generate_random_tiles(terrain_size=8.0, cell_size=0.5, height_lb=0.0, height_ub=0.06,
                          patch_ratio=0.0, patch_size=3.0, min_tile_height=0.01,
                          with_walls=True, wall_height=2.0, vertical_scale=0.005,
                          center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Heightfield port of create_random_tiles_prim_terrain (dense grid of small flat-topped cells)."""
    cell_size = max(float(cell_size), 1e-3)
    terrain_width = float(terrain_size); terrain_length = float(terrain_size)
    n_cells_x = max(1, int(round(terrain_width / cell_size)))
    n_cells_y = max(1, int(round(terrain_length / cell_size)))
    height_lb = max(0.0, float(height_lb))
    height_ub = max(height_lb, float(height_ub))
    min_tile_height = max(0.0, float(min_tile_height))
    cell_h = np.random.uniform(height_lb, height_ub, size=(n_cells_x, n_cells_y))
    if patch_ratio > 0.0:
        patch_cells = max(1, int(round(float(patch_size) / cell_size)))
        n_slots = max(1, (n_cells_x // patch_cells)) * max(1, (n_cells_y // patch_cells))
        n_patches = int(round(n_slots * float(patch_ratio)))
        for _ in range(n_patches):
            sx = np.random.randint(0, n_cells_x); sy = np.random.randint(0, n_cells_y)
            ex = min(n_cells_x, sx + patch_cells); ey = min(n_cells_y, sy + patch_cells)
            cell_h[sx:ex, sy:ey] = np.random.uniform(height_lb, height_ub)
    # cells below min_tile_height read as flat ground (matches the Isaac box-skip behavior).
    cell_h[cell_h < min_tile_height] = 0.0
    heightfield = np.rint(cell_h / vertical_scale).astype(np.int32)
    if with_walls:
        wu = max(1, int(round(wall_height / vertical_scale)))
        heightfield[0, :] = wu; heightfield[-1, :] = wu
        heightfield[:, 0] = wu; heightfield[:, -1] = wu
    # heightfield grid sample (ix,iy) is the cell center; horizontal_scale == cell_size, and the
    # corner (cell 0,0) is half a cell in from the centered low corner.
    tiled_width = n_cells_x * cell_size; tiled_length = n_cells_y * cell_size
    low = _corner(center, tiled_width, tiled_length)
    base_z = float(low[2])
    position = np.array([low[0] + 0.5 * cell_size, low[1] + 0.5 * cell_size, base_z])
    # primitive box colliders: a base slab + one box per raised cell + perimeter walls (Isaac-style)
    boxes = _slab_and_walls(center, tiled_width, tiled_length, base_z=base_z,
                            with_walls=with_walls, wall_height=wall_height)
    raised = np.argwhere(cell_h > 0.0)
    for ix, iy in raised:
        h = float(cell_h[ix, iy])
        boxes.append([float(low[0] + (ix + 0.5) * cell_size), float(low[1] + (iy + 0.5) * cell_size),
                      base_z + 0.5 * h, cell_size, cell_size, h])
    return TerrainData(heightfield, cell_size, vertical_scale,
                       position=position, orientation=_DEFAULT_ORIENT, boxes=np.asarray(boxes))


# ---------------------------------------------------------------------------------------------------
#  dispatch
# ---------------------------------------------------------------------------------------------------

def build_terrain_data(ground_type: str, ground_size: float, opts: dict,
                       center=(0.0, 0.0, 0.0)) -> TerrainData:
    """Build a TerrainData for ``ground_type`` from an env-opts dict (keys mirror the Isaac opts).

    Returns ``None`` for ``flat`` (caller should use a plane). Raises ValueError for unknown types.
    """
    gt = (ground_type or "flat").lower()
    if gt == "flat":
        return None
    if gt == "random":
        dh = float(opts.get("dh_ground", 0.05))
        return generate_random(terrain_size=ground_size, min_height=-dh, max_height=dh, step=max(2 * dh, 1e-3),
                               with_walls=opts.get("terrain_walls", True),
                               wall_height=float(opts.get("wall_height", 2.0)), center=center)
    if gt == "random_patches":
        dh = float(opts.get("dh_ground", 0.05))
        return generate_random_patches(terrain_size=ground_size, min_height=-dh, max_height=dh,
                                       step=max(2 * dh, 1e-3),
                                       patch_ratio=float(opts.get("tile_patch_ratio", 0.3)),
                                       patch_size=float(opts.get("tile_patch_size", 2.0)),
                                       with_walls=opts.get("terrain_walls", True),
                                       wall_height=float(opts.get("wall_height", 2.0)), center=center)
    if gt == "slopes":
        return generate_slopes(terrain_size=ground_size, slope=float(opts.get("slope", -0.5)), center=center)
    if gt == "stairs":
        return generate_stairs(terrain_size=ground_size,
                               step_width=float(opts.get("stairs_step_width", 0.3)),
                               step_height=float(opts.get("stairs_step_height", -0.1)),
                               center=center)
    if gt == "stepup":
        return generate_stepup(terrain_size=ground_size,
                               stairs_ratio=float(opts.get("step_stairs_ratio", 0.3)),
                               min_steps=int(opts.get("step_min", 1)),
                               max_steps=int(opts.get("step_max", 1)),
                               pyramid_platform_size=float(opts.get("step_platform_size", 3.0)),
                               step_height=float(opts.get("step_height_ub", 0.15)),
                               center=center)
    if gt == "stepup_prim":
        return generate_stepup_prim(terrain_size=ground_size,
                                    stairs_ratio=float(opts.get("step_stairs_ratio", 0.9)),
                                    platform_size=float(opts.get("step_platform_size", 3.0)),
                                    step_height_lb=float(opts.get("step_height_lb", 0.08)),
                                    step_height_ub=float(opts.get("step_height_ub", 0.15)),
                                    n_steps=int(opts.get("step_n", 1)),
                                    area_factor=float(opts.get("step_area_factor", 0.7)),
                                    min_step_width=opts.get("step_width_lb", None),
                                    max_step_width=opts.get("step_width_ub", None),
                                    random_n_steps=bool(opts.get("step_random_n_steps", False)),
                                    center=center)
    if gt == "random_tiles":
        return generate_random_tiles(terrain_size=ground_size,
                                     cell_size=float(opts.get("tile_cell_size", 0.5)),
                                     height_lb=float(opts.get("tile_height_lb", 0.0)),
                                     height_ub=float(opts.get("tile_height_ub", 0.06)),
                                     patch_ratio=float(opts.get("tile_patch_ratio", 0.0)),
                                     patch_size=float(opts.get("tile_patch_size", 3.0)),
                                     min_tile_height=float(opts.get("tile_min_height", 0.01)),
                                     with_walls=opts.get("terrain_walls", True),
                                     wall_height=float(opts.get("wall_height", 2.0)),
                                     center=center)
    raise ValueError(f"Unknown ground_type '{ground_type}'")
