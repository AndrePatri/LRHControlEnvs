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
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import numpy as np
import math

from aug_mpc_envs.utils.terrain_utils_isaac5x import *

from pxr import Usd, UsdGeom, Gf, UsdPhysics, PhysxSchema

class RlTerrains():

    def __init__(self,
                stage: Usd.Stage,
                prim_path: str = "/World/terrain"):

        self._stage = stage

        self._prim_path=prim_path

        # cache heightfield info to allow querying terrain height at world positions
        self.heightfield_world = None  # meters
        self._heightfield_raw = None    # original integer grid
        self._horizontal_scale = None
        self._vertical_scale = None
        self.position = None
        self.orientation = None

    def create_wave_terrain(self,
            terrain_size = 40,
            num_waves = 10,
            amplitude = 1,
            position = np.array([0.0, 0.0, 0.0]),
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1):

        # creates a terrain
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.08  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows,
                                num_cols), dtype=np.int16)

        def new_sub_terrain():

            return SubTerrain(width=num_rows,
                        length=num_cols,
                        vertical_scale=vertical_scale,
                        horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=num_waves,
                                                    amplitude=amplitude).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale,
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, -terrain_length/2.0, 0]) + position

        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        terrain_prim = add_terrain_to_stage(stage=self._stage,
                    vertices=vertices,
                    triangles=triangles,
                    position=position,
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def create_stepup_prim_terrain(self,
                    terrain_size=30,
                    stairs_ratio: float = 0.2,
                    platform_size: float = 5.0,
                    step_height_lb: float = 0.05,
                    step_height_ub: float = 0.15,
                    n_steps: int = 1,
                    area_factor: float = 0.5,
                    min_step_width: float = None,
                    max_step_width: float = None,
                    wall_height: float = 2.0,
                    res_low: float = 0.1,
                    res_high: float = 0.03,
                    position = np.array([0.0, 0.0, 0.0]),
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    restitution=0.1,
                    random_n_steps: bool = True):
        """
        Create a random tiled step-up terrain using only primitive colliders (boxes + base slab).
        Each tile of size ``platform_size`` has probability ``stairs_ratio`` of being raised
        by ``step_height``. A matching synthetic heightfield is generated for sensing.
        If min/max step width are provided, each successive platform shrinks by
        2 * sampled step width instead of using area_factor.
        """
        # Choose resolution: use high resolution when we expect steps
        use_high_res = (stairs_ratio > 0.0) or (n_steps > 1)
        horizontal_scale = res_high if use_high_res else res_low
        vertical_scale = 0.005

        terrain_width = terrain_size
        terrain_length = terrain_size

        num_rows = max(1, int(round(terrain_width / horizontal_scale)))
        num_cols = max(1, int(round(terrain_length / horizontal_scale)))

        heightfield = np.zeros((num_rows, num_cols), dtype=np.int32)

        # Tile layout based on metric platform size to keep colliders within bounds
        n_tiles_x = max(1, int(np.ceil(terrain_width / platform_size)))
        n_tiles_y = max(1, int(np.ceil(terrain_length / platform_size)))

        terrain_center = position

        step_height_lb = max(step_height_lb, 0.0)
        step_height_ub = max(step_height_ub, step_height_lb)
        wall_h_units = max(1, int(round(wall_height / vertical_scale)))
        ground_thickness = 0.1
        ground_top = terrain_center[2] + 0.5 * ground_thickness

        terrain_low_corner = position + np.array([-terrain_width/2.0, -terrain_length/2.0, 0.0])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # base slab
        ground_prim_path = self._prim_path + "_slab"
        ground_prim = UsdGeom.Cube.Define(self._stage, ground_prim_path)
        ground_prim.CreateSizeAttr(1.0)
        # scale then translate so translation is not scaled
        ground_prim.AddTranslateOp().Set(Gf.Vec3f(terrain_center[0],
                                                  terrain_center[1],
                                                  terrain_center[2]))
        ground_prim.AddScaleOp().Set(Gf.Vec3f(terrain_width, terrain_length, ground_thickness))
        UsdPhysics.CollisionAPI.Apply(ground_prim.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(ground_prim.GetPrim())
        mat_api = UsdPhysics.MaterialAPI.Apply(ground_prim.GetPrim())
        mat_api.CreateDynamicFrictionAttr(dynamic_friction)
        mat_api.CreateStaticFrictionAttr(static_friction)
        mat_api.CreateRestitutionAttr(restitution)

        # Raised tiles / pyramid
        area_factor = max(min(area_factor, 0.9999), 1e-3)  # keep meaningful decay
        for ix in range(n_tiles_x):
            for iy in range(n_tiles_y):
                # tile extents in meters
                start_x_m = ix * platform_size
                end_x_m = min(terrain_width, (ix + 1) * platform_size)
                start_y_m = iy * platform_size
                end_y_m = min(terrain_length, (iy + 1) * platform_size)

                # convert to heightfield indices
                start_row = int(round(start_x_m / horizontal_scale))
                end_row = min(num_rows, int(round(end_x_m / horizontal_scale)))
                start_col = int(round(start_y_m / horizontal_scale))
                end_col = min(num_cols, int(round(end_y_m / horizontal_scale)))
                cells_x = max(1, end_row - start_row)
                cells_y = max(1, end_col - start_col)

                if np.random.rand() >= stairs_ratio:
                    continue  # keep this tile flat

                size_x = end_x_m - start_x_m
                size_y = end_y_m - start_y_m
                if size_x <= 0.0 or size_y <= 0.0:
                    continue

                tile_center_x = terrain_low_corner[0] + start_x_m + 0.5 * size_x
                tile_center_y = terrain_low_corner[1] + start_y_m + 0.5 * size_y

                if n_steps <= 1:
                    # single raised tile
                    step_height = np.random.uniform(step_height_lb, step_height_ub)
                    step_h_units = max(1, int(round(step_height / vertical_scale)))
                    heightfield[start_row:end_row, start_col:end_col] = step_h_units
                    center_z = ground_top + 0.5 * step_height
                    tile_prim_path = f"{self._prim_path}/tile_{ix}_{iy}"
                    tile_prim = UsdGeom.Cube.Define(self._stage, tile_prim_path)
                    tile_prim.CreateSizeAttr(1.0)
                    tile_prim.AddTranslateOp().Set(Gf.Vec3f(tile_center_x, tile_center_y, center_z))
                    tile_prim.AddScaleOp().Set(Gf.Vec3f(size_x, size_y, step_height))
                    UsdPhysics.CollisionAPI.Apply(tile_prim.GetPrim())
                    PhysxSchema.PhysxCollisionAPI.Apply(tile_prim.GetPrim())
                    tile_mat = UsdPhysics.MaterialAPI.Apply(tile_prim.GetPrim())
                    tile_mat.CreateDynamicFrictionAttr(dynamic_friction)
                    tile_mat.CreateStaticFrictionAttr(static_friction)
                    tile_mat.CreateRestitutionAttr(restitution)
                else:
                    shrink_factor = math.sqrt(area_factor)
                    if random_n_steps:
                        steps_for_tile = np.random.randint(1, n_steps + 1)
                    else:
                        steps_for_tile = n_steps
                    accumulated_height = 0.0
                    accumulated_units = 0
                    curr_start_x = start_x_m
                    curr_start_y = start_y_m
                    curr_size_x = size_x
                    curr_size_y = size_y
                    use_step_width = (min_step_width is not None and max_step_width is not None)
                    if use_step_width:
                        min_w = max(0.0, float(min_step_width))
                        max_w = max(min_w, float(max_step_width))
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

                        # sample height for this level
                        level_height = np.random.uniform(step_height_lb, step_height_ub)
                        level_units = max(1, int(round(level_height / vertical_scale)))

                        # center the reduced platform within the tile bounds
                        if use_step_width:
                            level_start_x = curr_start_x + stair_w
                            level_start_y = curr_start_y + stair_w
                        else:
                            level_start_x = start_x_m + 0.5 * (size_x - level_size_x)
                            level_start_y = start_y_m + 0.5 * (size_y - level_size_y)
                        level_end_x = level_start_x + level_size_x
                        level_end_y = level_start_y + level_size_y

                        level_start_row = int(round(level_start_x / horizontal_scale))
                        level_end_row = min(num_rows, int(round(level_end_x / horizontal_scale)))
                        level_start_col = int(round(level_start_y / horizontal_scale))
                        level_end_col = min(num_cols, int(round(level_end_y / horizontal_scale)))
                        if level_end_row <= level_start_row or level_end_col <= level_start_col:
                            continue

                        accumulated_units += level_units
                        heightfield[level_start_row:level_end_row, level_start_col:level_end_col] = accumulated_units

                        accumulated_height += level_height
                        level_center_x = terrain_low_corner[0] + level_start_x + 0.5 * level_size_x
                        level_center_y = terrain_low_corner[1] + level_start_y + 0.5 * level_size_y
                        level_center_z = ground_top + (accumulated_height - 0.5 * level_height)

                        tile_prim_path = f"{self._prim_path}/tile_{ix}_{iy}_lvl{level}"
                        tile_prim = UsdGeom.Cube.Define(self._stage, tile_prim_path)
                        tile_prim.CreateSizeAttr(1.0)
                        tile_prim.AddTranslateOp().Set(Gf.Vec3f(level_center_x, level_center_y, level_center_z))
                        tile_prim.AddScaleOp().Set(Gf.Vec3f(level_size_x, level_size_y, level_height))
                        UsdPhysics.CollisionAPI.Apply(tile_prim.GetPrim())
                        PhysxSchema.PhysxCollisionAPI.Apply(tile_prim.GetPrim())
                        tile_mat = UsdPhysics.MaterialAPI.Apply(tile_prim.GetPrim())
                        tile_mat.CreateDynamicFrictionAttr(dynamic_friction)
                        tile_mat.CreateStaticFrictionAttr(static_friction)
                        tile_mat.CreateRestitutionAttr(restitution)
                        if use_step_width:
                            curr_start_x = level_start_x
                            curr_start_y = level_start_y
                            curr_size_x = level_size_x
                            curr_size_y = level_size_y

        # store synthetic heightfield for sensors
        # add walls to heightfield borders
        heightfield[0, :] = wall_h_units
        heightfield[-1, :] = wall_h_units
        heightfield[:, 0] = wall_h_units
        heightfield[:, -1] = wall_h_units

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=np.array([terrain_low_corner[0],
                                                   terrain_low_corner[1],
                                                   ground_top]),
                                orientation=orientation)

        # Perimeter walls (optional, thin boxes)
        wall_thickness = 0.1
        wall_z = ground_top + 0.5 * wall_height

        # +X wall
        wall_xp_path = f"{self._prim_path}/wall_xp"
        wall_xp = UsdGeom.Cube.Define(self._stage, wall_xp_path)
        wall_xp.CreateSizeAttr(1.0)
        wall_xp.AddTranslateOp().Set(Gf.Vec3f(terrain_center[0] + terrain_width * 0.5 + wall_thickness * 0.5,
                                              terrain_center[1],
                                              wall_z))
        wall_xp.AddScaleOp().Set(Gf.Vec3f(wall_thickness, terrain_length, wall_height))

        UsdPhysics.CollisionAPI.Apply(wall_xp.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(wall_xp.GetPrim())

        # -X wall
        wall_xm_path = f"{self._prim_path}/wall_xm"
        wall_xm = UsdGeom.Cube.Define(self._stage, wall_xm_path)
        wall_xm.CreateSizeAttr(1.0)
        wall_xm.AddTranslateOp().Set(Gf.Vec3f(terrain_center[0] - terrain_width * 0.5 - wall_thickness * 0.5,
                                              terrain_center[1],
                                              wall_z))
        wall_xm.AddScaleOp().Set(Gf.Vec3f(wall_thickness, terrain_length, wall_height))

        UsdPhysics.CollisionAPI.Apply(wall_xm.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(wall_xm.GetPrim())

        # +Y wall
        wall_yp_path = f"{self._prim_path}/wall_yp"
        wall_yp = UsdGeom.Cube.Define(self._stage, wall_yp_path)
        wall_yp.CreateSizeAttr(1.0)
        wall_yp.AddTranslateOp().Set(Gf.Vec3f(terrain_center[0],
                                              terrain_center[1] + terrain_length * 0.5 + wall_thickness * 0.5,
                                              wall_z))
        wall_yp.AddScaleOp().Set(Gf.Vec3f(terrain_width, wall_thickness, wall_height))
        UsdPhysics.CollisionAPI.Apply(wall_yp.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(wall_yp.GetPrim())

        # -Y wall
        wall_ym_path = f"{self._prim_path}/wall_ym"
        wall_ym = UsdGeom.Cube.Define(self._stage, wall_ym_path)
        wall_ym.CreateSizeAttr(1.0)
        wall_ym.AddTranslateOp().Set(Gf.Vec3f(terrain_center[0],
                                              terrain_center[1] - terrain_length * 0.5 - wall_thickness * 0.5,
                                              wall_z))
        wall_ym.AddScaleOp().Set(Gf.Vec3f(terrain_width, wall_thickness, wall_height))
        UsdPhysics.CollisionAPI.Apply(wall_ym.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(wall_ym.GetPrim())

        ground_wrapper = type("GroundWrapper", (), {})()
        ground_wrapper.prim = ground_prim.GetPrim()
        ground_wrapper.prim_path = str(ground_prim.GetPath())
        return ground_wrapper



    def create_stairs_terrain(self,
                terrain_size = 40,
                step_width = 0.75,
                step_height = -0.5,
                position = np.array([0.0, 0.0, 0.0]),
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1):

        # creates a terrain
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.08  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows,
                                num_cols), dtype=np.int16)

        def new_sub_terrain():

            return SubTerrain(width=num_rows,
                        length=num_cols,
                        vertical_scale=vertical_scale,
                        horizontal_scale=horizontal_scale)

        terrain=pyramid_stairs_terrain(new_sub_terrain(), step_width=step_width,
                                                    step_height=step_height,
                                                    platform_size=1.0)
        heightfield[0:num_rows, :] = terrain.height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                    horizontal_scale=horizontal_scale,
                                    vertical_scale=vertical_scale,
                                    slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, -terrain_length/2.0, 0]) + position

        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        terrain_prim = add_terrain_to_stage(stage=self._stage,
                    vertices=vertices,
                    triangles=triangles,
                    position=position,
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def create_stepup_terrain(self,
                    terrain_size=30,
                    stairs_ratio: float = 0.2,
                    min_steps: int = 1,
                    max_steps: int = 8,
                    pyramid_platform_size: float = 10.0,
                    step_height: float = 0.5,
                    patch_size: float = 5.0,
                    res_low: float = 0.1,
                    res_high: float = 0.03,
                    position = np.array([0.0, 0.0, 0.0]),
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    restitution=0.1,
                    perimeter_wall_height: float = 2.0):
        """
        Create a terrain composed of tiled patches. Each patch is either a straight stairs
        (created with `stairs_terrain`) or a pyramid stairs (created with `pyramid_stairs_terrain`).

        Parameters:
            terrain_size: total terrain size in meters (square)
            stairs_ratio: fraction [0..1] of patches that are straight stairs (rest are pyramid stairs)
            min_steps, max_steps: min/max number of steps to try to create per patch (integers, preserved)
            pyramid_platform_size: platform size (meters) at the center of pyramid stairs patches
            patch_platform_size: platform size parameter for individual stair/pyramid functions when needed
            step_width, step_height: nominal step dimensions (meters)
            patch_size: size in meters for each tiled patch
            perimeter_wall_height: height of the perimeter wall in meters (default 1.0)
        """

        # Terrain dimensions and scales
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        # choose horizontal resolution: use high resolution when stairs are present
        use_high_res = (stairs_ratio > 0.0) and (abs(int(min_steps)) > 0 or abs(int(max_steps)) > 0)
        horizontal_scale = res_high if use_high_res else res_low
        vertical_scale = 0.005  # [m]  -- discrete vertical unit

        # grid size in cells
        num_rows = max(1, int(round(terrain_width / horizontal_scale)))
        num_cols = max(1, int(round(terrain_length / horizontal_scale)))

        # use integer heightfield (units of vertical_scale) as before, but use int32 to avoid overflow
        heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int32)

        # patch sizes in cells
        patch_rows = max(1, int(round(patch_size / horizontal_scale)))
        patch_cols = patch_rows

        # how many patches along each axis
        n_patches_x = max(1, int(np.ceil(num_rows / patch_rows)))
        n_patches_y = max(1, int(np.ceil(num_cols / patch_cols)))

        for px in range(n_patches_x):
            for py in range(n_patches_y):
                # compute cell indices for this patch
                ix = px * patch_rows
                iy = py * patch_cols
                # limit sizes to remaining cells
                cur_rows = min(patch_rows, num_rows - ix)
                cur_cols = min(patch_cols, num_cols - iy)

                # create a subterrain for the patch
                sub = SubTerrain(width=cur_rows, length=cur_cols,
                                vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

                # initialize sub heightfield as integer units (same dtype)
                sub.height_field_raw = np.zeros((sub.width, sub.length), dtype=np.int32)

                # decide patch type
                # determine absolute number of steps to create (count must be positive and integer)
                abs_min = max(1, abs(int(min_steps)))
                abs_max = max(abs_min, abs(int(max_steps)))
                steps_count = int(np.random.randint(abs_min, abs_max + 1))

                if np.random.rand() < stairs_ratio:
                    # straight stairs patch
                    # compute a reasonable step width in cells and clamp steps_count so it fits
                    step_width_cells = max(1, cur_rows // max(1, steps_count))
                    max_fit_steps = max(1, cur_rows // step_width_cells)
                    steps_count = max(1, min(steps_count, max_fit_steps))

                    step_width_m = step_width_cells * horizontal_scale

                    # convert step_height to discrete vertical units robustly
                    step_h_units = max(1, int(round(abs(step_height) / vertical_scale)))

                    # If min_steps is negative, build stairs that start at negative heights and ascend to zero
                    if min_steps < 0:
                        for i in range(steps_count):
                            start = i * step_width_cells
                            stop = min(sub.width, (i + 1) * step_width_cells)
                            # heights go from more negative toward zero (units)
                            h_units = - (steps_count - i) * step_h_units
                            sub.height_field_raw[start:stop, :] += h_units
                    else:
                        # use existing helper for positive-step stairs (expects meters for step_width/step_height)
                        # If your stairs_terrain expects integer units, adapt accordingly.
                        sub = stairs_terrain(sub, step_width=step_width_m, step_height=step_height)
                else:
                    # pyramid stairs patch
                    platform_cells = max(1, int(round(pyramid_platform_size / horizontal_scale)))
                    denom = max(1, 2 * steps_count)
                    step_width_cells = max(1, (min(cur_rows, cur_cols) - platform_cells) // denom)
                    step_width_m = step_width_cells * horizontal_scale

                    # compute how many rings fit and clamp steps_count
                    min_dim = min(cur_rows, cur_cols)
                    max_possible_steps = (min_dim - platform_cells) // (2 * max(1, step_width_cells))
                    if max_possible_steps <= 0:
                        # nothing fits - leave patch flat
                        pass
                    else:
                        steps_count = min(steps_count, max_possible_steps)
                        # convert step_height to integer units
                        step_h_units = max(1, int(round(abs(step_height) / vertical_scale)))

                        if min_steps < 0:
                            # negative-to-zero pyramid manually using integer units
                            start_x = 0
                            stop_x = sub.width
                            start_y = 0
                            stop_y = sub.length
                            for k in range(steps_count):
                                start_x += step_width_cells
                                stop_x = max(start_x, stop_x - step_width_cells)
                                start_y += step_width_cells
                                stop_y = max(start_y, stop_y - step_width_cells)
                                if start_x >= stop_x or start_y >= stop_y:
                                    break
                                h_units = - (steps_count - k) * step_h_units
                                sub.height_field_raw[start_x:stop_x, start_y:stop_y] = h_units
                        else:
                            # call pyramid helper that uses meters for step_width/height
                            sub = pyramid_stairs_terrain(sub, step_width=step_width_m, step_height=step_height, platform_size=pyramid_platform_size)

                # write patch into global heightfield (clip to shapes)
                xr0 = ix
                xr1 = ix + sub.height_field_raw.shape[0]
                yr0 = iy
                yr1 = iy + sub.height_field_raw.shape[1]

                # directly assign - if you want seams smoothed, you can blend shared edges here
                heightfield[xr0:xr1, yr0:yr1] = sub.height_field_raw

        # === Add perimeter wall by setting border cells to the desired height ===
        wall_h_m = float(perimeter_wall_height)
        wall_units = max(1, int(round(wall_h_m / vertical_scale)))  # integer units for heightfield

        # set border rows/columns to at least wall_units (preserves any higher existing heights)
        heightfield[0, :] = np.maximum(heightfield[0, :], wall_units)
        heightfield[-1, :] = np.maximum(heightfield[-1, :], wall_units)
        heightfield[:, 0] = np.maximum(heightfield[:, 0], wall_units)
        heightfield[:, -1] = np.maximum(heightfield[:, -1], wall_units)

        # convert to mesh and add to stage
        vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                                            horizontal_scale=horizontal_scale,
                                                            vertical_scale=vertical_scale,
                                                            slope_threshold=1.5)

        position = np.array([-terrain_width / 2.0, -terrain_length / 2.0, 0]) + position
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        terrain_prim = add_terrain_to_stage(stage=self._stage,
                                    vertices=vertices,
                                    triangles=triangles,
                                    position=position,
                                    orientation=orientation,
                                    prim_path=self._prim_path,
                                    static_friction=static_friction,
                                    dynamic_friction=dynamic_friction,
                                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def create_random_patched_terrain(self,
                    terrain_size=40,
                    min_height=-0.2,
                    max_height=0.2,
                    step=0.2,
                    downsampled_scale=0.5,
                    position=np.array([0.0, 0.0, 0.0]),
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    restitution=0.1,
                    patch_ratio=0.3,
                    patch_size=10,
                    with_walls: bool =True,
                    wall_height: float = 2.0):

        # Terrain dimensions
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.08  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain():
            return SubTerrain(width=num_rows,
                            length=num_cols,
                            vertical_scale=vertical_scale,
                            horizontal_scale=horizontal_scale)

        # Generate base terrain
        terrain = random_uniform_terrain(new_sub_terrain(),
                                        min_height=min_height, max_height=max_height,
                                        step=step,
                                        downsampled_scale=downsampled_scale)
        heightfield[0:num_rows, :] = terrain.height_field_raw

        # Apply flat patches
        num_patches_x = terrain_width // patch_size
        num_patches_y = terrain_length // patch_size

        for i in range(int(num_patches_x * num_patches_y * patch_ratio)):
            patch_x = np.random.randint(0, num_patches_x) * (patch_size / horizontal_scale)
            patch_y = np.random.randint(0, num_patches_y) * (patch_size / horizontal_scale)
            patch_x = int(patch_x)
            patch_y = int(patch_y)

            patch_height = np.random.uniform(min_height, max_height)
            heightfield[patch_x:patch_x + int(patch_size / horizontal_scale),
                        patch_y:patch_y + int(patch_size / horizontal_scale)] = patch_height / vertical_scale

        if with_walls:
            # Add vertical walls at the borders (2 meters height)
            wall_height = wall_height / vertical_scale  # Convert meters to heightmap scale
            heightfield[0, :] = wall_height  # Top border
            heightfield[-1, :] = wall_height  # Bottom border
            heightfield[:, 0] = wall_height  # Left border
            heightfield[:, -1] = wall_height  # Right border

        # Convert to mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                                            horizontal_scale=horizontal_scale,
                                                            vertical_scale=vertical_scale,
                                                            slope_threshold=1.5)

        position = np.array([-terrain_width / 2.0, -terrain_length / 2.0, 0]) + position
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        terrain_prim = add_terrain_to_stage(stage=self._stage,
                                    vertices=vertices,
                                    triangles=triangles,
                                    position=position,
                                    orientation=orientation,
                                    prim_path=self._prim_path,
                                    static_friction=static_friction,
                                    dynamic_friction=dynamic_friction,
                                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def create_random_uniform_terrain(self,
                    terrain_size=40,
                    min_height=-0.2,
                    max_height=0.2,
                    step=0.2,
                    downsampled_scale=0.5,
                    position=np.array([0.0, 0.0, 0.0]),
                    dynamic_friction=0.5,
                    static_friction=0.5,
                    restitution=0.1,
                    with_walls: bool =True,
                    wall_height: float = 2.0):

        # Terrain dimensions
        num_terrains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.2  # [m]
        vertical_scale = 0.005  # [m]

        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)

        heightfield = np.zeros((num_terrains * num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain():
            return SubTerrain(width=num_rows,
                            length=num_cols,
                            vertical_scale=vertical_scale,
                            horizontal_scale=horizontal_scale)

        # Generate base terrain
        terrain = random_uniform_terrain(new_sub_terrain(),
                                        min_height=min_height, max_height=max_height,
                                        step=step,
                                        downsampled_scale=downsampled_scale)
        heightfield[0:num_rows, :] = terrain.height_field_raw

        if with_walls:
            # Add vertical walls at the borders (2 meters height)
            wall_height = wall_height / vertical_scale  # Convert meters to heightmap scale
            heightfield[0, :] = wall_height  # Top border
            heightfield[-1, :] = wall_height  # Bottom border
            heightfield[:, 0] = wall_height  # Left border
            heightfield[:, -1] = wall_height  # Right border

        # Convert to mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                                            horizontal_scale=horizontal_scale,
                                                            vertical_scale=vertical_scale,
                                                            slope_threshold=1.5)

        position = np.array([-terrain_width / 2.0, -terrain_length / 2.0, 0]) + position
        orientation = np.array([1.0, 0.0, 0.0, 0.0])

        terrain_prim = add_terrain_to_stage(stage=self._stage,
                                    vertices=vertices,
                                    triangles=triangles,
                                    position=position,
                                    orientation=orientation,
                                    prim_path=self._prim_path,
                                    static_friction=static_friction,
                                    dynamic_friction=dynamic_friction,
                                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def get_obstacles_terrain(self,
                    terrain_size = 40.0,
                    num_obs = 50,
                    max_height = 0.5,
                    min_size = 0.5,
                    max_size = 5.0,
                    position = np.array([0.0, 0.0, 0.0]),
                dynamic_friction=0.5,
                static_friction=0.5,
                restitution=0.1):

        # create all available terrain types
        num_terains = 1
        terrain_width = terrain_size
        terrain_length = terrain_size
        horizontal_scale = 0.08  # [m]
        vertical_scale = 0.005  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        heightfield = np.zeros((num_terains*num_rows, num_cols), dtype=np.int16)

        def new_sub_terrain():
            return SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        heightfield[0:num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(),
                                            max_height=max_height,
                                            min_size=min_size,
                                            max_size=max_size,
                                            num_rects=num_obs).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)

        position = np.array([-terrain_width/2.0, -terrain_length/2.0, 0]) + position

        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        terrain_prim = add_terrain_to_stage(stage=self._stage,
                    vertices=vertices,
                    triangles=triangles,
                    position=position,
                    orientation=orientation,
                    prim_path=self._prim_path,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=restitution)

        self._store_heightfield(heightfield=heightfield,
                                horizontal_scale=horizontal_scale,
                                vertical_scale=vertical_scale,
                                position=position,
                                orientation=orientation)

        return terrain_prim

    def post_reset(self):

        a = 1

    def get_observations(self):

        pass

    def calculate_metrics(self) -> None:

        pass

    def _store_heightfield(self, heightfield, horizontal_scale, vertical_scale, position, orientation):
        """Cache terrain heightfield (in meters) and transform for later queries."""
        self._heightfield_raw = heightfield
        base_z = float(position[2]) if position is not None else 0.0
        self.heightfield_world = heightfield.astype(np.float32) * float(vertical_scale) + base_z
        self._horizontal_scale = float(horizontal_scale)
        self._vertical_scale = float(vertical_scale)
        self.position = np.array(position, dtype=np.float64)
        self.orientation = np.array(orientation, dtype=np.float64) if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])

    def get_height_at(self, x_world: float, y_world: float) -> float:
        """Return terrain height (meters) at world coordinates (x_world, y_world)."""
        if self.heightfield_world is None:
            return 0.0

        # inverse transform from world to terrain local frame
        p = np.array([x_world, y_world, 0.0], dtype=np.float64) - self.position
        rot = self._quat_to_rot(self.orientation)
        p_local = rot.T @ p

        # convert to grid indices
        gx = p_local[0] / self._horizontal_scale
        gy = p_local[1] / self._horizontal_scale

        h, w = self.heightfield_world.shape
        if gx < 0 or gy < 0 or gx > h - 1 or gy > w - 1:
            # outside cached terrain
            return 0.0

        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, h - 1)
        y1 = min(y0 + 1, w - 1)

        fx = gx - x0
        fy = gy - y0

        h00 = self.heightfield_world[x0, y0]
        h10 = self.heightfield_world[x1, y0]
        h01 = self.heightfield_world[x0, y1]
        h11 = self.heightfield_world[x1, y1]

        hx0 = h00 * (1 - fx) + h10 * fx
        hx1 = h01 * (1 - fx) + h11 * fx
        return float(hx0 * (1 - fy) + hx1 * fy)

    def get_heights_at(self, x_world, y_world):
        """Vectorized height query. x_world and y_world must be same shape arrays; returns same shape."""
        if self.heightfield_world is None:
            return np.zeros_like(x_world, dtype=np.float32)

        x_arr = np.asarray(x_world, dtype=np.float64)
        y_arr = np.asarray(y_world, dtype=np.float64)
        orig_shape = x_arr.shape
        flat_x = x_arr.reshape(-1)
        flat_y = y_arr.reshape(-1)

        pts = np.stack([flat_x - self.position[0], flat_y - self.position[1], np.zeros_like(flat_x)], axis=0)
        rot = self._quat_to_rot(self.orientation)
        local = rot.T @ pts

        gx = local[0] / self._horizontal_scale
        gy = local[1] / self._horizontal_scale

        h, w = self.heightfield_world.shape
        mask = (gx >= 0) & (gy >= 0) & (gx <= (h - 1)) & (gy <= (w - 1))

        heights = np.zeros_like(gx, dtype=np.float32)
        if mask.any():
            gx_m = gx[mask]
            gy_m = gy[mask]

            x0 = np.floor(gx_m).astype(np.int64)
            y0 = np.floor(gy_m).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, h - 1)
            y1 = np.clip(y0 + 1, 0, w - 1)

            fx = gx_m - x0
            fy = gy_m - y0

            h00 = self.heightfield_world[x0, y0]
            h10 = self.heightfield_world[x1, y0]
            h01 = self.heightfield_world[x0, y1]
            h11 = self.heightfield_world[x1, y1]

            hx0 = h00 * (1 - fx) + h10 * fx
            hx1 = h01 * (1 - fx) + h11 * fx
            heights[mask] = hx0 * (1 - fy) + hx1 * fy

        return heights.reshape(orig_shape).astype(np.float32)

    def get_max_height_in_rect(self, x_world: float, y_world: float, half_extent: float = 0.3) -> float:
        """Return max height within a square centered at (x_world, y_world) with given half-extent (meters)."""
        if self.heightfield_world is None:
            return 0.0

        # build a small grid around the center and reuse vectorized sampling
        # choose a coarse 3x3 sample to reduce cost
        coords = np.linspace(-half_extent, half_extent, 3)
        dx, dy = np.meshgrid(coords, coords)
        xs = x_world + dx
        ys = y_world + dy
        heights = self.get_heights_at(xs, ys)
        return float(np.max(heights))


    def _quat_to_rot(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
        w, x, y, z = quat
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        return np.array(
            [[ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)],
             [2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)],
             [2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz]],
            dtype=np.float64
        )

    def is_done(self) -> None:

        pass
