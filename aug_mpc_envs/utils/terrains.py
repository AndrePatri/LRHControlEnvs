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

from aug_mpc_envs.utils.terrain_utils import *

from pxr import Usd

class RlTerrains():

    def __init__(self, 
                stage: Usd.Stage,
                prim_path: str = "/World/terrain"):
        
        self._stage = stage

        self._prim_path=prim_path

        # cache heightfield info to allow querying terrain height at world positions
        self._heightfield_world = None  # meters
        self._heightfield_raw = None    # original integer grid
        self._horizontal_scale = None
        self._vertical_scale = None
        self._position = None
        self._orientation = None

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
        horizontal_scale = 0.25  # [m]
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

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

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
        horizontal_scale = 0.25  # [m]
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

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

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
                    pyramid_platform_size: float = 5.0,
                    step_height: float = 0.5,
                    patch_size: float = 5.0,
                    res_low: float = 0.25,
                    res_high: float = 0.05,
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

        position = np.array([-terrain_width / 2.0, terrain_length / 2.0, 0]) + position
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

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
        horizontal_scale = 0.25  # [m]
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

        position = np.array([-terrain_width / 2.0, terrain_length / 2.0, 0]) + position
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

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

        position = np.array([-terrain_width / 2.0, terrain_length / 2.0, 0]) + position
        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])

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
        horizontal_scale = 0.25  # [m]
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

        position = np.array([-terrain_width/2.0, terrain_length/2.0, 0]) + position

        orientation = np.array([0.70711, 0.0, 0.0, -0.70711])
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
        self._heightfield_world = heightfield.astype(np.float32) * float(vertical_scale)
        self._horizontal_scale = float(horizontal_scale)
        self._vertical_scale = float(vertical_scale)
        self._position = np.array(position, dtype=np.float64)
        self._orientation = np.array(orientation, dtype=np.float64) if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])

    def get_height_at(self, x_world: float, y_world: float) -> float:
        """Return terrain height (meters) at world coordinates (x_world, y_world)."""
        if self._heightfield_world is None:
            return 0.0

        # inverse transform from world to terrain local frame
        p = np.array([x_world, y_world, 0.0], dtype=np.float64) - self._position
        rot = self._quat_to_rot(self._orientation)
        p_local = rot.T @ p

        # convert to grid indices
        gx = p_local[0] / self._horizontal_scale
        gy = p_local[1] / self._horizontal_scale

        h, w = self._heightfield_world.shape
        if gx < 0 or gy < 0 or gx > h - 1 or gy > w - 1:
            # outside cached terrain
            return 0.0

        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(x0 + 1, h - 1)
        y1 = min(y0 + 1, w - 1)

        fx = gx - x0
        fy = gy - y0

        h00 = self._heightfield_world[x0, y0]
        h10 = self._heightfield_world[x1, y0]
        h01 = self._heightfield_world[x0, y1]
        h11 = self._heightfield_world[x1, y1]

        hx0 = h00 * (1 - fx) + h10 * fx
        hx1 = h01 * (1 - fx) + h11 * fx
        return float(hx0 * (1 - fy) + hx1 * fy)


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
