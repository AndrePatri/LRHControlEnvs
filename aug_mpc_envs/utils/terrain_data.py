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
# Backend-neutral terrain payload + height queries. A ``TerrainData`` carries the heightfield and its
# transform in a simulator-independent way, and exposes exactly the attributes the height-grid sensor
# reads (``heightfield_world``, ``_horizontal_scale``, ``position``, ``orientation``) plus the
# height-query helpers used for spawn placement. It is produced by ``terrain_generation`` and consumed
# by any backend (Genesis feeds ``heightfield_raw`` to ``gs.morphs.Terrain``; Isaac authors USD and
# already caches the same fields on ``RlTerrains``).

import numpy as np


class TerrainData:
    """Heightfield + transform, with backend-neutral height queries.

    Convention (shared with Genesis ``gs.morphs.Terrain`` and the Isaac heightfields): grid cell
    ``(i, j)`` sits at world position ``position + (i * horizontal_scale, j * horizontal_scale,
    heightfield_raw[i, j] * vertical_scale)``, i.e. ``position`` is the world location of cell
    ``(0, 0)`` (the terrain corner) and the grid grows along +x (rows) and +y (cols).

    Args:
        heightfield_raw: integer (or float) 2D grid of heights in units of ``vertical_scale``.
        horizontal_scale: cell size [m].
        vertical_scale: height per unit [m].
        position: world position of cell (0,0) [m], shape (3,). ``z`` is the terrain base height.
        orientation: terrain quaternion [w, x, y, z] (defaults to identity).
        vertices, faces: optional trimesh (meters) for backends that import meshes instead of a
            native heightfield. Not required by Genesis (which builds the mesh from the heightfield).
        boxes: optional (N, 6) array of axis-aligned boxes in WORLD coords [cx, cy, cz, sx, sy, sz]
            (center + full size, meters). Provided by the "prim" generators (random_tiles, stepup_prim)
            so a backend can build the terrain from cheap primitive box colliders (crisp, exact contact,
            no SDF) instead of the heightfield mesh. The heightfield is still kept for height sensing.
    """

    def __init__(self,
                 heightfield_raw: np.ndarray,
                 horizontal_scale: float,
                 vertical_scale: float,
                 position=None,
                 orientation=None,
                 vertices: np.ndarray = None,
                 faces: np.ndarray = None,
                 boxes: np.ndarray = None):
        if heightfield_raw is None:
            raise ValueError("heightfield_raw must be provided")
        self._heightfield_raw = np.asarray(heightfield_raw)
        self._horizontal_scale = float(horizontal_scale)
        self._vertical_scale = float(vertical_scale)
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.position = np.array(position, dtype=np.float64)
        self.orientation = (np.array(orientation, dtype=np.float64)
                            if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0]))
        base_z = float(self.position[2])
        # heightfield in meters (world z), what the sensor and queries use.
        self.heightfield_world = self._heightfield_raw.astype(np.float32) * self._vertical_scale + base_z
        self.vertices = vertices
        self.faces = faces
        self.boxes = (np.asarray(boxes, dtype=np.float64).reshape(-1, 6)
                      if boxes is not None and len(boxes) > 0 else None)

    def num_boxes(self) -> int:
        return 0 if self.boxes is None else int(self.boxes.shape[0])

    # --- compatibility aliases (RlTerrains/HeightGridSensor field names) -------------------------
    @property
    def heightfield_raw(self):
        return self._heightfield_raw

    @property
    def horizontal_scale(self):
        return self._horizontal_scale

    @property
    def vertical_scale(self):
        return self._vertical_scale

    def num_faces(self) -> int:
        """Triangle-mesh face count a heightfield terrain expands to (2 per grid cell)."""
        h, w = self._heightfield_raw.shape
        return 2 * max(0, h - 1) * max(0, w - 1)

    def coarsened(self, max_grid_side: int) -> "TerrainData":
        """Return a coarser copy whose grid side <= ``max_grid_side`` (or self if already coarse
        enough). Used for backends that build an SDF from the terrain mesh (genesis): a fine
        heightfield explodes the SDF pre-processing (>50k faces -> minutes + GBs of RAM). Downsampling
        is a MAX-pool over stride x stride blocks so raised steps/obstacles are preserved (not averaged
        away). The transform is kept consistent (horizontal_scale *= stride, same corner position), so
        the height sensor and the spawned terrain stay aligned."""
        h, w = self._heightfield_raw.shape
        side = max(h, w)
        if max_grid_side is None or side <= int(max_grid_side):
            return self
        stride = int(np.ceil(side / float(max_grid_side)))
        if stride <= 1:
            return self
        hf = self._heightfield_raw
        ph = (-h) % stride
        pw = (-w) % stride
        if ph or pw:
            hf = np.pad(hf, ((0, ph), (0, pw)), mode="edge")
        H, W = hf.shape
        hf = hf.reshape(H // stride, stride, W // stride, stride).max(axis=(1, 3))
        return TerrainData(hf, self._horizontal_scale * stride, self._vertical_scale,
                           position=self.position, orientation=self.orientation)

    # --- height queries (identical semantics to RlTerrains) -------------------------------------
    def get_height_at(self, x_world: float, y_world: float) -> float:
        """Return terrain height (meters) at world coordinates (x_world, y_world)."""
        if self.heightfield_world is None:
            return 0.0
        p = np.array([x_world, y_world, 0.0], dtype=np.float64) - self.position
        rot = self._quat_to_rot(self.orientation)
        p_local = rot.T @ p
        gx = p_local[0] / self._horizontal_scale
        gy = p_local[1] / self._horizontal_scale
        h, w = self.heightfield_world.shape
        if gx < 0 or gy < 0 or gx > h - 1 or gy > w - 1:
            return 0.0
        x0 = int(np.floor(gx)); y0 = int(np.floor(gy))
        x1 = min(x0 + 1, h - 1); y1 = min(y0 + 1, w - 1)
        fx = gx - x0; fy = gy - y0
        h00 = self.heightfield_world[x0, y0]; h10 = self.heightfield_world[x1, y0]
        h01 = self.heightfield_world[x0, y1]; h11 = self.heightfield_world[x1, y1]
        hx0 = h00 * (1 - fx) + h10 * fx
        hx1 = h01 * (1 - fx) + h11 * fx
        return float(hx0 * (1 - fy) + hx1 * fy)

    def get_heights_at(self, x_world, y_world):
        """Vectorized height query. x_world and y_world must be same-shape arrays; returns same shape."""
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
            gx_m = gx[mask]; gy_m = gy[mask]
            x0 = np.floor(gx_m).astype(np.int64); y0 = np.floor(gy_m).astype(np.int64)
            x1 = np.clip(x0 + 1, 0, h - 1); y1 = np.clip(y0 + 1, 0, w - 1)
            fx = gx_m - x0; fy = gy_m - y0
            h00 = self.heightfield_world[x0, y0]; h10 = self.heightfield_world[x1, y0]
            h01 = self.heightfield_world[x0, y1]; h11 = self.heightfield_world[x1, y1]
            hx0 = h00 * (1 - fx) + h10 * fx
            hx1 = h01 * (1 - fx) + h11 * fx
            heights[mask] = hx0 * (1 - fy) + hx1 * fy
        return heights.reshape(orig_shape).astype(np.float32)

    def get_max_height_in_rect(self, x_world: float, y_world: float, half_extent: float = 0.3) -> float:
        """Return max height within a square centered at (x_world, y_world) with given half-extent (meters)."""
        if self.heightfield_world is None:
            return 0.0
        coords = np.linspace(-half_extent, half_extent, 3)
        dx, dy = np.meshgrid(coords, coords)
        xs = x_world + dx
        ys = y_world + dy
        heights = self.get_heights_at(xs, ys)
        return float(np.max(heights))

    @staticmethod
    def _quat_to_rot(quat: np.ndarray) -> np.ndarray:
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
