# Copyright (C) 2023  Andrea Patrizi
#
# Simple height grid sensor built on top of cached terrain heightfields.

import torch


class HeightGridSensor:
    def __init__(self,
                 terrain_utils,
                 grid_size: int,
                 resolution: float,
                 n_envs: int,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            terrain_utils: instance of RlTerrains. Uses its cached heightfield if available.
            grid_size: number of pixels per side (square grid).
            resolution: meters per pixel.
            n_envs: total number of environments (preallocates output buffer).
            device, dtype: output tensor placement/type.
        """
        if n_envs is None:
            raise ValueError("n_envs must be provided")

        self._grid_size = int(grid_size)
        self._resolution = float(resolution)
        self._device = device
        self._dtype = dtype
        self._n_envs = int(n_envs)

        idx = torch.arange(self._grid_size, device=self._device, dtype=self._dtype)
        center = (self._grid_size - 1) / 2.0
        coords = (idx - center) * self._resolution
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        self._grid_offsets = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        # terrain data cached on device to avoid cpu<->gpu copies during read
        self._heightfield = None
        self._horizontal_scale = None
        self._rot_w2t_2x2 = None
        self._terrain_xy = None

        if terrain_utils is not None and getattr(terrain_utils, "_heightfield_world", None) is not None:
            hf_np = terrain_utils._heightfield_world  # stored in meters
            self._heightfield = torch.as_tensor(hf_np, device=self._device, dtype=self._dtype)
            self._horizontal_scale = float(terrain_utils._horizontal_scale)

            # world to terrain rotation (2x2) and translation
            quat = torch.as_tensor(terrain_utils._orientation, device=self._device, dtype=self._dtype)
            rot = self._quat_to_rotmat(quat.unsqueeze(0))[0]  # (3,3)
            self._rot_w2t_2x2 = rot[:2, :2].t()  # transpose for world->terrain
            self._terrain_xy = torch.as_tensor(terrain_utils._position[:2], device=self._device, dtype=self._dtype)

            self._h, self._w = self._heightfield.shape
        else:
            self._heightfield = None
            self._horizontal_scale = 1.0
            self._rot_w2t_2x2 = torch.eye(2, device=self._device, dtype=self._dtype)
            self._terrain_xy = torch.zeros(2, device=self._device, dtype=self._dtype)
            self._h, self._w = 1, 1

        self._buffer = torch.zeros((self._n_envs, self._grid_size, self._grid_size),
                                   device=self._device, dtype=self._dtype)

    def read(self, base_positions: torch.Tensor, base_quats: torch.Tensor) -> torch.Tensor:
        """Return height images of shape (N, grid, grid) for provided bases.

        base_quats are expected in [w, x, y, z] order (consistent with the env).
        """
        num_envs = base_positions.shape[0]

        rot_mats = self._quat_to_rotmat(base_quats)  # (N,3,3)
        rot_xy = rot_mats[:, :2, :2]  # planar rotation

        offsets = self._grid_offsets.unsqueeze(0).expand(num_envs, -1, -1)
        world_xy = torch.bmm(offsets, rot_xy.transpose(1, 2))
        world_xy += base_positions[:, :2].unsqueeze(1)

        if self._heightfield is None:
            heights = self._buffer[:num_envs]
            heights.zero_()
            return heights

        # world -> terrain local
        rel_xy = world_xy - self._terrain_xy
        local_xy = torch.matmul(rel_xy, self._rot_w2t_2x2)

        gx = local_xy[..., 0] / self._horizontal_scale
        gy = local_xy[..., 1] / self._horizontal_scale

        # mask inside
        mask = (gx >= 0) & (gy >= 0) & (gx <= (self._h - 1)) & (gy <= (self._w - 1))

        # clamp indices for sampling
        x0 = torch.floor(gx).long().clamp(0, self._h - 1)
        y0 = torch.floor(gy).long().clamp(0, self._w - 1)
        x1 = (x0 + 1).clamp(0, self._h - 1)
        y1 = (y0 + 1).clamp(0, self._w - 1)

        fx = (gx - x0).to(self._dtype)
        fy = (gy - y0).to(self._dtype)

        flat = self._heightfield.view(-1)
        idx00 = x0 * self._w + y0
        idx10 = x1 * self._w + y0
        idx01 = x0 * self._w + y1
        idx11 = x1 * self._w + y1

        h00 = flat.gather(0, idx00.view(-1)).view_as(idx00)
        h10 = flat.gather(0, idx10.view(-1)).view_as(idx10)
        h01 = flat.gather(0, idx01.view(-1)).view_as(idx01)
        h11 = flat.gather(0, idx11.view(-1)).view_as(idx11)

        hx0 = h00 * (1 - fx) + h10 * fx
        hx1 = h01 * (1 - fx) + h11 * fx
        heights = hx0 * (1 - fy) + hx1 * fy

        heights = heights.masked_fill(~mask, 0.0)
        heights = heights.view(num_envs, self._grid_size, self._grid_size)

        self._buffer[:num_envs].copy_(heights)
        return self._buffer[:num_envs]

    def _quat_to_rotmat(self, quat: torch.Tensor) -> torch.Tensor:
        # quat: (N,4) -> rot: (N,3,3); assumes [w,x,y,z]
        w, x, y, z = quat.unbind(-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rot = torch.stack([
            torch.stack([ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)], dim=-1),
            torch.stack([2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
            torch.stack([2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz], dim=-1)
        ], dim=-2)
        return rot
