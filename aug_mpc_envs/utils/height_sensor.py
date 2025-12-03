# Copyright (C) 2023  Andrea Patrizi
#
# Simple height grid sensor built on top of cached terrain heightfields.

import torch
import torch.nn.functional as F

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
        self._heightfield_4d = None
        self._horizontal_scale = None
        self._rot_w2t_3x3 = None
        self._terrain_pos = None

        hf_np = terrain_utils.heightfield_world  # stored in meters

        self._heightfield = torch.as_tensor(hf_np, device=self._device, dtype=self._dtype).contiguous()
        self._horizontal_scale = float(terrain_utils._horizontal_scale)

        # world to terrain rotation and translation
        quat = torch.as_tensor(terrain_utils.orientation, device=self._device, dtype=self._dtype)
        rot = self._quat_to_rotmat(quat.unsqueeze(0))[0]  # (3,3) expects (w,x,y,z)
        self._rot_w2t_3x3 = rot.t()  # world -> terrain
        self._terrain_pos = torch.as_tensor(terrain_utils.position, device=self._device, dtype=self._dtype)

        self._h, self._w = self._heightfield.shape
        self._heightfield_4d = self._heightfield.view(1, 1, self._h, self._w)

        self._buffer = torch.zeros((self._n_envs, self._grid_size, self._grid_size),
                                   device=self._device, dtype=self._dtype)

    def read(self, basepositions: torch.Tensor, base_quats: torch.Tensor) -> torch.Tensor:
        """Return height images of shape (N, grid, grid) for provided bases.

        base_quats are expected in [w, x, y, z] order (consistent with the env).
        """
        num_envs = basepositions.shape[0]

        rot_mats = self._quat_to_rotmat(base_quats)  # (N,3,3)

        rot_xy = rot_mats[:, :2, :2]  # planar rotation

        offsets = self._grid_offsets.unsqueeze(0).expand(num_envs, -1, -1)

        world_xy = torch.bmm(offsets, rot_xy.transpose(1, 2))
        world_xy += basepositions[:, :2].unsqueeze(1)

        if self._heightfield is None:
            heights = self._buffer[:num_envs]
            heights.zero_()
            return heights

        # world -> terrain local
        rel = torch.zeros((num_envs, world_xy.shape[1], 3), device=self._device, dtype=self._dtype)
        rel[..., :2] = world_xy - self._terrain_pos[:2]
        local = torch.matmul(rel, self._rot_w2t_3x3)

        gx = local[..., 0] / self._horizontal_scale
        gy = local[..., 1] / self._horizontal_scale

        # grid_sample expects normalized coordinates [-1,1], order (x,y) = (col,row)
        u = (gy / (self._w - 1) * 2.0 - 1.0).clamp(-2.0, 2.0)  # width
        v = (gx / (self._h - 1) * 2.0 - 1.0).clamp(-2.0, 2.0)  # height

        grid = torch.stack([u, v], dim=-1)
        grid = grid.view(num_envs, self._grid_size, self._grid_size, 2)

        hf = self._heightfield_4d.expand(num_envs, 1, self._h, self._w)
        sampled = F.grid_sample(hf, grid, align_corners=True, padding_mode="zeros")
        heights = sampled.view(num_envs, self._grid_size, self._grid_size)

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
