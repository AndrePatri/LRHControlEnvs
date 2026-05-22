# Debug visualizer for height grid samples in Isaac Sim.
# Creates individual non-physical spheres at sampled world positions.

import torch


class HeightGridVisualizer:
    def __init__(self,
                 robot_name: str,
                 num_envs: int,
                 grid_size: int,
                 resolution: float,
                 base_prim_path: str = "/World/debug/height_grid",
                 marker_radius: float = 0.03,
                 forward_offset: float = 0.0,
                 lateral_offset: float = 0.0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.robot_name = robot_name
        self.num_envs = num_envs
        self.grid_size = grid_size
        self.resolution = resolution
        self.marker_radius = marker_radius
        self._base_offset = torch.tensor([forward_offset, lateral_offset],
                                         device=device,
                                         dtype=dtype)
        self.device = device
        self.dtype = dtype

        # lazy import Isaac APIs (requires SimulationApp already started)
        try:
            from isaacsim.core.utils import prims as prim_utils
            from isaacsim.core.utils.stage import get_current_stage
        except ImportError:
            from omni.isaac.core.utils import prims as prim_utils
            from omni.isaac.core.utils.stage import get_current_stage
        from pxr import UsdGeom, Gf

        self._prim_utils = prim_utils
        self._get_stage = get_current_stage
        self._UsdGeom = UsdGeom
        self._Gf = Gf

        # precompute grid offsets in robot frame
        idx = torch.arange(self.grid_size, device=self.device, dtype=self.dtype)
        center = (self.grid_size - 1) / 2.0
        coords = (idx - center) * self.resolution
        gy, gx = torch.meshgrid(coords, coords, indexing="ij")
        self._grid_offsets = torch.stack([gx, gy], dim=-1).reshape(-1, 2)  # (P,2)

        self._prim_paths = []
        self._define_markers(base_prim_path)

    def _define_markers(self, base_path: str):
        stage = self._get_stage()
        # create sphere prims for each env and grid cell
        for env_idx in range(self.num_envs):
            for cell_idx in range(self.grid_size * self.grid_size):
                prim_path = f"{base_path}/{self.robot_name}/env_{env_idx}/marker_{cell_idx}"
                self._prim_paths.append(prim_path)
                if not stage.GetPrimAtPath(prim_path).IsValid():
                    self._prim_utils.define_prim(prim_path, "Sphere")
                radius_attr = self._UsdGeom.Sphere.Get(stage, prim_path).GetRadiusAttr()
                radius_attr.Set(self.marker_radius)
                # set a light blue display color
                color = self._Gf.Vec3f(1.0, 0.3, 0.3)
                sphere = self._UsdGeom.Sphere.Get(stage, prim_path)
                sphere.GetDisplayColorAttr().Set([color])

    def update(self,
               base_positions: torch.Tensor,
               base_quats: torch.Tensor,
               heights: torch.Tensor,
               env_indxs: torch.Tensor = None):
        """
        base_positions: (N,3) world
        base_quats: (N,4) wxyz world (unused here but kept for interface symmetry)
        heights: (N, G, G)
        env_indxs: optional indices to update subset (absolute env IDs)
        """
        if base_positions.numel() == 0:
            return
        if env_indxs is None:
            env_ids_real = torch.arange(base_positions.shape[0], device=base_positions.device)
        else:
            env_ids_real = env_indxs
        local_ids = torch.arange(base_positions.shape[0], device=base_positions.device)

        # rotation matrices (planar part)
        rot_mats = self._quat_to_rotmat(base_quats)  # (N,3,3)
        rot_xy = rot_mats[:, :2, :2]  # (N,2,2)

        offsets = self._grid_offsets.to(device=base_positions.device, dtype=base_positions.dtype)
        base_off = self._base_offset.to(device=base_positions.device, dtype=base_positions.dtype)
        offsets = (offsets + base_off).unsqueeze(0).expand(local_ids.numel(), -1, -1)  # (n_envs, P, 2)

        world_xy = torch.bmm(offsets, rot_xy.transpose(1, 2))
        world_xy += base_positions[:, :2].unsqueeze(1)

        z = heights.reshape(local_ids.numel(), -1, 1)
        world_pos = torch.cat([world_xy, z], dim=-1)  # (n_envs, P, 3)

        world_pos_cpu = world_pos.detach().cpu().reshape(local_ids.numel(), self.grid_size * self.grid_size, 3).numpy()

        stage = self._get_stage()
        for i, env_real in enumerate(env_ids_real.tolist()):
            for j in range(self.grid_size * self.grid_size):
                prim_index = env_real * self.grid_size * self.grid_size + j
                prim_path = self._prim_paths[prim_index]
                pos = world_pos_cpu[i, j, :]
                self._set_world_position(stage, prim_path, pos)

    def _set_world_position(self, stage, prim_path: str, position):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        xform = self._UsdGeom.Xformable(prim)
        # find or create a translate op
        translate_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == self._UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if translate_op is None:
            translate_op = xform.AddTranslateOp()
        pos_vec = self._Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
        translate_op.Set(value=pos_vec)

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
