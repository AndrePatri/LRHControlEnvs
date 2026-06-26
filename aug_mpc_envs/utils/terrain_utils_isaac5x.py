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
import numpy as np
import torch

# Backend-neutral terrain primitives (SubTerrain, *_terrain heightfield generators,
# convert_heightfield_to_trimesh, ...) now live in terrain_primitives.py so they can be reused by
# non-Isaac backends (e.g. Genesis) without importing isaacsim/pxr. They are re-exported here so all
# existing Isaac code (which does `from terrain_utils_isaac5x import *`) keeps working unchanged.
from aug_mpc_envs.utils.terrain_primitives import (
    SubTerrain,
    random_uniform_terrain,
    sloped_terrain,
    discrete_obstacles_terrain,
    wave_terrain,
    stairs_terrain,
    pyramid_stairs_terrain,
    stepping_stones_terrain,
    convert_heightfield_to_trimesh,
)

# Isaac/USD-specific authoring stays here (needs isaacsim + pxr).
from isaacsim.core.prims import XFormPrim
from pxr import UsdPhysics, Sdf, Gf, PhysxSchema, UsdGeom


def add_terrain_to_stage(stage, vertices, triangles, position=None, orientation=None,
                         prim_path: str = "/World/terrain",
            dynamic_friction=0.5, static_friction=0.5,
            restitution=0.1, density=1000):
    num_faces = triangles.shape[0]

    # Create the terrain mesh in the stage
    terrain_mesh = stage.DefinePrim(prim_path, "Mesh")
    terrain_mesh.GetAttribute("points").Set(vertices)
    terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
    terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

    # Create an Xform (transform) for the terrain, if position/ orientation is specified
    # terrain = UsdGeom.Xform.Define(stage, prim_path)
    # if position is not None:
    #     terrain.AddTranslateOp().Set(value=position)
    # if orientation is not None:
    #     terrain.AddRotateXYZOp().Set(value=orientation)
    # Isaac 5's XFormPrim uses a torch backend (set_world_poses -> .detach()), so poses must be
    # torch tensors, not numpy arrays.
    _pose_device = "cuda" if torch.cuda.is_available() else "cpu"
    terrain = XFormPrim(prim_paths_expr=prim_path,
                        name="terrain",
                        positions=torch.as_tensor(np.asarray([position]), dtype=torch.float32, device=_pose_device)
                            if position is not None else None,
                        orientations=torch.as_tensor(np.asarray([orientation]), dtype=torch.float32, device=_pose_device)
                            if orientation is not None else None)
    terrain_prim = terrain.prims[0]
    # Apply Collision API for the terrain
    collision_api = UsdPhysics.CollisionAPI.Apply(terrain_prim)

    # For PhysX, use PhysxCollisionAPI to adjust properties for collision behavior
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain_prim)
    physx_collision_api.GetContactOffsetAttr().Set(0.02)
    physx_collision_api.GetRestOffsetAttr().Set(0.00)

    # Add Physics Material properties (friction, restitution, density)
    physics_material=UsdPhysics.MaterialAPI.Apply(terrain_prim)
    physics_material.CreateDynamicFrictionAttr().Set(dynamic_friction)
    physics_material.CreateStaticFrictionAttr().Set(static_friction)
    physics_material.CreateRestitutionAttr().Set(restitution)
    physxMaterialAPI=PhysxSchema.PhysxMaterialAPI.Apply(terrain_prim)
    physxMaterialAPI.CreateFrictionCombineModeAttr().Set("multiply") # average, min, multiply, max
    physxMaterialAPI.CreateRestitutionCombineModeAttr().Set("multiply")

    wrapper = type("TerrainWrapper", (), {})()
    wrapper.prim = terrain_prim
    wrapper.prim_path = prim_path
    wrapper.view = terrain
    return wrapper
