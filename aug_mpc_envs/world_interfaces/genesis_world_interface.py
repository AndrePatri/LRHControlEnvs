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
# Genesis world interface (https://github.com/Genesis-Embodied-AI/genesis-world).
# It wraps adarl's vectorized GenesisJointImpedanceAdapter, mirroring the XMJ adapter path.
# Current scope: vectorized environments, joint/base state, optional render-env switching,
# optional root perturbations, partial environment reset, and optional contact-force readout.
# Terrain and heightmap sensing are not implemented here yet.
from typing import Dict, List
from typing_extensions import override

import math
import queue
import sys
import threading
import xml.etree.ElementTree as ET

import numpy as np
import torch

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase

from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.adapters.BaseVecAdapter import JointType
from adarl.utils.utils import build_pose

from mpc_hive.utilities.math_utils_torch import world2base_frame, world2base_frame3D

from aug_mpc_envs.utils.math_utils import quat_to_omega
from aug_mpc_envs.utils.genesis_jnt_imp_cntrl import GenesisJntImpCntrl
from aug_mpc_envs.utils.height_sensor import HeightGridSensor
from aug_mpc_envs.utils import terrain_generation


class GenesisSim(AugMPCWorldInterfaceBase):

    def __init__(self,
        robot_names: List[str],
        robot_urdf_paths: List[str],
        robot_srdf_paths: List[str],
        jnt_imp_config_paths: List[str],
        n_contacts: List[int],
        cluster_dt: List[float],
        use_remote_stepping: List[bool],
        name: str = "GenesisSim",
        num_envs: int = 1,
        debug=False,
        verbose: bool = False,
        vlevel: VLevel = VLevel.V1,
        n_init_step: int = 0,
        timeout_ms: int = 60000,
        env_opts: Dict = None,
        use_gpu: bool = True,
        dtype: torch.dtype = torch.float32,
        override_low_lev_controller: bool = False):

        if not len(robot_names) == 1:
            Journal.log(self.__class__.__name__,
                "__init__",
                "Multi-robot simulation is not supported yet!",
                LogType.EXCEP,
                throw_when_excep=True)

        self._genesis_adapter: GenesisJointImpedanceAdapter = None
        self._isrunning = False
        self._step_counter = 0
        self._render_env_idx = 0
        self._render_env_cmd_queue = None
        self._render_env_thread = None
        # random perturbations (set up in _configure_scene when use_random_pertub)
        self._pert_steps_remaining = None
        self._pert_force_world = None
        self._pert_torque_world = None
        self._pert_det_counter = None
        self._pert_det_steps = 1
        self._robot_weight = 0.0
        # contact sensing: name -> index in the robot entity's link list (lazily built)
        self._contact_link_idx_cache = None
        # terrain + heightmap sensing (set up in _configure_scene when not flat ground)
        self._terrain_data = None
        self._terrain_use_boxes = False
        self._height_sensors = {}
        self._height_imgs = {}
        # heightmap debug viz (genesis debug spheres for the rendered env)
        self._height_vis_node = None
        self._height_vis_step = 0

        super().__init__(name=name,
            robot_names=robot_names,
            robot_urdf_paths=robot_urdf_paths,
            robot_srdf_paths=robot_srdf_paths,
            jnt_imp_config_paths=jnt_imp_config_paths,
            n_contacts=n_contacts,
            cluster_dt=cluster_dt,
            use_remote_stepping=use_remote_stepping,
            num_envs=num_envs,
            debug=debug,
            verbose=verbose,
            vlevel=vlevel,
            n_init_step=n_init_step,
            timeout_ms=timeout_ms,
            env_opts=env_opts,
            use_gpu=use_gpu,
            dtype=dtype,
            override_low_lev_controller=override_low_lev_controller)

    # ------------------------------------------------------------------ setup

    def _pre_setup(self):
        self._render = (not self._env_opts["headless"])

    def _parse_env_opts(self):
        g_opts = {}
        g_opts["use_gpu"] = True
        g_opts["device"] = "cuda" if g_opts["use_gpu"] else "cpu"
        g_opts["sim_device"] = g_opts["device"]
        g_opts["physics_dt"] = 1.0 / 256.0
        g_opts["rendering_dt"] = g_opts["physics_dt"]
        g_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        g_opts["use_diff_vels"] = False
        g_opts["headless"] = True
        g_opts["base_linkname"] = "base_link"
        g_opts["spawning_height"] = 1.0
        g_opts["add_ground"] = True
        g_opts["genesis_fixed_base"] = False
        g_opts["genesis_merge_fixed_links"] = True
        g_opts["genesis_logging_level"] = "warning"
        # genesis contact/solver knobs (forwarded as RigidOptions to the adapter). These are the
        # genesis defaults spelled out explicitly so they are easy to tweak. The contact model is a
        # MuJoCo-style soft constraint (solref/solimp): pyramidal friction (4 constraints/contact),
        # iterative Newton/CG solve, compliance set by constraint_timeconst. constraint_solver and
        # integrator are passed as strings and converted to genesis enums by the adapter.
        # NOTE: dt/gravity come from SimOptions (physics_dt); substeps (SimOptions, default 1) and
        #       per-geom friction are NOT here (friction is a material/geom property, not RigidOptions).
        g_opts["genesis_rigid_options"] = {
            # --- constraint solver ---
            "constraint_solver": "Newton",              # "Newton" or "CG"
            "iterations": 100,                            # main solver iterations
            "tolerance": None,                           # None -> auto from float precision
            "ls_iterations": 50,                         # line-search iterations
            "ls_tolerance": 1e-2,
            "integrator": "Euler",    # "Euler" | "implicitfast" | "approximate_implicitfast"
            # --- contact compliance (MuJoCo solref) ---
            "constraint_timeconst": 0.005,                # lower -> stiffer contact (>= 2*physics_dt)
            # --- friction drift cleanup (post-pass, off by default) ---
            "noslip_iterations": 0,                      # >0 suppresses tangential slip/drift
            "noslip_tolerance": 1e-6,
            # --- collision / contact generation ---
            "enable_collision": True,
            "enable_self_collision": True,
            "enable_joint_limit": True,
            "enable_neutral_collision": False,
            "enable_adjacent_collision": False,
            "enable_multi_contact": True,                # multi-point manifolds (support polygon)
            "box_box_detection": False,
            "max_collision_pairs": 150,
            "max_contacts": None,                        # None -> auto-sized
            "contact_pruning_tolerance": 0.02,
            "multiplier_collision_broad_phase": 8,
            # --- misc ---
            "disable_constraint": False,
            "use_contact_island": False,
            "sparse_solve": None,                        # None -> auto (CPU only)
            "use_gjk_collision": None,                   # None -> auto
            "enable_mujoco_compatibility": False,
        }
        # rendering: vis_mode is "visual" or "collision"; visualize_contact draws per-link contact
        # force arrows in the viewer (needs headless=False). contact_force_scale is m/N: the genesis
        # default (0.02) makes hundreds-of-N foot forces meters long, so use a much smaller scale.
        g_opts["genesis_vis_mode"] = "visual"
        g_opts["genesis_visualize_contact"] = True
        g_opts["genesis_contact_force_scale"] = 0.001
        g_opts["genesis_render_env_idx"] = 0
        g_opts["genesis_render_envs_idx"] = None
        g_opts["genesis_enable_camera_rendering"] = False
        g_opts["genesis_use_batch_renderer"] = False
        g_opts["genesis_render_env_keyboard"] = False
        # Global contact constraint params (MuJoCo solref+solimp), applied to all geoms after build.
        # 7-vec [timeconst, dampratio, dmin, dmax, width, mid, power]. RigidOptions only exposes the
        # global timeconst (constraint_timeconst), NOT dmin/dmax, so the near-rigid MuJoCo foot
        # impedance (solimp dmin=dmax=0.995) can only be reproduced here. This matches the XMJ Talos
        # robot-geom contact (solref="0.005 1.2", solimp="0.995 0.995 0.001 0.5 2"). Set to None to
        # leave the genesis defaults ([0,1,0.9,0.95,1e-3,0.5,2]).
        g_opts["genesis_global_sol_params"] = [0.005, 1.2, 0.995, 0.995, 1e-3, 0.5, 2.0]
        # random base perturbations (external pushes), mirroring the isaac5x interface. Forces are
        # sampled relative to the robot weight; applied to the base link via the adapter impulse API.
        # Impulse-based (matches isaac5x): a push is sampled as a target IMPULSE (= m*delta_v), then
        # converted to a per-step force = impulse/duration, so the resulting velocity change is ~delta_v
        # regardless of physics_dt and duration. The *weight_scale clips only cap the peak force.
        g_opts["use_random_pertub"] = False
        g_opts["pert_planar_only"] = True          # linear xy pushes only, no torque
        g_opts["pert_wrenches_rate"] = 15.0        # ~1 push every N seconds (per env)
        g_opts["pert_wrenches_min_duration"] = 0.25
        g_opts["pert_wrenches_max_duration"] = 0.6
        g_opts["pert_target_delta_v"] = 0.4        # [m/s] max base velocity change a push imparts
        g_opts["lin_impulse_mag_min"] = 0.5        # sampled impulse = [min,max] * (m*delta_v)
        g_opts["lin_impulse_mag_max"] = 1.0
        g_opts["max_ang_impulse_lever"] = 0.2      # [m] lever turning delta_v into an angular impulse
        g_opts["pert_force_min_weight_scale"] = 0.0  # clip the derived force norm to [min,max]*weight
        g_opts["pert_force_max_weight_scale"] = 1.2
        g_opts["pert_torque_max_weight_scale"] = 1.0 # clip torque to scale*weight*lever (non-planar only)
        g_opts["det_pert_rate"] = True             # deterministic spacing vs poisson
        # measured contact wrenches: list of ROBOT ENTITY LINK names (not virtual/merged frames) whose
        # net contact force (world frame) is read and written to the cluster contact_wrenches, in the
        # MPC's contact order. Empty -> no measured contacts (generic cluster names; the MPC uses its
        # own contact estimate, matching isaac with an empty contact_prims).
        g_opts["contact_prims"] = []

        # ------------------------------------------------------------------ terrain
        # Heightfield terrain, mirroring the isaac5x ground opts. Genesis builds the terrain (visual
        # mesh + SDF collision) directly from a heightfield (gs.morphs.Terrain), so every type reduces
        # to "generate heightfield -> hand to genesis"; no per-box authoring. Because genesis envs are
        # vectorized in place (not spatially tiled like isaac), the terrain is shared by all envs and
        # can stay SMALL (a few metres), unlike isaac's hundreds-of-metres ground.
        g_opts["use_flat_ground"] = True
        # ground_type: flat | random | random_patches | slopes | stairs | stepup | stepup_prim | random_tiles
        g_opts["ground_type"] = "flat"
        g_opts["ground_size"] = 8.0                 # [m] square terrain side (keep small for genesis)
        g_opts["static_friction"] = 0.5             # genesis uses a single friction coeff (no dyn/restitution split)
        g_opts["dynamic_friction"] = 0.5            # kept for config parity with isaac (unused by genesis)
        g_opts["restitution"] = 0.1                 # kept for config parity with isaac (unused by genesis)
        g_opts["dh_ground"] = 0.05                  # +/- height range for "random"/"random_patches" [m]
        g_opts["terrain_walls"] = True              # add perimeter walls so the robot can't walk off
        g_opts["wall_height"] = 2.0                 # [m]
        g_opts["slope"] = -0.5                      # "slopes"
        g_opts["stairs_step_width"] = 0.3           # "stairs"
        g_opts["stairs_step_height"] = -0.1
        # stepup / stepup_prim args (mirror isaac names)
        g_opts["step_stairs_ratio"] = 0.9
        g_opts["step_platform_size"] = 3.0
        g_opts["step_height_lb"] = 0.08
        g_opts["step_height_ub"] = 0.15
        g_opts["step_n"] = 1
        g_opts["step_min"] = 1
        g_opts["step_max"] = 1
        g_opts["step_area_factor"] = 0.7
        g_opts["step_random_n_steps"] = False
        g_opts["step_width_lb"] = None
        g_opts["step_width_ub"] = None
        # random_tiles args (dense grid of small flat-topped cells at random heights)
        g_opts["tile_cell_size"] = 0.5
        g_opts["tile_height_lb"] = 0.0
        g_opts["tile_height_ub"] = 0.06
        g_opts["tile_patch_ratio"] = 0.0
        g_opts["tile_patch_size"] = 3.0
        g_opts["tile_min_height"] = 0.01
        # terrain rendering: vis mode for the terrain entity ("visual" | "collision" | None=default)
        g_opts["terrain_vis_mode"] = None
        # ------------------------------------------------------- terrain collider representation
        # The "prim" terrains (random_tiles, stepup_prim) can be built as PRIMITIVE BOX colliders
        # (one fixed gs.morphs.Box per tile/step + a base slab + walls), mirroring isaac's *_prim
        # terrains. Boxes give exact, crisp contact (no SDF quantization) and are cheap for SPARSE
        # terrain (few large steps). They are NOT free, though: each box is a separate genesis entity,
        # so a DENSE terrain (e.g. random_tiles with small cells over a big area) becomes thousands of
        # entities -> slow build, broadphase/GPU-memory pressure. So box terrains are capped: if the
        # generated box count exceeds terrain_max_boxes, the interface falls back to the heightfield
        # (SDF) path. Box terrains never get downsampled (no resolution loss). The height sensor always
        # samples the heightfield, regardless of the collision representation.
        g_opts["terrain_primitive_colliders"] = True   # use box colliders for random_tiles/stepup_prim
        g_opts["terrain_max_boxes"] = 1500             # over this -> fall back to the heightfield path
        # terrain SDF cost cap (heightfield path only). genesis builds the heightfield terrain collision
        # as ONE mesh + an auto-SDF; the SDF pre-processing gets very slow past ~50k faces (face count =
        # 2*(grid_side-1)^2). When a generated heightfield exceeds this it is auto-coarsened (max-pool,
        # preserving raised steps) -- which costs resolution, so prefer box colliders or a smaller
        # ground_size for terrains with sharp features. Set <= 0 to disable the cap.
        g_opts["terrain_max_faces"] = 120000
        # spawn-height adjustment: lift the robot by the max terrain height under it (+ cushion)
        g_opts["spawn_height_check_half_extent"] = 0.45
        g_opts["spawn_height_cushion"] = 0.06

        # ------------------------------------------------------------- height sensor
        # Square height-grid sensor around the base, in the base frame, sampled from the terrain
        # heightfield (see HeightGridSensor). Output is cached in self._height_imgs[robot_name] and
        # exposed via get_height_images(); on flat ground it reads all-zeros.
        g_opts["enable_height_sensor"] = False
        g_opts["height_sensor_pixels"] = 16         # grid side (pixels)
        g_opts["height_sensor_resolution"] = 0.1    # [m] per pixel
        g_opts["height_sensor_forward_offset"] = 0.0
        g_opts["height_sensor_lateral_offset"] = 0.0
        # heightmap visualization: draw the sampled height-grid points as debug spheres in the genesis
        # viewer (rendered env only). Needs a live visualizer (not headless, or camera rendering on).
        g_opts["enable_height_vis"] = False
        g_opts["height_vis_radius"] = 0.03          # [m] sphere radius
        g_opts["height_vis_update_period"] = 1      # redraw every N sim steps
        g_opts["height_vis_color"] = [0.1, 0.9, 0.2, 0.8]  # RGBA

        g_opts.update(self._env_opts)  # override defaults with provided opts

        if g_opts["use_diff_vels"]:
            Journal.log(self.__class__.__name__,
                "_parse_env_opts",
                "Genesis interface does not support use_diff_vels yet. Use simulator joint velocities until finite-difference timing is implemented.",
                LogType.EXCEP,
                throw_when_excep=True)

        if not g_opts["use_gpu"]:
            g_opts["device"] = "cpu"
            g_opts["sim_device"] = "cpu"

        self._env_opts = g_opts
        self._device = g_opts["device"]
        self._use_gpu = g_opts["use_gpu"]

    def _init_world(self):
        self._configure_scene()

    def _configure_scene(self):

        robot_name = self._robot_names[0]
        base_link = self._env_opts["base_linkname"]

        # generate URDF/SRDF from the xacro sources (handled by the base class)
        self._generate_rob_descriptions(robot_name=robot_name,
            urdf_path=self._robot_urdf_paths[robot_name],
            srdf_path=self._robot_srdf_paths[robot_name])
        with open(self._urdf_dump_paths[robot_name], "r", encoding="utf-8") as f:
            urdf_str = f.read()

        # Use the URDF effort limits as the clamp, like the other interfaces do.
        max_torques = self._parse_urdf_effort_limits(urdf_str, robot_name)

        render_envs_idx = self._env_opts["genesis_render_envs_idx"]
        if render_envs_idx is None:
            render_envs_idx = [int(self._env_opts["genesis_render_env_idx"])]
        elif isinstance(render_envs_idx, (int, float)):
            render_envs_idx = [int(render_envs_idx)]
        self._render_env_idx = int(render_envs_idx[0]) if len(render_envs_idx) > 0 else 0

        # build the (shared) terrain heightfield before the adapter, so the spawn height can be lifted
        # above it and the heightfield can be handed to the adapter at build time (scene is static
        # after build). None when flat -> the adapter keeps the default flat ground plane.
        self._terrain_data = None
        self._terrain_use_boxes = False
        if not bool(self._env_opts["use_flat_ground"]):
            self._terrain_data = terrain_generation.build_terrain_data(
                ground_type=self._env_opts["ground_type"],
                ground_size=float(self._env_opts["ground_size"]),
                opts=self._env_opts,
                center=(0.0, 0.0, 0.0))
        if self._terrain_data is not None:
            # Prefer primitive BOX colliders for the "prim" terrains (exact, crisp contact, no SDF
            # quantization) when the box count is reasonable; otherwise use the heightfield (+ SDF).
            n_boxes = self._terrain_data.num_boxes()
            max_boxes = int(self._env_opts["terrain_max_boxes"])
            if bool(self._env_opts["terrain_primitive_colliders"]) and n_boxes > 0 and n_boxes <= max_boxes:
                self._terrain_use_boxes = True
                Journal.log(self.__class__.__name__, "_configure_scene",
                    f"Terrain '{self._env_opts['ground_type']}' uses {n_boxes} primitive box colliders "
                    f"(exact contact, no SDF/downsampling).", LogType.INFO, throw_when_excep=False)
            else:
                if n_boxes > max_boxes:
                    Journal.log(self.__class__.__name__, "_configure_scene",
                        f"Terrain '{self._env_opts['ground_type']}' would need {n_boxes} box colliders "
                        f"(> terrain_max_boxes={max_boxes}); using the heightfield (SDF) path instead. "
                        f"Use a smaller ground_size / coarser tiles, or raise terrain_max_boxes.",
                        LogType.WARN, throw_when_excep=False)
                # heightfield path: cap the SDF cost. genesis builds one mesh + SDF, very slow past
                # ~50k faces. Auto-coarsen (max-pool, preserving raised steps) when over the budget.
                max_faces = int(self._env_opts["terrain_max_faces"])
                if max_faces > 0 and self._terrain_data.num_faces() > max_faces:
                    max_side = max(2, int(math.sqrt(max_faces / 2.0)) + 1)
                    before = self._terrain_data.heightfield_raw.shape
                    before_faces = self._terrain_data.num_faces()
                    self._terrain_data = self._terrain_data.coarsened(max_side)
                    Journal.log(self.__class__.__name__, "_configure_scene",
                        f"Terrain '{self._env_opts['ground_type']}' heightfield {before} ({before_faces} faces) "
                        f"exceeds terrain_max_faces={max_faces}; coarsened to "
                        f"{self._terrain_data.heightfield_raw.shape} ({self._terrain_data.num_faces()} faces, "
                        f"horizontal_scale={self._terrain_data._horizontal_scale:.3f} m) to keep the genesis "
                        f"SDF build fast. Prefer box colliders (terrain_primitive_colliders) or a smaller "
                        f"ground_size/coarser resolution for sharp terrain.", LogType.WARN, throw_when_excep=False)
        # spawn-height offset: lift the robot by the max terrain height under the spawn xy (+ cushion).
        # genesis envs share world coords, so all envs spawn at the same (0,0).
        terrain_spawn_h = 0.0
        if self._terrain_data is not None:
            terrain_spawn_h = self._terrain_data.get_max_height_in_rect(
                0.0, 0.0, half_extent=float(self._env_opts["spawn_height_check_half_extent"])) \
                + float(self._env_opts["spawn_height_cushion"])
        # keep the flat ground plane only when no terrain is used
        add_ground_flag = bool(self._env_opts["add_ground"]) and (self._terrain_data is None)

        # create the adapter (sim + impedance). sim_step_dt == step_length_sec so a single
        # adapter.step() advances exactly one physics step (physics_dt), like XMJ.
        self._genesis_adapter = GenesisJointImpedanceAdapter(
            vec_size=self._num_envs,
            output_th_device=torch.device(self._device),
            sim_step_dt=self._env_opts["physics_dt"],
            step_length_sec=self._env_opts["physics_dt"],
            enable_rendering=bool(self._env_opts["genesis_enable_camera_rendering"]),
            render_envs_idx=render_envs_idx,
            add_ground=add_ground_flag,
            show_gui=(not self._env_opts["headless"]),
            max_joint_impedance_ctrl_torques=max_torques,
            rigid_options_override=self._env_opts["genesis_rigid_options"],
            vis_options_override={"contact_force_scale": float(self._env_opts["genesis_contact_force_scale"])},
            reference_filter_mode="none",  # run the impedance refs unfiltered by default
            genesis_logging_level=self._env_opts["genesis_logging_level"],
            use_batch_renderer=bool(self._env_opts["genesis_use_batch_renderer"]))

        spawn_pose = build_pose(0.0, 0.0,
            float(self._env_opts["spawning_height"]) + float(terrain_spawn_h), 0.0, 0.0, 0.0, 1.0)
        model = ModelSpawnDef(
            name=robot_name,
            definition_string=urdf_str,
            format="urdf",
            pose=spawn_pose,
            kwargs={"genesis_fixed": bool(self._env_opts["genesis_fixed_base"]),
                    "genesis_merge_fixed_links": bool(self._env_opts["genesis_merge_fixed_links"]),
                    "genesis_vis_mode": self._env_opts["genesis_vis_mode"],
                    "genesis_visualize_contact": bool(self._env_opts["genesis_visualize_contact"])})
        # genesis scenes are static after build: all models (and the terrain) must be passed to
        # build_scenario. The terrain dict is backend-neutral (the adapter turns it into a
        # gs.morphs.Terrain); cell (0,0) lands at terrain position so it lines up with the sensor.
        build_kwargs = {}
        if self._terrain_data is not None:
            if self._terrain_use_boxes:
                build_kwargs["terrain"] = {
                    "boxes": self._terrain_data.boxes,
                    "friction": float(self._env_opts["static_friction"]),
                    "vis_mode": self._env_opts["terrain_vis_mode"],
                    "visualize_contact": False,
                    "name": "terrain",
                }
            else:
                build_kwargs["terrain"] = {
                    "height_field": self._terrain_data.heightfield_raw,
                    "horizontal_scale": self._terrain_data._horizontal_scale,
                    "vertical_scale": self._terrain_data.vertical_scale,
                    "pos": tuple(float(v) for v in self._terrain_data.position),
                    "friction": float(self._env_opts["static_friction"]),
                    "vis_mode": self._env_opts["terrain_vis_mode"],
                    "visualize_contact": False,
                    "name": "terrain",
                }
        self._genesis_adapter.build_scenario(models=[model], **build_kwargs)

        # apply MuJoCo-style global contact constraint params (solref+solimp) to all geoms; this is
        # the only way to reach the near-rigid foot impedance the XMJ/MuJoCo Talos uses (RigidOptions
        # only exposes the global timeconst). Must run after build_scenario.
        if self._env_opts["genesis_global_sol_params"] is not None:
            self._genesis_adapter.set_global_sol_params(self._env_opts["genesis_global_sol_params"])

        # detected joints come back as (model_name, joint_name); keep only this robot's
        detected = self._genesis_adapter.get_detected_joints()
        properties = self._genesis_adapter.get_detected_joints_properties()
        supported_types = {JointType.REVOLUTE, JointType.PRISMATIC}
        self._robot_joints = [
            tuple(j) for j in detected
            if j[0] == robot_name
            and properties[tuple(j)].joint_type in supported_types
        ]
        self._robot_jnames = [j[1] for j in self._robot_joints]
        self._base_link_id = (robot_name, base_link)

        self._genesis_adapter.set_monitored_joints(self._robot_joints)
        self._genesis_adapter.set_monitored_links([self._base_link_id])
        self._genesis_adapter.set_impedance_controlled_joints(self._robot_joints)

        self._init_robots_state()

        # height-grid sensor: samples the terrain heightfield around the base (base frame). On flat
        # ground the sensor reads all-zeros. Shares the exact heightfield/transform genesis built from.
        if bool(self._env_opts["enable_height_sensor"]):
            pixels = int(self._env_opts["height_sensor_pixels"])
            self._height_sensors[robot_name] = HeightGridSensor(
                terrain_utils=self._terrain_data if not bool(self._env_opts["use_flat_ground"]) else None,
                grid_size=pixels,
                resolution=float(self._env_opts["height_sensor_resolution"]),
                n_envs=self._num_envs,
                forward_offset=float(self._env_opts["height_sensor_forward_offset"]),
                lateral_offset=float(self._env_opts["height_sensor_lateral_offset"]),
                device=self._device,
                dtype=self._dtype)
            self._height_imgs[robot_name] = torch.zeros((self._num_envs, pixels, pixels),
                device=self._device, dtype=self._dtype)

        if self._env_opts["use_random_pertub"]:
            self._setup_perturbations(robot_name)

        self._reset_sim()

        self._fill_robot_info_from_world()

        # contact wrenches: set the cluster's contact link names from contact_prims (entity link names)
        # so _get_contact_f reads their net contact force. Runs in _init_world (before the base creates
        # the cluster server with contact_linknames=_contact_names). Empty -> None (generic, no measured
        # contacts). The names must be real entity links in the MPC's contact order.
        cp = self._env_opts["contact_prims"]
        if isinstance(cp, str):  # config passes it as a comma-separated string (custom args are scalar)
            cp = [s.strip() for s in cp.split(",") if s.strip()]
        self._contact_names[robot_name] = list(cp) if cp else None

        self._start_render_env_keyboard_control()

        self._isrunning = True

    def _parse_urdf_effort_limits(self, urdf_str: str, robot_name: str) -> Dict:
        """Map each actuated joint to its URDF effort limit, keyed by (robot_name, joint_name).
        Used as the adapter's per-joint torque clamp so heavy joints can hold the stance."""
        limits = {}
        root = ET.fromstring(urdf_str)
        for joint in root.findall("joint"):
            if joint.get("type") not in ("revolute", "prismatic"):
                continue
            limit = joint.find("limit")
            if limit is None or limit.get("effort") is None:
                continue
            limits[(robot_name, joint.get("name"))] = float(limit.get("effort"))
        return limits

    def _fill_robot_info_from_world(self):
        pass

    # ------------------------------------------------------------- state I/O

    def _base_state(self, robot_name: str):
        """Return (p[E,3], q_wxyz[E,4], v[E,3], omega[E,3]) for the base link."""
        ls = self._genesis_adapter.getLinksState([self._base_link_id])  # (E,1,13): pos,quat_xyzw,linvel,angvel
        ls = ls[:, 0, :].to(self._dtype)
        p = ls[:, 0:3]
        q_xyzw = ls[:, 3:7]
        q_wxyz = q_xyzw[:, [3, 0, 1, 2]]  # adarl/IBRIDO convention is [w,x,y,z]
        v = ls[:, 7:10]
        omega = ls[:, 10:13]
        return p, q_wxyz, v, omega

    def _joints_state(self, robot_name: str):
        """Return (q[E,n], v[E,n], eff[E,n]) for the monitored joints."""
        js = self._genesis_adapter.getJointsState(self._robot_joints)  # (E,n,3): pos,vel,eff
        js = js.to(self._dtype)
        return js[:, :, 0], js[:, :, 1], js[:, :, 2]

    def _init_robots_state(self):
        self._p_ref_reset = {}
        for robot_name in self._robot_names:
            p, q, v, omega = self._base_state(robot_name)
            jq, jv, jeff = self._joints_state(robot_name)

            self._root_p[robot_name] = p.clone()
            self._root_p_prev[robot_name] = p.clone()
            self._root_p_default[robot_name] = p.clone()
            self._root_q[robot_name] = q.clone()
            self._root_q_prev[robot_name] = q.clone()
            self._root_q_default[robot_name] = q.clone()

            self._root_v[robot_name] = v.clone()
            self._root_v_base_loc[robot_name] = torch.zeros_like(v)
            self._root_v_prev[robot_name] = torch.zeros_like(v)
            self._root_v_default[robot_name] = v.clone()
            self._root_omega[robot_name] = omega.clone()
            self._root_omega_prev[robot_name] = torch.zeros_like(omega)
            self._root_omega_base_loc[robot_name] = torch.zeros_like(omega)
            self._root_omega_default[robot_name] = omega.clone()

            self._root_a[robot_name] = torch.zeros_like(v)
            self._root_a_base_loc[robot_name] = torch.zeros_like(v)
            self._root_alpha[robot_name] = torch.zeros_like(v)
            self._root_alpha_base_loc[robot_name] = torch.zeros_like(v)

            self._jnts_q[robot_name] = jq.clone()
            self._jnts_q_prev[robot_name] = jq.clone()
            self._jnts_q_default[robot_name] = jq.clone()
            self._jnts_v[robot_name] = jv.clone()
            self._jnts_v_default[robot_name] = jv.clone()
            self._jnts_eff[robot_name] = jeff.clone()
            self._jnts_eff_default[robot_name] = jeff.clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), device=self._device)
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0

            self._p_ref_reset[robot_name] = jq.clone()

    def _read_root_state_from_robot(self, robot_name: str, env_indxs: torch.Tensor = None):
        p, q, v, omega = self._base_state(robot_name)
        dt = self._cluster_dt[robot_name]

        self._root_p[robot_name][:, :] = p
        self._root_q[robot_name][:, :] = q

        if not self._env_opts["use_diff_vels"]:
            self._root_v[robot_name][:, :] = v
            self._root_omega[robot_name][:, :] = omega
        else:
            self._root_v[robot_name][:, :] = (p - self._root_p_prev[robot_name]) / dt
            self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name], q, dt)

        # world-frame accelerations (numerical, at cluster rate), like the XMJ interface
        self._root_a[robot_name][:, :] = (self._root_v[robot_name] - self._root_v_prev[robot_name]) / dt
        self._root_alpha[robot_name][:, :] = (self._root_omega[robot_name] - self._root_omega_prev[robot_name]) / dt

        # The MPC consumes the root twist/accel/gravity in the BASE frame (base_loc=True). Without
        # these the cluster sees a still, gravity-aligned robot and cannot perceive/correct tilt,
        # so Talos falls. Rotate world quantities into the base frame, mirroring the XMJ interface.
        twist_w = torch.cat((self._root_v[robot_name], self._root_omega[robot_name]), dim=1)
        twist_bl = torch.cat((self._root_v_base_loc[robot_name], self._root_omega_base_loc[robot_name]), dim=1)
        world2base_frame(t_w=twist_w, q_b=self._root_q[robot_name], t_out=twist_bl)
        self._root_v_base_loc[robot_name] = twist_bl[:, 0:3]
        self._root_omega_base_loc[robot_name] = twist_bl[:, 3:6]

        a_w = torch.cat((self._root_a[robot_name], self._root_alpha[robot_name]), dim=1)
        a_bl = torch.cat((self._root_a_base_loc[robot_name], self._root_alpha_base_loc[robot_name]), dim=1)
        world2base_frame(t_w=a_w, q_b=self._root_q[robot_name], t_out=a_bl)
        self._root_a_base_loc[robot_name] = a_bl[:, 0:3]
        self._root_alpha_base_loc[robot_name] = a_bl[:, 3:6]

        world2base_frame3D(v_w=self._gravity_normalized[robot_name], q_b=self._root_q[robot_name],
            v_out=self._gravity_normalized_base_loc[robot_name])

        # update "previous" values for numerical differentiation
        self._root_p_prev[robot_name][:, :] = p
        self._root_q_prev[robot_name][:, :] = q
        self._root_v_prev[robot_name][:, :] = self._root_v[robot_name]
        self._root_omega_prev[robot_name][:, :] = self._root_omega[robot_name]

        # heightmap readout (base frame), mirroring the isaac5x interface. On flat ground -> zeros.
        if robot_name in self._height_sensors:
            if env_indxs is None:
                heights = self._height_sensors[robot_name].read(self._root_p[robot_name], self._root_q[robot_name])
                if bool(self._env_opts["use_flat_ground"]):
                    heights = heights * 0.0
                self._height_imgs[robot_name][:, :, :] = heights
            else:
                heights = self._height_sensors[robot_name].read(
                    self._root_p[robot_name][env_indxs], self._root_q[robot_name][env_indxs])
                if bool(self._env_opts["use_flat_ground"]):
                    heights = heights * 0.0
                self._height_imgs[robot_name][env_indxs] = heights.clone()

    def get_height_images(self, robot_name: str = None):
        """Return the latest height-grid images (N, pixels, pixels), or None if the sensor is off.
        With no robot_name, returns the per-robot dict."""
        if robot_name is None:
            return self._height_imgs
        return self._height_imgs.get(robot_name, None)

    def _read_jnts_state_from_robot(self, robot_name: str, env_indxs: torch.Tensor = None):
        jq, jv, jeff = self._joints_state(robot_name)
        if self._env_opts["use_diff_vels"]:
            dt = self._cluster_dt[robot_name]
            self._jnts_v[robot_name][:, :] = (jq - self._jnts_q_prev[robot_name]) / dt
        else:
            self._jnts_v[robot_name][:, :] = jv
        self._jnts_q[robot_name][:, :] = jq
        self._jnts_eff[robot_name][:, :] = jeff
        self._jnts_q_prev[robot_name][:, :] = jq

    # --------------------------------------------------------- reset / homing

    def _env_indxs_to_mask(self, env_indxs: torch.Tensor):
        """Convert an env index tensor (or None=all) to a (num_envs,) bool mask for vec_mask args."""
        if env_indxs is None:
            return None
        mask = torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device)
        mask[env_indxs] = True
        return mask

    def _set_jnts_to_homing(self, robot_name: str, vec_mask: torch.Tensor = None):
        # joints: (E, n, 3) -> [pos, vel, eff]. vec_mask (bool, num_envs) selects which envs to set.
        jq = self._jnts_q_default[robot_name]
        pve = torch.zeros((self._num_envs, len(self._robot_joints), 3), device=jq.device, dtype=jq.dtype)
        pve[:, :, 0] = jq
        self._genesis_adapter.setJointsStateDirect(self._robot_joints, pve, vec_mask=vec_mask)

    def _set_root_to_defconfig(self, robot_name: str, vec_mask: torch.Tensor = None):
        # base link: pose (p[3] + quat_xyzw[4]) + vel (lin[3] + ang[3]) -> (E, 1, 13)
        p = self._root_p_default[robot_name]
        q_wxyz = self._root_q_default[robot_name]
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        link_state = torch.zeros((self._num_envs, 1, 13), device=p.device, dtype=p.dtype)
        link_state[:, 0, 0:3] = p
        link_state[:, 0, 3:7] = q_xyzw
        self._genesis_adapter.setLinksStateDirect([self._base_link_id], link_state, vec_mask=vec_mask)

    def _reset_sim(self, env_indxs: torch.Tensor = None):
        # Full reset (env_indxs is None): resetWorld() for a clean solver state, then re-apply the
        # default joint + base config (resetWorld restores the build-time/zero spawn).
        # Subset reset (env_indxs given): write the default joint + base state ONLY to those envs via
        # vec_mask (the *StateDirect setters set pos + zero vel), WITHOUT resetWorld() -- which would
        # reset every env. This lets the base class reset individual terminated envs in isolation.
        if env_indxs is None:
            self._genesis_adapter.resetWorld()
            for robot_name in self._robot_names:
                self._set_jnts_to_homing(robot_name)
                self._set_root_to_defconfig(robot_name)
        else:
            mask = self._env_indxs_to_mask(env_indxs)
            for robot_name in self._robot_names:
                self._set_jnts_to_homing(robot_name, vec_mask=mask)
                self._set_root_to_defconfig(robot_name, vec_mask=mask)

    def _pre_warmup_step(self, robot_name: str, env_indxs: torch.Tensor = None):
        # During warmup (before the MPC produces solutions) the impedance controller only holds the
        # joints at homing; the free base has no balancing control and can tip/drift and fall,
        # especially on uneven terrain. Mirror the isaac5x interface: each warmup step zero the base
        # linear xy and the full angular velocity, keeping only the vertical (falling) velocity, so the
        # robot settles straight down in a healthy upright pose before the MPC takes over.
        self._alter_twist_warmup(robot_name=robot_name, env_indxs=env_indxs)

    def _alter_twist_warmup(self, robot_name: str, env_indxs: torch.Tensor = None):
        """Zero the base linear-xy and angular velocity (keep vertical) for the given robot/envs."""
        p, q_wxyz, v, omega = self._base_state(robot_name)
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        # link state is (E, 1, 13): pos[3] + quat_xyzw[4] + linvel[3] + angvel[3]. Re-write the
        # current pose (so it is left untouched) and a twist that keeps only linear z.
        link_state = torch.zeros((self._num_envs, 1, 13), device=p.device, dtype=p.dtype)
        link_state[:, 0, 0:3] = p
        link_state[:, 0, 3:7] = q_xyzw
        link_state[:, 0, 9] = v[:, 2]   # keep vertical linear vel; vx, vy and all angular stay zero
        vec_mask = self._env_indxs_to_mask(env_indxs)
        self._genesis_adapter.setLinksStateDirect([self._base_link_id], link_state, vec_mask=vec_mask)

    def _reset_state(self, robot_name: str, env_indxs: torch.Tensor = None, randomize: bool = False):
        if randomize:
            # randomize spawn yaw (writes _root_q_default, applied by _reset_sim -> _set_root_to_defconfig)
            self._randomize_yaw(robot_name=robot_name, env_indxs=env_indxs)
        self._reset_sim(env_indxs=env_indxs)
        if self._env_opts["use_random_pertub"]:
            self._reset_perturbations(env_indxs=env_indxs)

    def _start_render_env_keyboard_control(self):
        if self._env_opts["headless"] or not bool(self._env_opts["genesis_render_env_keyboard"]):
            return
        self._render_env_cmd_queue = queue.SimpleQueue()

        def read_commands():
            print("[GenesisSim] render env control: type n/], p/[, or an env index + Enter", flush=True)
            for line in sys.stdin:
                token = line.strip()
                if token:
                    self._render_env_cmd_queue.put(token)

        self._render_env_thread = threading.Thread(target=read_commands, daemon=True)
        self._render_env_thread.start()

    def _apply_render_env_commands(self):
        if self._render_env_cmd_queue is None:
            return
        token = None
        while True:
            try:
                token = self._render_env_cmd_queue.get_nowait()
            except queue.Empty:
                break
        if token is None:
            return
        try:
            if token in ("n", "]", "+", "next"):
                idx = self._render_env_idx + 1
            elif token in ("p", "[", "-", "prev"):
                idx = self._render_env_idx - 1
            else:
                idx = int(token)
            idx = max(0, min(self._num_envs - 1, idx))
            if idx == self._render_env_idx:
                return
            self._genesis_adapter.set_rendered_envs_idx([idx])
            self._render_env_idx = idx
            Journal.log(self.__class__.__name__, "_apply_render_env_commands",
                f"Genesis viewer now rendering env {idx}", LogType.STAT, throw_when_excep=False)
        except Exception as e:
            Journal.log(self.__class__.__name__, "_apply_render_env_commands",
                f"Could not switch Genesis render env from token '{token}': {e}", LogType.WARN, throw_when_excep=False)

    # ----------------------------------------------------- random perturbations

    def _compute_robot_weight(self, robot_name: str) -> float:
        """Total robot weight [N] = sum of link masses * g, used to scale perturbation forces."""
        g = float(abs(self._env_opts["gravity"][2]))
        try:
            rs = self._genesis_adapter._rigid_solver()
            entity = self._genesis_adapter._entities[robot_name]
            masses = rs.get_links_inertial_mass()
            masses = masses[0] if masses.dim() > 1 else masses
            total_mass = float(masses[entity.link_start:entity.link_end].sum())
        except Exception as e:
            Journal.log(self.__class__.__name__, "_compute_robot_weight",
                f"Could not read robot mass ({e}); perturbation forces will be 0.", LogType.WARN,
                throw_when_excep=False)
            total_mass = 0.0
        return total_mass * g

    def _setup_perturbations(self, robot_name: str):
        n = self._num_envs
        dev = self._device
        dt = self.physics_dt()
        self._pert_steps_remaining = torch.zeros((n,), dtype=torch.int32, device=dev)
        self._pert_force_world = torch.zeros((n, 3), dtype=self._dtype, device=dev)
        self._pert_torque_world = torch.zeros((n, 3), dtype=self._dtype, device=dev)
        rate = float(self._env_opts["pert_wrenches_rate"])
        self._pert_det_steps = max(1, int(round(rate / dt)))
        # random initial phase so deterministic-rate pushes are staggered across envs
        self._pert_det_counter = torch.randint(0, self._pert_det_steps, (n,), dtype=torch.int32, device=dev)
        # duration sampled in physics steps (accounts for dt)
        self._pert_min_steps = max(1, int(math.ceil(float(self._env_opts["pert_wrenches_min_duration"]) / dt)))
        self._pert_max_steps = max(self._pert_min_steps, int(math.ceil(float(self._env_opts["pert_wrenches_max_duration"]) / dt)))
        self._robot_weight = self._compute_robot_weight(robot_name)
        g = float(abs(self._env_opts["gravity"][2]))
        mass = self._robot_weight / g if g > 0 else 0.0
        # max linear impulse = m*delta_v ; max angular impulse = m*delta_v*lever
        self._max_lin_impulse = mass * float(self._env_opts["pert_target_delta_v"])
        self._max_ang_impulse = self._max_lin_impulse * float(self._env_opts["max_ang_impulse_lever"])

    def _reset_perturbations(self, env_indxs: torch.Tensor = None):
        if self._pert_steps_remaining is None:
            return
        n = self._num_envs
        dev = self._device
        if env_indxs is None:
            self._pert_steps_remaining.zero_()
            self._pert_force_world.zero_()
            self._pert_torque_world.zero_()
            self._pert_det_counter.copy_(torch.randint(0, self._pert_det_steps, (n,), dtype=torch.int32, device=dev))
        else:
            self._pert_steps_remaining[env_indxs] = 0
            self._pert_force_world[env_indxs, :] = 0
            self._pert_torque_world[env_indxs, :] = 0
            self._pert_det_counter[env_indxs] = torch.randint(0, self._pert_det_steps,
                (env_indxs.numel(),), dtype=torch.int32, device=dev)

    def _sample_perturbations(self, mask: torch.Tensor):
        # Impulse-based: sample target impulse (= m*delta_v), pick a duration, then force = impulse/duration
        # (so delta_v is duration/dt-independent), and clip the force/torque norm to [min,max]*weight.
        k = int(mask.sum())
        if k == 0:
            return
        dev = self._device
        w = self._robot_weight
        dt = self.physics_dt()
        # duration in steps, then seconds (>=1 step), accounts for physics_dt
        steps = torch.randint(self._pert_min_steps, self._pert_max_steps + 1, (k,), dtype=torch.int32, device=dev)
        dur_s = (steps.to(self._dtype) * dt).clamp(min=1e-6)
        # linear impulse -> force
        imp_mag = (torch.rand((k,), device=dev)
                   * (float(self._env_opts["lin_impulse_mag_max"]) - float(self._env_opts["lin_impulse_mag_min"]))
                   + float(self._env_opts["lin_impulse_mag_min"])) * self._max_lin_impulse
        if self._env_opts["pert_planar_only"]:
            ang = torch.rand((k,), device=dev) * 2 * torch.pi
            dirv = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=1)
        else:
            dirv = torch.randn((k, 3), device=dev)
            dirv = dirv / dirv.norm(dim=1, keepdim=True).clamp(min=1e-6)
        force = (dirv * imp_mag.unsqueeze(1)) / dur_s.unsqueeze(1)
        # clip force norm to [min,max]*weight
        fnorm = force.norm(dim=1, keepdim=True).clamp(min=1e-9)
        target = fnorm.clone()
        fmax = float(self._env_opts["pert_force_max_weight_scale"])
        if fmax > 0.0:
            target = torch.minimum(target, torch.full_like(target, fmax * w))
        fmin = float(self._env_opts["pert_force_min_weight_scale"])
        if fmin > 0.0:
            target = torch.maximum(target, torch.full_like(target, fmin * w))
        force = force * (target / fnorm)
        self._pert_force_world[mask] = force.to(self._dtype)
        # torque (only when not planar-only)
        if self._env_opts["pert_planar_only"]:
            self._pert_torque_world[mask] = 0
        else:
            tdir = torch.randn((k, 3), device=dev)
            tdir = tdir / tdir.norm(dim=1, keepdim=True).clamp(min=1e-6)
            t_imp = torch.rand((k,), device=dev) * self._max_ang_impulse
            torque = (tdir * t_imp.unsqueeze(1)) / dur_s.unsqueeze(1)
            tscale = float(self._env_opts["pert_torque_max_weight_scale"])
            if tscale > 0.0:
                tmax = w * float(self._env_opts["max_ang_impulse_lever"]) * tscale
                tnorm = torque.norm(dim=1, keepdim=True).clamp(min=1e-9)
                torque = torque * torch.minimum(torch.ones_like(tnorm), tmax / tnorm)
            self._pert_torque_world[mask] = torque.to(self._dtype)
        self._pert_steps_remaining[mask] = steps

    def _process_perturbations(self):
        """Advance perturbation state and (re)apply the current push to the base link for one step."""
        if self._pert_steps_remaining is None:
            return
        n = self._num_envs
        dev = self._device
        rem = self._pert_steps_remaining
        active = rem > 0
        if bool(active.any()):
            rem[active] -= 1
        ended = active & (rem <= 0)
        if bool(ended.any()):
            self._pert_force_world[ended, :] = 0
            self._pert_torque_world[ended, :] = 0
        # trigger new pushes for envs whose current push has ended
        self._pert_det_counter += 1
        can_trigger = rem <= 0
        if self._env_opts["det_pert_rate"]:
            trig = (self._pert_det_counter >= self._pert_det_steps) & can_trigger
        else:
            prob = self.physics_dt() / max(1e-6, float(self._env_opts["pert_wrenches_rate"]))
            trig = (torch.rand((n,), device=dev) < prob) & can_trigger
        if bool(trig.any()):
            self._sample_perturbations(trig)
            self._pert_det_counter[trig] = 0
        # apply the current push as a one-step impulse (adapter applies it during step())
        active = self._pert_steps_remaining > 0
        ft = torch.zeros((n, 1, 6), dtype=self._dtype, device=dev)
        ft[:, 0, 0:3] = self._pert_force_world
        ft[:, 0, 3:6] = self._pert_torque_world
        durations = torch.full((n, 1), self.physics_dt(), dtype=self._dtype, device=dev)
        delays = torch.zeros((n, 1), dtype=self._dtype, device=dev)
        self._genesis_adapter.set_link_impulses([self._base_link_id], ft, durations, delays, vec_mask=active)

    # ------------------------------------------------------- control / step

    @override
    def _set_startup_jnt_imp_gains(self,
            robot_name:str, 
            env_indxs: torch.Tensor = None):
        super()._set_startup_jnt_imp_gains(robot_name=robot_name,env_indxs=env_indxs)
        # apply the impedance command immediately so the robot is held from startup
        self._genesis_adapter.set_current_joint_impedance_command(
            self._jnt_imp_controllers[robot_name].get_pvesd())


    def _step_world(self):
        self._apply_render_env_commands()
        if self._env_opts["use_random_pertub"]:
            self._process_perturbations()
        time_elapsed = self._genesis_adapter.step()
        self._step_counter += 1
        if not (abs(time_elapsed - self.physics_dt()) < 1e-6):
            Journal.log(self.__class__.__name__,
                "_step_world",
                f"simulation stepped of {time_elapsed} [s], expected {self.physics_dt()} [s]",
                LogType.WARN,
                throw_when_excep=False)
        if self._env_opts["enable_height_vis"]:
            self._update_height_vis()

    def _update_height_vis(self):
        """Draw the sampled height-grid points (rendered env) as genesis debug spheres. No-op when
        the sensor is off or there is no live visualizer (headless without cameras)."""
        robot_name = self._robot_names[0]
        if robot_name not in self._height_sensors:
            return
        period = max(1, int(self._env_opts["height_vis_update_period"]))
        self._height_vis_step += 1
        if (self._height_vis_step - 1) % period != 0:
            return
        env = int(self._render_env_idx)
        p = self._root_p[robot_name][env:env + 1]
        q = self._root_q[robot_name][env:env + 1]
        pts = self._height_sensors[robot_name].sample_world_points(p, q)[0]  # (P, 3) world
        node = self._genesis_adapter.draw_debug_spheres(
            poss=pts, radius=float(self._env_opts["height_vis_radius"]),
            color=tuple(self._env_opts["height_vis_color"]))
        if node is not None:
            # clear the previous frame's spheres only once the new ones are up (avoids flicker)
            self._genesis_adapter.clear_debug_object(self._height_vis_node)
            self._height_vis_node = node

    def _generate_jnt_imp_control(self, robot_name: str):
        return GenesisJntImpCntrl(
            genesis_adapter=self._genesis_adapter,
            num_envs=self._num_envs,
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)

    def _apply_cmds_to_jnt_imp_control(self, robot_name: str):
        super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
        self._genesis_adapter.setJointsImpedanceCommand(
            self._jnt_imp_controllers[robot_name].get_pvesd())

    def _get_contact_f(self, robot_name: str, contact_link: str, env_indxs: torch.Tensor) -> torch.Tensor:
        # net contact force (world frame, N) on the given link, read from the genesis entity.
        # Returns (n_envs, 3); None if the contact frame is not a rigid link of the robot (then the
        # base class leaves the contact wrench unset, like the XMJ interface).
        idx = self._contact_link_entity_index(robot_name, contact_link)
        if idx is None:
            return None
        entity = self._genesis_adapter._entities[robot_name]
        forces = entity.get_links_net_contact_force()  # (n_envs, n_links, 3)
        f = forces[:, idx, :].to(self._dtype)
        if env_indxs is not None:
            f = f[env_indxs, :]
        return f

    def _contact_link_entity_index(self, robot_name: str, contact_link: str):
        if self._contact_link_idx_cache is None:
            entity = self._genesis_adapter._entities[robot_name]
            self._contact_link_idx_cache = {l.name: i for i, l in enumerate(entity.links)}
        return self._contact_link_idx_cache.get(contact_link, None)

    @override
    def _update_contact_state(self, robot_name: str, env_indxs: torch.Tensor = None):
        # base: writes the measured per-contact forces (via _get_contact_f). Then also report the
        # applied root perturbation wrench (world frame) on the RobotState root-wrench buffer, so the
        # external disturbance is visible on shared memory (e.g. for pert-recovery obs).
        super()._update_contact_state(robot_name=robot_name, env_indxs=env_indxs)
        if not self._env_opts["use_random_pertub"] or self._pert_force_world is None:
            return
        root_w = getattr(self.cluster_servers[robot_name].get_state(), "contact_wrenches_root", None)
        if root_w is None:
            return
        f = self._pert_force_world if env_indxs is None else self._pert_force_world[env_indxs]
        t = self._pert_torque_world if env_indxs is None else self._pert_torque_world[env_indxs]
        root_w.set(data=f, data_type="f", contact_name="root", robot_idxs=env_indxs, gpu=self._use_gpu)
        root_w.set(data=t, data_type="t", contact_name="root", robot_idxs=env_indxs, gpu=self._use_gpu)

    # ------------------------------------------------------------- misc info

    def is_running(self):
        return self._isrunning

    def current_tstep(self):
        return self._step_counter

    def world_time(self, robot_name: str):
        return self._genesis_adapter.getEnvTimeFromStartup()

    def physics_dt(self):
        return self._env_opts["physics_dt"]

    def rendering_dt(self):
        return self._env_opts["rendering_dt"]

    def set_physics_dt(self, physics_dt: float):
        raise NotImplementedError()

    def set_rendering_dt(self, rendering_dt: float):
        raise NotImplementedError()

    def _robot_jnt_names(self, robot_name: str):
        return self._robot_jnames

    def _render_sim(self, mode: str = "human"):
        return None

    def _close(self):
        if self._genesis_adapter is not None:
            try:
                self._genesis_adapter.destroy_scenario()
            except Exception:
                pass
        self._isrunning = False
