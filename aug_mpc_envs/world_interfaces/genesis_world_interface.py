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
# Minimal world interface for the Genesis simulator (https://github.com/Genesis-Embodied-AI/genesis-world).
# It wraps adarl's vectorized GenesisJointImpedanceAdapter, mirroring the XMJ interface
# (which also wraps an adarl adapter). Scope is intentionally minimal: a single robot on
# flat ground, proprioceptive joint state + base-link state, no terrain/perturbation/
# camera/heightmap. Enough to run e.g. Talos and verify the sim is healthy over shared mem.
from typing import Dict, List
from typing_extensions import override

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
        g_opts["genesis_vis_mode"] = "collision"
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

        g_opts.update(self._env_opts)  # override defaults with provided opts

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

        # create the adapter (sim + impedance). sim_step_dt == step_length_sec so a single
        # adapter.step() advances exactly one physics step (physics_dt), like XMJ.
        self._genesis_adapter = GenesisJointImpedanceAdapter(
            vec_size=self._num_envs,
            output_th_device=torch.device(self._device),
            sim_step_dt=self._env_opts["physics_dt"],
            step_length_sec=self._env_opts["physics_dt"],
            enable_rendering=bool(self._env_opts["genesis_enable_camera_rendering"]),
            render_envs_idx=render_envs_idx,
            add_ground=self._env_opts["add_ground"],
            show_gui=(not self._env_opts["headless"]),
            max_joint_impedance_ctrl_torques=max_torques,
            rigid_options_override=self._env_opts["genesis_rigid_options"],
            vis_options_override={"contact_force_scale": float(self._env_opts["genesis_contact_force_scale"])},
            reference_filter_mode="none",  # run the impedance refs unfiltered by default
            genesis_logging_level=self._env_opts["genesis_logging_level"],
            use_batch_renderer=bool(self._env_opts["genesis_use_batch_renderer"]))

        spawn_pose = build_pose(0.0, 0.0, float(self._env_opts["spawning_height"]), 0.0, 0.0, 0.0, 1.0)
        model = ModelSpawnDef(
            name=robot_name,
            definition_string=urdf_str,
            format="urdf",
            pose=spawn_pose,
            kwargs={"genesis_fixed": bool(self._env_opts["genesis_fixed_base"]),
                    "genesis_merge_fixed_links": bool(self._env_opts["genesis_merge_fixed_links"]),
                    "genesis_vis_mode": self._env_opts["genesis_vis_mode"],
                    "genesis_visualize_contact": bool(self._env_opts["genesis_visualize_contact"])})
        # genesis scenes are static after build: all models must be passed to build_scenario
        self._genesis_adapter.build_scenario(models=[model])

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

        self._reset_sim()
        
        self._fill_robot_info_from_world()

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

    def _set_jnts_to_homing(self, robot_name: str):
        # joints: (E, n, 3) -> [pos, vel, eff]
        jq = self._jnts_q_default[robot_name]
        pve = torch.zeros((self._num_envs, len(self._robot_joints), 3), device=jq.device, dtype=jq.dtype)
        pve[:, :, 0] = jq
        self._genesis_adapter.setJointsStateDirect(self._robot_joints, pve)

    def _set_root_to_defconfig(self, robot_name: str):
        # base link: pose (p[3] + quat_xyzw[4]) + vel (lin[3] + ang[3]) -> (E, 1, 13)
        p = self._root_p_default[robot_name]
        q_wxyz = self._root_q_default[robot_name]
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        link_state = torch.zeros((self._num_envs, 1, 13), device=p.device, dtype=p.dtype)
        link_state[:, 0, 0:3] = p
        link_state[:, 0, 3:7] = q_xyzw
        self._genesis_adapter.setLinksStateDirect([self._base_link_id], link_state)

    def _reset_sim(self):
        self._genesis_adapter.resetWorld()
        # resetWorld() restores the build-time spawn (URDF-default / zero joints), which would
        # undo the homing the base sets right before calling _reset_sim. Re-apply the default
        # joint + base config so the robot actually starts at the homing pose.
        for robot_name in self._robot_names:
            self._set_jnts_to_homing(robot_name)
            self._set_root_to_defconfig(robot_name)

    def _reset_state(self, robot_name: str, env_indxs: torch.Tensor = None, randomize: bool = False):
        self._reset_sim()

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
        time_elapsed = self._genesis_adapter.step()
        self._step_counter += 1
        if not (abs(time_elapsed - self.physics_dt()) < 1e-6):
            Journal.log(self.__class__.__name__,
                "_step_world",
                f"simulation stepped of {time_elapsed} [s], expected {self.physics_dt()} [s]",
                LogType.WARN,
                throw_when_excep=False)

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
        # minimal interface: no contact forces exposed yet (matches the XMJ interface)
        return None

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
