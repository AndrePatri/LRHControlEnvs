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

import numpy as np
import torch

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase

from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter
from adarl.adapters.BaseSimulationAdapter import ModelSpawnDef
from adarl.utils.utils import build_pose

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

        # create the adapter (sim + impedance). sim_step_dt == step_length_sec so a single
        # adapter.step() advances exactly one physics step (physics_dt), like XMJ.
        self._genesis_adapter = GenesisJointImpedanceAdapter(
            vec_size=self._num_envs,
            output_th_device=torch.device(self._device),
            sim_step_dt=self._env_opts["physics_dt"],
            step_length_sec=self._env_opts["physics_dt"],
            enable_rendering=False,
            add_ground=self._env_opts["add_ground"],
            show_gui=(not self._env_opts["headless"]),
            genesis_logging_level=self._env_opts["genesis_logging_level"])

        spawn_pose = build_pose(0.0, 0.0, float(self._env_opts["spawning_height"]), 0.0, 0.0, 0.0, 1.0)
        model = ModelSpawnDef(
            name=robot_name,
            definition_string=urdf_str,
            format="urdf",
            pose=spawn_pose,
            kwargs={"genesis_fixed": bool(self._env_opts["genesis_fixed_base"]),
                    "genesis_merge_fixed_links": bool(self._env_opts["genesis_merge_fixed_links"])})
        # genesis scenes are static after build: all models must be passed to build_scenario
        self._genesis_adapter.build_scenario(models=[model])

        # detected joints come back as (model_name, joint_name); keep only this robot's
        detected = self._genesis_adapter.get_detected_joints()
        self._robot_joints = [tuple(j) for j in detected if j[0] == robot_name]
        self._robot_jnames = [j[1] for j in self._robot_joints]
        self._base_link_id = (robot_name, base_link)

        self._genesis_adapter.set_monitored_joints(self._robot_joints)
        self._genesis_adapter.set_monitored_links([self._base_link_id])
        self._genesis_adapter.set_impedance_controlled_joints(self._robot_joints)

        self._fill_robot_info_from_world()
        self._init_robots_state()
        self._isrunning = True

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
        self._root_p[robot_name][:, :] = p
        self._root_q[robot_name][:, :] = q
        if not self._env_opts["use_diff_vels"]:
            self._root_v[robot_name][:, :] = v
            self._root_omega[robot_name][:, :] = omega
        else:
            dt = self._cluster_dt[robot_name]
            self._root_v[robot_name][:, :] = (p - self._root_p_prev[robot_name]) / dt
            # NOTE: angular numdiff from quaternions omitted for the minimal interface
            self._root_omega[robot_name][:, :] = omega
        self._root_p_prev[robot_name][:, :] = p
        self._root_q_prev[robot_name][:, :] = q

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

    def _apply_state_to_sim(self, robot_name: str):
        """Push the current default joint + base state into the genesis sim."""
        # joints: (E, n, 3) -> [pos, vel, eff]
        jq = self._jnts_q_default[robot_name]
        pve = torch.zeros((self._num_envs, len(self._robot_joints), 3), device=jq.device, dtype=jq.dtype)
        pve[:, :, 0] = jq
        self._genesis_adapter.setJointsStateDirect(self._robot_joints, pve)
        # base link: pose (p[3] + quat_xyzw[4]) + vel (lin[3] + ang[3]) -> (E, 1, 13)
        p = self._root_p_default[robot_name]
        q_wxyz = self._root_q_default[robot_name]
        q_xyzw = q_wxyz[:, [1, 2, 3, 0]]
        link_state = torch.zeros((self._num_envs, 1, 13), device=p.device, dtype=p.dtype)
        link_state[:, 0, 0:3] = p
        link_state[:, 0, 3:7] = q_xyzw
        self._genesis_adapter.setLinksStateDirect([self._base_link_id], link_state)

    def _set_jnts_to_homing(self, robot_name: str):
        self._apply_state_to_sim(robot_name)

    def _set_root_to_defconfig(self, robot_name: str):
        self._apply_state_to_sim(robot_name)

    def _reset_sim(self):
        self._genesis_adapter.resetWorld()
        # resetWorld restores the build-time state, so re-apply the homing/default config
        for robot_name in self._robot_names:
            self._apply_state_to_sim(robot_name)

    def _reset_state(self, robot_name: str, env_indxs: torch.Tensor = None, randomize: bool = False):
        self._reset_sim()

    # ------------------------------------------------------- control / step

    def _step_world(self):
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
