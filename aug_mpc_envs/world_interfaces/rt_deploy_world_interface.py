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

import torch
import numpy as np

from typing import Dict, List
from typing_extensions import override

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc_envs.utils.math_utils import quat_to_omega
from aug_mpc_envs.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from adarl.adapters.ZmqXbotAdapter import ZmqXbotAdapter, XbotSafetyError
from mpc_hive.utilities.timing import high_resolution_sleep_s
from mpc_hive.utilities.math_utils_torch import world2base_frame3D

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase
from aug_mpc_envs.utils.xbot_runtime_config import XbotRuntimeConfigMixin

class RtDeploymentEnv(XbotRuntimeConfigMixin, AugMPCWorldInterfaceBase):
    """Deployment interface for an already-running XBot2 stack exposed over ZMQ."""

    def __init__(self,
        robot_names: List[str],
        robot_urdf_paths: List[str],
        robot_srdf_paths: List[str],
        jnt_imp_config_paths: List[str],
        n_contacts: List[int],
        cluster_dt: List[float],
        use_remote_stepping: List[bool],
        name: str = "IsaacSimEnv",
        num_envs: int = 1,
        debug = False,
        verbose: bool = False,
        vlevel: VLevel = VLevel.V1,
        n_init_step: int = 0,
        timeout_ms: int = 60000,
        env_opts: Dict = None,
        use_gpu: bool = False,
        dtype: torch.dtype = torch.float32,
        override_low_lev_controller: bool = False):

        if not len(robot_names)==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Multiple robots are not supported!",
            LogType.EXCEP,
            throw_when_excep = True)

        if not num_envs==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Parallel deployment is not supported!",
            LogType.EXCEP,
            throw_when_excep = True)

        if use_gpu:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Remote deployment env should run on CPU!",
            LogType.EXCEP,
            throw_when_excep = True)

        self._xbot_adapter_init_tsteps=n_init_step
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
            n_init_step=0, # adapter will handle init steppingy
            timeout_ms=timeout_ms,
            env_opts=env_opts,
            use_gpu=use_gpu,
            dtype=dtype,
            override_low_lev_controller=False)
        # BaseTask.__init__(self,name=self._name,offset=None)

    def _pre_setup(self):
        self._render = False

    def _parse_env_opts(self):
        rt_opts={}
        rt_opts["use_gpu"]=False
        rt_opts["state_from_xbot"]=True
        rt_opts["device"]="cpu"
        rt_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        rt_opts["use_diff_vels"] = False

        rt_opts["xmj_files_dir"]=None
        rt_opts["xbot_config_path"]=None
        rt_opts["xbot_runtime_config_dir"]=None

        rt_opts["rt_safety_perf_coeff"]=1.0
        rt_opts["is_sim"]=False
        rt_opts["time_source"]=None

        rt_opts["xbot2_filter_prof"]="medium"
        rt_opts["xbot2_comm_protocol"]="tcp"
        rt_opts["xbot2_remote_ip"]="localhost"
        rt_opts["xbot2_tcp_service_port"]=5557
        rt_opts["xbot2_tcp_state_port"]=5559
        rt_opts["xbot2_tcp_cmd_port"]=5558
        rt_opts["xbot2_ipc_state_path"]="/tmp/xbot2_zmq_pub.ipc"
        rt_opts["xbot2_ipc_cmd_path"]="/tmp/xbot2_zmq_cmd.ipc"
        rt_opts["xbot2_ipc_service_path"]="/tmp/xbot2_zmq_rep.ipc"
        rt_opts["xbot2_sense_timeout_s"]=2.0
        rt_opts["xbot2_health_check_timeout_s"]=0.2
        rt_opts["xbot2_health_check_period_s"]=1.0
        rt_opts["xbot2_health_stale_after_s"]=5.0

        rt_opts["base_linkname"]="base_link"

        rt_opts["use_mpc_pos_for_robot"]=True
        rt_opts["use_rel_q_from_startup"]=True

        rt_opts["torque_correction"]=1.0 # correction factor for torques sent to real robot
        # (useful if no torque sensors are available)

        rt_opts["max_imp_torque"]=150.0 # [Nm]

        rt_opts["ramp_to_homing"]=True
        rt_opts["ramp_impedances"]=True
        rt_opts["jnt_imp_ramp_time"]=1.0
        rt_opts["jnt_pos_ramp_time"]=4.0
        rt_opts["jnt_imp_ramp_time_onclose"]=2.0

        rt_opts.update(self._env_opts) # update defaults with provided opts
        if rt_opts["time_source"] is None:
            rt_opts["time_source"]="sim" if rt_opts["is_sim"] else "wall"

        rt_opts["use_gpu_pipeline"]=False
        rt_opts["device"]="cpu"
        rt_opts["sim_device"]="cpu"
        rt_opts["run_cluster_bootstrap"] = True # to avoid initial jumps

        # overwrite env opts in case some sim params were missing
        self._env_opts=rt_opts

        # update device flag based on sim opts
        self._device=rt_opts["device"]
        self._use_gpu=rt_opts["use_gpu"]

    def _init_world(self):

        info = "Initializing deployment environment"
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)

        self._configure_scene()

    @override
    def _setup(self):
        # last thing called before spinning
        setup_ok=super()._setup()

        self._isrunning=True

        return setup_ok

    def _configure_scene(self):

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            urdf_path = self._robot_urdf_paths[robot_name]
            srdf_path = self._robot_srdf_paths[robot_name]
            self._generate_rob_descriptions(robot_name=robot_name,
                                    urdf_path=urdf_path,
                                    srdf_path=srdf_path)
            self._prepare_xbot_runtime_config(robot_name=robot_name)

            self._xbot_adapter=ZmqXbotAdapter(model_name=robot_name,
                stepLength_sec=self._cluster_dt[robot_name],
                is_floating_base=True,
                reference_frame="world",
                torch_device=torch.device(self._device),
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=60.0,
                allow_fallback=True,
                enable_filters=True,
                base_link=self._env_opts["base_linkname"],
                is_simulated=self._env_opts["is_sim"],
                time_source=self._env_opts["time_source"],
                remote_ip=self._env_opts["xbot2_remote_ip"],
                comm_protocol=self._env_opts["xbot2_comm_protocol"],
                tcp_service_port=self._env_opts["xbot2_tcp_service_port"],
                tcp_state_port=self._env_opts["xbot2_tcp_state_port"],
                tcp_cmd_port=self._env_opts["xbot2_tcp_cmd_port"],
                ipc_pub_path=self._env_opts["xbot2_ipc_state_path"],
                ipc_cmd_path=self._env_opts["xbot2_ipc_cmd_path"],
                ipc_service_path=self._env_opts["xbot2_ipc_service_path"],
                sense_timeout_s=self._env_opts["xbot2_sense_timeout_s"],
                health_check_timeout_s=self._env_opts["xbot2_health_check_timeout_s"],
                health_check_period_s=self._env_opts["xbot2_health_check_period_s"],
                health_stale_after_s=self._env_opts["xbot2_health_stale_after_s"])
            # self._xbot_adapter.build_scenario()
            self._xbot_adapter.startup()
            self._xbot_adapter.position_ramp_time=self._env_opts["jnt_pos_ramp_time"] # [s]
            self._xbot_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time"] # [s]

            to_monitor=[]
            self._robot_iface_enabled_jnts=[jname for _, jname in self._xbot_adapter.get_xbot_controlled_joints()]

            for jnt_name in self._robot_iface_enabled_jnts:
                to_monitor.append((self._robot_names[i], jnt_name))

            self._xbot_adapter.set_monitored_joints(to_monitor)
            self._xbot_adapter.set_impedance_controlled_joints(to_monitor)

            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing sim pre-setup...",
                        LogType.STAT,
                        throw_when_excep = True)

            self._reset_sim()
            self._fill_robot_info_from_world()
            # initializes robot state data
            self._init_robots_state()
            # update solver options
            self._print_envs_info() # debug print

            self.scene_setup_completed = True

            # set joint reference filters
            self._xbot_adapter.set_filters(set_enabled=True,
                profile_name=self._env_opts["xbot2_filter_prof"])

        self._last_control_time=0.0
        self._last_jntv_numdiff_time=0.0
        self._last_twist_numdiff_time=0.0

    @override
    def _xrdf_cmds(self, robot_name:str):
        cmds=super()._xrdf_cmds(robot_name=robot_name)
        for i, s in enumerate(cmds):
            if "floating_joint:=" in s:
                cmds[i] = "floating_joint:=true"
        return cmds

    def _render_sim(self, mode="human"):
        pass

    def _cleanup_warn(self, message: str):
        try:
            Journal.log(self.__class__.__name__,
                "_close",
                message,
                LogType.WARN,
                throw_when_excep=False)
        except Exception:
            pass

    def _xbot_cleanup_available(self) -> bool:
        if not hasattr(self, "_xbot_adapter"):
            return False
        if not getattr(self._xbot_adapter, "_started", False):
            return False
        try:
            return not self._xbot_adapter.is_safety_triggered()
        except Exception:
            return False

    def _close(self):
        if not hasattr(self, "_xbot_adapter"):
            return

        adapter_can_command = self._xbot_cleanup_available()
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]

            if adapter_can_command:
                try:
                    self._xbot_adapter.set_filters(set_enabled=True, profile_name="safe")
                except Exception as exc:
                    adapter_can_command = False
                    self._cleanup_warn(f"Skipping XBot2 filter cleanup: {exc}")

            self._xbot_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time_onclose"] # setting slower
            self._env_opts["ramp_to_homing"]=False # skip homing when closing

            if adapter_can_command and robot_name in self._jnt_imp_controllers:
                try:
                    self._reset_jnt_imp_control(robot_name=robot_name) # will set jnt imp gains to initial vals and
                    # pos ref to homing and apply them with the adapter
                except XbotSafetyError as exc:
                    adapter_can_command = False
                    self._cleanup_warn(f"Skipping XBot2 impedance cleanup after safety trigger: {exc}")
                except Exception as exc:
                    adapter_can_command = False
                    self._cleanup_warn(f"Skipping XBot2 impedance cleanup: {exc}")

            self._isrunning=False

    @override
    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):
        super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
        jnt_imp_cmds=self._jnt_imp_controllers[self._robot_names[0]].get_pvesd()
        jnt_imp_cmds[:, 2]=self._env_opts["torque_correction"]*jnt_imp_cmds[:, 2] # scaling efforts for real robot
        jnt_imp_cmds[:, 2]=torch.clamp(jnt_imp_cmds[:, 2], min=-self._env_opts["max_imp_torque"], max=self._env_opts["max_imp_torque"])
        self._xbot_adapter.setJointsImpedanceCommand(jnt_imp_cmds)
        elapsed_since_last_cmd=self.world_time(robot_name=robot_name)-\
            self._last_control_time
        walltime_to_sleep=self.physics_dt()-elapsed_since_last_cmd
        if walltime_to_sleep<-1e-2:
            Journal.log(self.__class__.__name__,
                "_apply_cmds_to_jnt_imp_control",
                f"RT performance violated of {walltime_to_sleep} s.",
                LogType.WARN,
            throw_when_excep = True)
            walltime_to_sleep=0 # do not sleep

        high_resolution_sleep_s(self._env_opts["rt_safety_perf_coeff"]*walltime_to_sleep) # make sure cmds are applied to
        # jnt imp controller at a constant rate
        self._xbot_adapter.apply_joint_impedances(jnt_imp_cmds) # write to robot (there could be
        # communication delays)
        self._last_control_time=self.world_time(robot_name=robot_name)
        self._p_ref_reset[robot_name][:, :]= self._jnt_imp_controllers[robot_name].pos_ref() # store last sent pos ref

    @override
    def _jnt_imp_reset_overrride(self,
        robot_name: str):

        # before env applies jnt imp reset to robot, we ensure no discountinuities by always ramping impedances
        # with the current position as target and no velocity and effort references

        if self._env_opts["ramp_to_homing"]: # ramp position references to target values smoothly
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=self._homing,
                robot_indxs = None)
            super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name) # need to be called here to propoerly apply pvesd tensor
            pvesd = self._jnt_imp_controllers[robot_name].get_pvesd()
            self._xbot_adapter.apply_joint_ref_with_ramp(pvesd, tolerance=0.1)
        else: # set p ref to current value to avoid jumps (pref or meas. p?)
            reset_ref=self._p_ref_reset[robot_name]
            # reset_ref=self._jnts_q[robot_name]
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=reset_ref,
                robot_indxs = None)
            super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)

        if self._env_opts["ramp_impedances"]: # ramp impedances
            pvesd = self._jnt_imp_controllers[robot_name].get_pvesd()
            self._xbot_adapter.apply_joint_impedances_with_ramp(pvesd, tolerance=0.1) # ramps impeances

    def _generate_jnt_imp_control(self, robot_name: str):

        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=self._xbot_adapter,
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)

        return jnt_imp_controller

    @override
    def _pre_step(self):
        try:
            return super()._pre_step()
        except XbotSafetyError as exc:
            Journal.log(self.__class__.__name__,
                "_pre_step",
                str(exc),
                LogType.WARN,
                throw_when_excep=False)
            self._isrunning = False
            return False

    def _step_world(self): # real world steps by itself (hopefully)
        pass

    def _reset_sim(self):
        self._xbot_adapter.resetWorld()
        self._last_control_time=0.0
        self._last_jntv_numdiff_time=0.0
        self._last_twist_numdiff_time=0.0

    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):

        self._reset_sim()

    def _read_root_state_from_robot(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            ):

        if (not self._env_opts["state_from_xbot"]):
            self._get_root_state(numerical_diff=False,
                    env_indxs=env_indxs,
                    robot_name=robot_name)
        else:
            self._get_root_state_xbot(numerical_diff=False,
                    env_indxs=env_indxs,
                    robot_name=robot_name)

    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):

        if (not self._env_opts["state_from_xbot"]):
            self._get_robots_jnt_state(
                numerical_diff=self._env_opts["use_diff_vels"],
                env_indxs=env_indxs,
                robot_name=robot_name)
        else:
            self._get_robots_jnt_state_xbot(
                numerical_diff=self._env_opts["use_diff_vels"],
                env_indxs=env_indxs,
                robot_name=robot_name)

    def _get_root_state(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        base_loc: bool = True):

        raise NotImplementedError()

    def _get_root_state_xbot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        base_loc: bool = True):

        # update IMU and get base link state (assumed to return torch tensors)
        self._xbot_adapter.read_imu_data()
        frame_name, q, omega, linacc = self._xbot_adapter.get_base_link_state()

        # position handling (same as before)
        if self._env_opts["use_mpc_pos_for_robot"]:
            actions = self.cluster_servers[robot_name].get_actions()
            rhc_p = actions.root_state.get(data_type="p", gpu=self._use_gpu)
            self._root_p[robot_name][:, :] = rhc_p
        else:
            raise NotImplementedError("Only root position from MPC is implemented. No odometry available yet.")

        # store raw IMU quaternion (ensure dtype/device). Startup yaw-rel projection
        # for MPC state publishing is now handled in the base world interface.
        self._root_q[robot_name][:, :] = torch.from_numpy(q).reshape(self._num_envs, -1).to(self._dtype)

        # dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate
        dt=self.world_time(robot_name=robot_name)-self._last_twist_numdiff_time

        if not numerical_diff:
            # we get velocities from the simulation. This is not good since
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            # self._root_v[robot_name][:, :] = torch.from_numpy(self._xbot_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)
            self._root_omega[robot_name][:, :] = torch.from_numpy(omega).reshape(self._num_envs, -1).to(self._dtype)

            self._root_a[robot_name][env_indxs, :] = torch.from_numpy(linacc).reshape(self._num_envs, -1).to(self._dtype)

            # self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
            #                                 self._root_omega_prev[robot_name][env_indxs, :]) / dt

        else:
            # differentiate numerically

            Journal.log(self.__class__.__name__,
                "_get_root_state_xbot",
                "Reading root state with differentiation not supported yet!!",
                LogType.EXCEP,
                throw_when_excep = True)

            # self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
            #                                 self._root_p_prev[robot_name]) / dt
            # self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name],
            #                                             self._root_q[robot_name],
            #                                             dt)
            # self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
            #                                     self._root_v_prev[robot_name][env_indxs, :]) / dt

            # self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
            #                                 self._root_omega_prev[robot_name][env_indxs, :]) / dt

        self._last_twist_numdiff_time=self.world_time(robot_name=robot_name)

        # update "previous" data for numerical differentiation
        # self._root_p_prev[robot_name][:, :] = self._root_p[robot_name]
        self._root_q_prev[robot_name][:, :] = self._root_q[robot_name]
        # self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :]
        self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]

        world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
                v_out=self._gravity_normalized_base_loc[robot_name])

        #  no need to rotate robot twist in base local
        self._root_omega_base_loc[robot_name][:, :]=self._root_omega[robot_name]
        self._root_a_base_loc[robot_name][:, :]=self._root_a[robot_name]
        # self._root_alpha_base_loc[robot_name][:, :]=self._root_alpha[robot_name]

    def _get_robots_jnt_state(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):

        raise NotImplementedError()

    def _get_robots_jnt_state_xbot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):

        jnt_state_from_xbot=self._xbot_adapter.getJointsState().T # [3(p, v, e)x n_jnts]

        self._jnts_q[robot_name][:, :] = jnt_state_from_xbot[0,:]

        dt= None
        if numerical_diff:
            # dt= self.physics_dt() if self._override_low_lev_controller else self._cluster_dt[robot_name]
            dt=self.world_time(robot_name=robot_name)-self._last_jntv_numdiff_time

        if dt is None:
            self._jnts_v[robot_name][:, :] = jnt_state_from_xbot[1,:]
        else:
            self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt

            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

            self._last_jntv_numdiff_time=self.world_time(robot_name=robot_name)

        self._jnts_eff[robot_name][env_indxs, :] = jnt_state_from_xbot[2,:]

    def _set_jnts_to_homing(self, robot_name: str):
    	pass

    def _set_root_to_defconfig(self, robot_name: str):
        msg="Cannot teleport robot in real world! Please ensure the robot is in the desired reset configuration"
        Journal.log(self.__class__.__name__,
            "_set_root_to_defconfig",
            msg,
            LogType.WARN,
            throw_when_excep = True)

    def _get_solver_info(self):
        raise NotImplementedError()

    def _print_envs_info(self):
        pass

    def _fill_robot_info_from_world(self):
        pass

    def _set_initial_camera_params(self,
                                camera_position=[10, 10, 3],
                                camera_target=[0, 0, 0]):
        raise NotImplementedError()

    def _init_contact_sensors(self):
        raise NotImplementedError()

    def _get_contact_f(self,
        robot_name: str,
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        return None

    def _init_robots_state(self):

        self._p_ref_reset={}
        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]

            # root p (measured, previous, default)
            self._root_p[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_p_prev[robot_name] = self._root_p[robot_name].clone()
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = self._root_p[robot_name].clone()
            # root q (measured, previous, default)
            self._root_q[robot_name] = torch.zeros((self._num_envs, 4), dtype=self._dtype)
            self._root_q[robot_name][:, 0]=1
            self._root_q_prev[robot_name] = self._root_q[robot_name].clone()
            self._root_q_default[robot_name] = self._root_q[robot_name].clone()

            # jnt q (measured, previous, default)
            n_jnts=len(self._robot_iface_enabled_jnts)
            self._jnts_q[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_q_prev[robot_name] = self._jnts_q[robot_name].clone()
            self._jnts_q_default[robot_name] = self._jnts_q[robot_name].clone()

            # root v (measured, default)
            self._root_v[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_v_base_loc[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_prev[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_default[robot_name] = self._root_v[robot_name].clone()

            # root omega (measured, default)
            self._root_omega[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype)
            self._root_omega_prev[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_base_loc[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_default[robot_name] = self._root_omega[robot_name].clone()

            # root a (measured,)
            self._root_a[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_a_base_loc[robot_name] = torch.full_like(self._root_a[robot_name], fill_value=0.0)
            self._root_alpha[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_alpha_base_loc[robot_name] = torch.full_like(self._root_alpha[robot_name], fill_value=0.0)

            # joints v (measured, default)
            self._jnts_v[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_v_default[robot_name] = self._jnts_v[robot_name].clone()

            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.zeros((self._num_envs, n_jnts), dtype=self._dtype)
            self._jnts_eff_default[robot_name] = self._jnts_eff[robot_name].clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3),
                                device=self._device) # reference position offses

            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4),
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            self._p_ref_reset[robot_name] = self._jnts_q[robot_name].clone() # last p ref for set to jnt imp control

            # self._update_root_offsets(robot_name)

    def current_tstep(self):
        return int(self.world_time(self._robot_names[0])/self.physics_dt())

    def world_time(self, robot_name: str) -> float: # get relative time from last reset
        return self._xbot_adapter.getEnvTimeFromReset()

    def physics_dt(self):
        robot_name = self._robot_names[0]
        return self._cluster_dt[robot_name]

    def rendering_dt(self):
        robot_name = self._robot_names[0]
        return self._cluster_dt[robot_name]

    def set_physics_dt(self, physics_dt:float):
        raise NotImplementedError()

    def set_rendering_dt(self, rendering_dt:float):
        raise NotImplementedError()

    def _robot_jnt_names(self, robot_name: str):
        return self._robot_iface_enabled_jnts

    def is_running(self):
        running=self._xbot_adapter.is_xbot_control_running()
        if not running:
            if self._xbot_adapter.is_safety_triggered():
                msg = "XBot2 safety is triggered; stopping RT deployment interface"
            else:
                msg = "XBot2/ZMQ control plugin is not running"
            Journal.log(self.__class__.__name__,
            "_is_running",
            msg,
            LogType.EXCEP,
            throw_when_excep = False)

        return running and self._isrunning
