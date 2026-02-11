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
import math

from typing import Dict, List
from typing_extensions import override

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc_envs.utils.math_utils import quat_to_omega
from aug_mpc_envs.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from adarl_ros.adapters.XbotMjAdapter import RosXbotAdapter
from mpc_hive.utilities.math_utils_torch import world2base_frame3D

from mpc_hive.utilities.math_utils_torch import quaternion_multiply

import rospy

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase

class RtDeploymentEnv(AugMPCWorldInterfaceBase):

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

        self._ros_xbot_adapter_init_tsteps=n_init_step
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
        xmj_opts={}
        xmj_opts["use_gpu"]=False
        xmj_opts["state_from_xbot"]=True
        xmj_opts["device"]="cpu"
        xmj_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        xmj_opts["use_diff_vels"] = False

        xmj_opts["xmj_files_dir"]=None

        xmj_opts["rt_safety_perf_coeff"]=1.0
        xmj_opts["is_sim"]=False

        xmj_opts["xbot2_filter_prof"]="medium"
        
        xmj_opts["base_linkname"]="base_link"

        xmj_opts["use_mpc_pos_for_robot"]=True
        xmj_opts["use_rel_q_from_startup"]=True

        xmj_opts["torque_correction"]=1.0 # correction factor for torques sent to real robot
        # (useful if no torque sensors are available)

        xmj_opts["max_imp_torque"]=150.0 # [Nm]

        xmj_opts["ramp_to_homing"]=True
        xmj_opts["xbot_homing_on_close"]=False
        xmj_opts["ramp_impedances"]=True
        xmj_opts["jnt_imp_ramp_time"]=1.0
        xmj_opts["jnt_pos_ramp_time"]=4.0
        xmj_opts["jnt_imp_ramp_time_onclose"]=2.0

        xmj_opts.update(self._env_opts) # update defaults with provided opts
        
        xmj_opts["use_gpu_pipeline"]=False
        xmj_opts["device"]="cpu"
        xmj_opts["sim_device"]="cpu"
        # overwrite env opts in case some sim params were missing
        self._env_opts=xmj_opts

        # update device flag based on sim opts
        self._device=xmj_opts["device"]
        self._use_gpu=xmj_opts["use_gpu"]

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

        # for safety: get dtype/device from existing storage
        # assume self._root_q_offset[robot_name] exists and is a torch tensor
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            if self._root_q_offset[robot_name] is not None:
                # get rhc quaternion from cluster (assumed torch)
                actions = self.cluster_servers[robot_name].get_actions()
                rhc_q = actions.root_state.get(data_type="q", gpu=self._use_gpu)  # expecting torch tensor
                # ensure shape: (num_envs,4) or (1,4)
                if rhc_q.dim() == 1:
                    rhc_q = rhc_q.unsqueeze(0)
                rhc_q = rhc_q.to(self._dtype)

                # get robot (sim) quaternion stored in self._root_q (torch)
                robot_q = self._root_q[robot_name][:, :].to(self._dtype)

                # normalize both

                # extract yaw-only quaternions (vectorized)
                yaw_rhc = self.quat_to_yaw(rhc_q)                # (num_envs,) or (1,)
                yaw_robot = self.quat_to_yaw(robot_q)            # (num_envs,)

                # build yaw-only quaternions
                device = self._root_q_offset[robot_name].device
                rhc_yaw_q = self.yaw_quat(yaw_rhc)
                robot_yaw_q = self.yaw_quat(yaw_robot)

                # offset = rhc_yaw^{-1} * robot_yaw  so that rhc_yaw * offset = robot_yaw
                offset = quaternion_multiply(self._quat_inverse(rhc_yaw_q), robot_yaw_q)  # shape: (num_envs,4) or (1,4)

                # store offset into tensors; match layout (repeat if needed)
                # if storage expects one row per env, ensure shape matches
                # self._root_q_offset[robot_name] assumed shape (num_envs,4)
                storage_shape = self._root_q_offset[robot_name].shape
                if offset.shape[0] == 1 and storage_shape[0] > 1:
                    offset_to_store = offset.repeat(storage_shape[0], 1)
                else:
                    offset_to_store = offset.reshape(storage_shape)

                self._root_q_offset[robot_name][:, :] = offset_to_store.to(self._dtype).to(device)

                # store inverse of offset for runtime use
                offsetm1=self._quat_inverse(offset_to_store)
                self._root_q_offsetm1[robot_name][:, :] = offsetm1.to(self._dtype).to(device)

        self._q_offset_acquired = True

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

            self._ros_xbot_adapter=RosXbotAdapter(model_name=robot_name,
                stepLength_sec=self._cluster_dt[robot_name],
                forced_ros_master_uri= None,
                blocking_observation=False,
                is_floating_base=True,
                reference_frame="world",
                torch_device=torch.device(self._device),
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=60.0,
                allow_fallback=True,
                enable_filters=True,
                base_link=self._env_opts["base_linkname"],
                is_simulated=self._env_opts["is_sim"])
            # self._ros_xbot_adapter.build_scenario()
            self._ros_xbot_adapter.startup()
            self._ros_xbot_adapter.position_ramp_time=self._env_opts["jnt_pos_ramp_time"] # [s]
            self._ros_xbot_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time"] # [s]
        
            to_monitor=[]
            self._robot_iface_enabled_jnts=self._ros_xbot_adapter.get_robot_interface().getEnabledJointNames()
            
            for jnt in range(len(self._robot_iface_enabled_jnts)):
                to_monitor.append((self._robot_names[i],self._robot_iface_enabled_jnts[jnt]))

            self._ros_xbot_adapter.set_monitored_joints(to_monitor)
            self._ros_xbot_adapter.set_impedance_controlled_joints(to_monitor)

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
            self._ros_xbot_adapter.set_filters(set_enabled=True, 
                profile_name=self._env_opts["xbot2_filter_prof"])
            
        # self._rospy_startime=rospy.get_time()
        self._last_control_time=0.0
        self._last_jntv_numdiff_time=0.0
        self._last_twist_numdiff_time=0.0
            
        self._q_offset_acquired=False

    @override
    def _xrdf_cmds(self, robot_name:str):
        cmds=super()._xrdf_cmds(robot_name=robot_name)
        for i, s in enumerate(cmds):
            if "floating_joint:=" in s:
                cmds[i] = "floating_joint:=true" 
        return cmds

    def _render_sim(self, mode="human"):
        pass

    def _close(self):
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            
            # set filters to safe
            self._ros_xbot_adapter.set_filters(set_enabled=True, 
                profile_name="safe")
            
            # resets jnt imp gain to the startups with a ramp
            self._ros_xbot_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time_onclose"] # setting slower
            self._env_opts["ramp_to_homing"]=False # skip homing when closing
            
            # read last pos ref from jnt imp control before reset
            
            # ramp since impedances will generally be ramped up when cleaning up
            self._reset_jnt_imp_control(robot_name=robot_name) # will set jnt imp gains to initial vals and 
            # pos ref to homing and apply them with the adapter

            if self._env_opts["xbot_homing_on_close"]:
                self._ros_xbot_adapter.trigger_xbot_homing() # perform a final
            # homing to reset the robot to its default xbot state

            self._isrunning=False

    @override
    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):
        super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
        jnt_imp_cmds=self._jnt_imp_controllers[self._robot_names[0]].get_pvesd()
        jnt_imp_cmds[:, 2]=self._env_opts["torque_correction"]*jnt_imp_cmds[:, 2] # scaling efforts for real robot
        jnt_imp_cmds[:, 2]=torch.clamp(jnt_imp_cmds[:, 2], min=-self._env_opts["max_imp_torque"], max=self._env_opts["max_imp_torque"])
        self._ros_xbot_adapter.setJointsImpedanceCommand(jnt_imp_cmds)
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
        
        # while self.world_time(robot_name=robot_name)-self._last_control_time < walltime_to_sleep:
        #     ns=1000
        #     PerfSleep.thread_sleep(ns)
        rospy.sleep(self._env_opts["rt_safety_perf_coeff"]*walltime_to_sleep) # make sure cmds are applied to
        # jnt imp controller at a constant rate (rospy will use sim time if enabled, otherwise walltime)
        self._ros_xbot_adapter.apply_joint_impedances(jnt_imp_cmds) # write to robot (there could be
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
            self._ros_xbot_adapter.apply_joint_ref_with_ramp(pvesd, tolerance=0.1)
        else: # set p ref to current value to avoid jumps (pref or meas. p?)
            reset_ref=self._p_ref_reset[robot_name]
            # reset_ref=self._jnts_q[robot_name]
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=reset_ref,
                robot_indxs = None)
            super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
            
        if self._env_opts["ramp_impedances"]: # ramp impedances
            pvesd = self._jnt_imp_controllers[robot_name].get_pvesd()
            self._ros_xbot_adapter.apply_joint_impedances_with_ramp(pvesd, tolerance=0.1) # ramps impeances

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=self._ros_xbot_adapter,
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)
        
        return jnt_imp_controller

    def _step_world(self): # real world steps by itself (hopefully)
        pass
    
    def _reset_sim(self):
        self._ros_xbot_adapter.resetWorld()
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
            self._get_root_state(numerical_diff=self._env_opts["use_diff_vels"],
                    env_indxs=env_indxs,
                    robot_name=robot_name)
        else:
            self._get_root_state_xbot(numerical_diff=self._env_opts["use_diff_vels"],
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
        self._ros_xbot_adapter.read_imu_data()
        frame_name, q, omega, linacc = self._ros_xbot_adapter.get_base_link_state()

        # position handling (same as before)
        if self._env_opts["use_mpc_pos_for_robot"]:
            actions = self.cluster_servers[robot_name].get_actions()
            rhc_p = actions.root_state.get(data_type="p", gpu=self._use_gpu)
            self._root_p[robot_name][:, :] = rhc_p
        else:
            raise NotImplementedError("Only root position from MPC is implemented. No odometry available yet.")

        if self._root_q_offset[robot_name] is not None and self._q_offset_acquired:

            # extract yaw-only part of incoming q
            yaw_q = self.yaw_quat(self.quat_to_yaw(q))

            # pitch-roll part: q_pr = q_yaw^{-1} * q_full  (so q_full = q_yaw * q_pr)
            yaw_q_inv=  self._quat_inverse(yaw_q)
            q_pr = quaternion_multiply(yaw_q_inv.flatten(), q.flatten())

            # adjusted yaw = offsetm1 * q_yaw  (offsetm1 maps from rhc frame to sim frame; we apply its inverse stored earlier)

            adjusted_yaw = quaternion_multiply(self._root_q_offsetm1[robot_name].flatten(), yaw_q.flatten())

            # new quaternion: adjusted_yaw * q_pr  (applies yaw offset only, keeps pitch+roll from IMU)
            self._root_q[robot_name][:, :] = quaternion_multiply(adjusted_yaw, q_pr)

        else:
            # no offset acquired: store raw IMU quaternion (ensure dtype/device)
            self._root_q[robot_name][:, :] = torch.from_numpy(q).reshape(self._num_envs, -1).to(self._dtype)

        # dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate
        dt=self.world_time(robot_name=robot_name)-self._last_twist_numdiff_time

        if not numerical_diff:
            # we get velocities from the simulation. This is not good since 
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            # self._root_v[robot_name][:, :] = torch.from_numpy(self._ros_xbot_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)   
            self._root_omega[robot_name][:, :] = torch.from_numpy(omega).reshape(self._num_envs, -1).to(self._dtype)        
            
            self._root_a[robot_name][env_indxs, :] = torch.from_numpy(linacc).reshape(self._num_envs, -1).to(self._dtype)  

            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
            
        else:
            # differentiate numerically
            # self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
            #                                 self._root_p_prev[robot_name]) / dt 
            self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name], 
                                                        self._root_q[robot_name], 
                                                        dt)
            
            # self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
            #                                     self._root_v_prev[robot_name][env_indxs, :]) / dt 
        
            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
        
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
        self._root_alpha_base_loc[robot_name][:, :]=self._root_alpha[robot_name]

    def _get_robots_jnt_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):
        
        raise NotImplementedError()

    def _get_robots_jnt_state_xbot(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):

        jnt_state_from_xbot=self._ros_xbot_adapter.getJointsState().T # [3(p, v, e)x n_jnts]

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
        # self._ros_xbot_adapter.trigger_xbot_homing() # blocking, moves the robot using plugins
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
        
        self._root_q_offset={}
        self._root_q_offsetm1={}
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
            self._root_q_offset[robot_name]=None
            if  self._env_opts["use_rel_q_from_startup"]:
                self._root_q_offset[robot_name]=self._root_q[robot_name].clone()
                self._root_q_offsetm1[robot_name]=self._root_q[robot_name].clone()

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
        return self._ros_xbot_adapter.xmj_env().step_counter
    
    def world_time(self, robot_name: str) -> float: # get relative time from last reset
        return self._ros_xbot_adapter.getEnvTimeFromReset()
        # return rospy.get_time()-self._rospy_startime

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
        running=self._ros_xbot_adapter.is_ros_control_running()
        if not running:
            Journal.log(self.__class__.__name__,
            "_is_running",
            "ros_control is not running",
            LogType.EXCEP,
            throw_when_excep = False)
        
        return running and self._isrunning
    
    def quat_to_yaw(self, q : torch.Tensor):
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

    def yaw_quat(self, yaw):
        return torch.tensor([math.cos(yaw/2.0), 0.0, 0.0, math.sin(yaw/2.0)], dtype=self._dtype, device=self._device)

    def _quat_inverse(self, q: torch.Tensor) -> torch.Tensor:
        # inverse for unit quaternion: [w, -x, -y, -z]
        qi = q.clone()
        qi[..., 1:] = -qi[..., 1:]
        return qi
