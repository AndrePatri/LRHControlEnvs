# Copyright (C) 2023  Andrea Patrizi (AndrePatri, andreapatrizi1b6e6@gmail.com)
# 
# This file is part of OmniRoboGym and distributed under the General Public License version 2 license.
# 
# OmniRoboGym is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
# 
# OmniRoboGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with OmniRoboGym.  If not, see <http://www.gnu.org/licenses/>.
# 
from isaacsim import SimulationApp

import carb

import os
import signal

import torch
import numpy as np

from typing import Union, Tuple, Dict, List

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from omni_robo_gym.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel

from lrhc_control.envs.lrhc_remote_env_base import LRhcEnvBase
from OmniRoboGym.omni_robo_gym.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl

class XMjSimEnv(LRhcEnvBase):

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
        use_gpu: bool = True,
        dtype: torch.dtype = torch.float32,
        override_low_lev_controller: bool = False):

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
        # BaseTask.__init__(self,name=self._name,offset=None)

    def _pre_setup(self):
        
        self._render = (not self._env_opts["headless"])

    def _parse_env_opts(self):
        xmj_opts={}
        xmj_opts["use_gpu"]=False
        xmj_opts["device"]="cpu"
        xmj_opts["sim_device"]="cpu" if xmj_opts["use_gpu"] else "cpu"
        xmj_opts["physics_dt"]=1e-3
        xmj_opts["rendering_dt"]=xmj_opts["physics_dt"]
        xmj_opts["substeps"]=1 # number of physics steps to be taken for for each rendering step
        xmj_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        xmj_opts["use_diff_vels"] = False

        xmj_opts.update(self._env_opts) # update defaults with provided opts
        xmj_opts["rendering_dt"]=xmj_opts["physics_dt"]
        
        if not xmj_opts["use_gpu"]: # don't use GPU at all
            xmj_opts["use_gpu_pipeline"]=False
            xmj_opts["device"]="cpu"
            xmj_opts["sim_device"]="cpu"
        else: # use GPU
            Journal.log(self.__class__.__name__,
            "_parse_env_opts",
            "GPU not supported yet for XMjSimEnv!!",
            LogType.EXCEP,
            throw_when_excep = True)        
        # overwrite env opts in case some sim params were missing
        self._env_opts=xmj_opts

        # update device flag based on sim opts
        self._device=xmj_opts["device"]
        self._use_gpu=xmj_opts["use_gpu"]

    def _init_world(self):
        
        info = "Using sim device: " + str(self._env_opts["sim_device"])
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
                         
        big_info = "[World] Creating Mujoco-xbot2 simulation " + self._name + "\n" + \
            "use_gpu_pipeline: " + str(self._env_opts["use_gpu_pipeline"]) + "\n" + \
            "device: " + str(self._env_opts["sim_device"]) + "\n" +\
            "integration_dt: " + str(self._env_opts["physics_dt"]) + "\n" + \
            "rendering_dt: " + str(self._env_opts["rendering_dt"]) + "\n" 
        Journal.log(self.__class__.__name__,
            "_init_world",
            big_info,
            LogType.STAT,
            throw_when_excep = True)
    
        self._configure_scene()

        # if "enable_viewport" in sim_params:
        #     self._render = sim_params["enable_viewport"]

    def _configure_scene(self):

        # environment 
        self._fix_base = [False] * len(self._robot_names)
        self._self_collide = [False] * len(self._robot_names)
        self._merge_fixed = [True] * len(self._robot_names)
        
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            urdf_path = self._robot_urdf_paths[robot_name]
            srdf_path = self._robot_srdf_paths[robot_name]
            fix_base = self._fix_base[i]
            self_collide = self._self_collide[i]
            merge_fixed = self._merge_fixed[i]
            self._generate_rob_descriptions(robot_name=robot_name, 
                                    urdf_path=urdf_path,
                                    srdf_path=srdf_path)
            
            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing sim pre-setup...",
                        LogType.STAT,
                        throw_when_excep = True)
     
            self._reset_sim()
            self._fill_robot_info_from_world() 
            # initializes robot state data
            # self._init_robots_state()
            # update solver options 
            self._print_envs_info() # debug print

            self.scene_setup_completed = True

    def _render_sim(self, mode="human"):
        pass

    def _close(self):
        pass
    
    def _step_sim(self): 
        pass

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=,
            device=self._device,
            dtype=self._dtype,
            enable_safety=True,
            urdf_path=self._urdf_dump_paths[robot_name],
            config_path=self._jnt_imp_config_paths[robot_name],
            enable_profiling=False,
            debug_checks=self._debug,
            override_art_controller=self._override_low_lev_controller)
        
        return jnt_imp_controller

    def _reset(self,
        env_indxs: torch.Tensor = None,
        robot_names: List[str] =None,
        randomize: bool = False):

        self._reset_state(env_indxs=env_indxs,
            robot_names=robot_names,
            randomize=randomize)

        for i in range(len(robot_names)):
            self._reset_jnt_imp_control(robot_name=robot_names[i],
                                env_indxs=env_indxs)


    def _reset_sim(self):
        self._world.reset(soft=False)
    
    def _reset_state(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None,
            randomize: bool = False):

        rob_names = robot_names if (robot_names is not None) else self._robot_names
        if env_indxs is not None:
            if self._debug:
                if self._use_gpu:
                    if not env_indxs.device.type == "cuda":
                            error = "Provided env_indxs should be on GPU!"
                            Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
                else:
                    if not env_indxs.device.type == "cpu":
                        error = "Provided env_indxs should be on CPU!"
                        Journal.log(self.__class__.__name__,
                            "_step_jnt_imp_control",
                            error,
                            LogType.EXCEP,
                            True)
            for i in range(len(rob_names)):
                robot_name = rob_names[i]
                if randomize:
                    self._randomize_yaw(robot_name=robot_name,env_indxs=env_indxs)

                # root q
                self._robots_art_views[robot_name].set_world_poses(positions = self._root_p_default[robot_name][env_indxs, :],
                                                    orientations=self._root_q_default[robot_name][env_indxs, :],
                                                    indices = env_indxs)
                # jnts q
                self._robots_art_views[robot_name].set_joint_positions(positions = self._jnts_q_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
                # root v and omega
                self._robots_art_views[robot_name].set_joint_velocities(velocities = self._jnts_v_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
                # jnts v
                concatenated_vel = torch.cat((self._root_v_default[robot_name][env_indxs, :], 
                                                self._root_omega_default[robot_name][env_indxs, :]), dim=1)
                self._robots_art_views[robot_name].set_velocities(velocities = concatenated_vel,
                                                        indices = env_indxs)
                # jnts eff
                self._robots_art_views[robot_name].set_joint_efforts(efforts = self._jnts_eff_default[robot_name][env_indxs, :],
                                                        indices = env_indxs)
        else:

            for i in range(len(rob_names)):
                robot_name = rob_names[i]

                if randomize:
                    self.randomize_yaw(robot_name=robot_name,env_indxs=None)

                # root q
                self._robots_art_views[robot_name].set_world_poses(positions = self._root_p_default[robot_name][:, :],
                                                    orientations=self._root_q_default[robot_name][:, :],
                                                    indices = None)
                # jnts q
                self._robots_art_views[robot_name].set_joint_positions(positions = self._jnts_q_default[robot_name][:, :],
                                                        indices = None)
                # root v and omega
                self._robots_art_views[robot_name].set_joint_velocities(velocities = self._jnts_v_default[robot_name][:, :],
                                                        indices = None)
                # jnts v
                concatenated_vel = torch.cat((self._root_v_default[robot_name][:, :], 
                                                self._root_omega_default[robot_name][:, :]), dim=1)
                self._robots_art_views[robot_name].set_velocities(velocities = concatenated_vel,
                                                        indices = None)
                # jnts eff
                self._robots_art_views[robot_name].set_joint_efforts(efforts = self._jnts_eff_default[robot_name][:, :],
                                                        indices = None)

        # we update the robots state 
        self._update_state_from_sim(env_indxs=env_indxs, 
                        robot_names=rob_names)
        
    def _import_urdf(self, 
        robot_name: str,
        fix_base = False, 
        self_collide = False, 
        merge_fixed = True):
        
        import_config=_urdf.ImportConfig()
        # status,import_config=omni_kit.commands.execute("URDFCreateImportConfig")

        Journal.log(self.__class__.__name__,
            "update_root_offsets",
            "importing robot URDF",
            LogType.STAT,
            throw_when_excep = True)
        _urdf.acquire_urdf_interface()  
        # we overwrite some settings which are bound to be fixed
        import_config.merge_fixed_joints = merge_fixed # makes sim more stable
        # in case of fixed joints with light objects
        import_config.import_inertia_tensor = True
        # import_config.convex_decomp = False
        import_config.fix_base = fix_base
        import_config.self_collision = self_collide
        # import_config.distance_scale = 1
        # import_config.make_default_prim = True
        # import_config.create_physics_scene = True
        # import_config.default_drive_strength = 1047.19751
        # import_config.default_position_drive_damping = 52.35988
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import URDF
        success, robot_prim_path_default = omni_kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self._urdf_dump_paths[robot_name],
            import_config=import_config, 
            # get_articulation_root=True,
        )

        robot_base_prim_path = self._env_opts["template_env_ns"] + "/" + robot_name
        # moving default prim to base prim path for cloning
        move_prim(robot_prim_path_default, # from
                robot_base_prim_path) # to
        
        return success

    def apply_collision_filters(self, 
                                physicscene_path: str, 
                                coll_root_path: str):

        self._cloner.filter_collisions(physicsscene_path = physicscene_path,
                                collision_root_path = coll_root_path, 
                                prim_paths=self._envs_prim_paths, 
                                global_paths=[self._env_opts["ground_plane_prim_path"]] # can collide with these prims
                                )

    def _update_state_from_sim(self,
                env_indxs: torch.Tensor = None,
                robot_names: List[str] = None):
        
        if self._env_opts["use_diff_vels"]:
            self._get_robots_state(dt = self.physics_dt(),
                            env_indxs = env_indxs,
                            robot_names = robot_names) # updates robot states
            # but velocities are obtained via num. differentiation
        else:
            self._get_robots_state(env_indxs = env_indxs,
                            robot_names = robot_names) # velocities directly from simulator (can 
            # introduce relevant artifacts, making them unrealistic)

    def _get_robots_state(self, 
        env_indxs: torch.Tensor = None,
        robot_names: List[str] = None,
        dt: float = None, 
        reset: bool = False):
         
        rob_names = robot_names if (robot_names is not None) else self._robot_names
        if env_indxs is not None:
            for i in range(0, len(rob_names)):
                robot_name = rob_names[i]
                pose = self._robots_art_views[robot_name].get_world_poses( 
                                                clone = True,
                                                indices=env_indxs) # tuple: (pos, quat)
                
                self._root_p[robot_name][env_indxs, :] = pose[0] 
                self._root_q[robot_name][env_indxs, :] = pose[1] # root orientation
                self._jnts_q[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_positions(
                                                clone = True,
                                                indices=env_indxs) # joint positions 
                if dt is None:
                    # we get velocities from the simulation. This is not good since 
                    # these can actually represent artifacts which do not have physical meaning.
                    # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                    self._root_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                                clone = True,
                                                indices=env_indxs) # root lin. velocity               
                    self._root_omega[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                                clone = True,
                                                indices=env_indxs) # root ang. velocity
                    self._jnts_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                                clone = True,
                                                indices=env_indxs) # joint velocities
                else:
                    # differentiate numerically
                    if not reset:                    
                        self._root_v[robot_name][env_indxs, :] = (self._root_p[robot_name][env_indxs, :] - \
                                                        self._root_p_prev[robot_name][env_indxs, :]) / dt 
                        self._root_omega[robot_name][env_indxs, :] = quat_to_omega(self._root_q[robot_name][env_indxs, :], 
                                                                    self._root_q_prev[robot_name][env_indxs, :], 
                                                                    dt)
                        self._jnts_v[robot_name][env_indxs, :] = (self._jnts_q[robot_name][env_indxs, :] - \
                                                        self._jnts_q_prev[robot_name][env_indxs, :]) / dt
                    else:
                        # to avoid issues when differentiating numerically
                        self._root_v[robot_name][env_indxs, :].zero_()
                        self._root_omega[robot_name][env_indxs, :].zero_()
                        self._jnts_v[robot_name][env_indxs, :].zero_()
                    # update "previous" data for numerical differentiation
                    self._root_p_prev[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :] 
                    self._root_q_prev[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]
                    self._jnts_q_prev[robot_name][env_indxs, :] = self._jnts_q[robot_name][env_indxs, :]

                self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                                clone = True,
                                                joint_indices=None,
                                                indices=env_indxs) # measured joint efforts (computed by joint force solver)
        else:
            # updating data for all environments
            for i in range(0, len(rob_names)):
                robot_name = rob_names[i]
                pose = self._robots_art_views[robot_name].get_world_poses( 
                                                clone = True) # tuple: (pos, quat)
                self._root_p[robot_name][:, :] = pose[0]  
                self._root_q[robot_name][:, :] = pose[1] # root orientation
                self._jnts_q[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_positions(
                                                clone = True) # joint positions 
                if dt is None:
                    # we get velocities from the simulation. This is not good since 
                    # these can actually represent artifacts which do not have physical meaning.
                    # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                    self._root_v[robot_name][:, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                                clone = True) # root lin. velocity 
                    self._root_omega[robot_name][:, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                                    clone = True) # root ang. velocity
                    self._jnts_v[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                                    clone = True) # joint velocities
                else: 
                    # differentiate numerically
                    if not reset:        
                        self._root_v[robot_name][:, :] = (self._root_p[robot_name][:, :] - \
                                                        self._root_p_prev[robot_name][:, :]) / dt 
                        self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q[robot_name][:, :], 
                                                                    self._root_q_prev[robot_name][:, :], 
                                                                    dt)
                        self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name][:, :] - \
                                                        self._jnts_q_prev[robot_name][:, :]) / dt
                        # self._jnts_v[robot_name][:, :].zero_()
                    else:
                        # to avoid issues when differentiating numerically
                        self._root_v[robot_name][:, :].zero_()
                        self._root_omega[robot_name][:, :].zero_()
                        self._jnts_v[robot_name][:, :].zero_()
                    # update "previous" data for numerical differentiation
                    self._root_p_prev[robot_name][:, :] = self._root_p[robot_name][:, :] 
                    self._root_q_prev[robot_name][:, :] = self._root_q[robot_name][:, :]
                    self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name][:, :]
                
                self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                                clone = True) # measured joint efforts (computed by joint force solver)
    
    def _move_jnts_to_homing(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robots_art_views[robot_name].set_joints_default_state(positions=self._homing, 
                velocities = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                    dtype=self._dtype, device=self._device), 
                efforts = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                    dtype=self._dtype, device=self._device))
                
    def _move_root_to_defconfig(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robots_art_views[robot_name].set_default_state(positions=self._root_p_default[robot_name], 
                orientations=self._root_q_default[robot_name])
            
    def _get_solver_info(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._solver_position_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_position_iteration_counts()
            self._solver_velocity_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_velocity_iteration_counts()
            self._solver_stabilization_threshs[robot_name] = self._robots_art_views[robot_name].get_stabilization_thresholds()
    
    def _update_art_solver_options(self):
        
        # sets new solver iteration options for specifc articulations
        self._get_solver_info() # gets current solver info for the articulations of the 
        # environments, so that dictionaries are filled properly
        
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            # increase by a factor
            self._solver_position_iteration_counts[robot_name] = torch.full((self._num_envs,), self._solver_position_iteration_count)
            self._solver_velocity_iteration_counts[robot_name] = torch.full((self._num_envs,), self._solver_velocity_iteration_count)
            self._solver_stabilization_threshs[robot_name] = torch.full((self._num_envs,), self._solver_stabilization_thresh)
            self._robots_art_views[robot_name].set_solver_position_iteration_counts(self._solver_position_iteration_counts[robot_name])
            self._robots_art_views[robot_name].set_solver_velocity_iteration_counts(self._solver_velocity_iteration_counts[robot_name])
            self._robots_art_views[robot_name].set_stabilization_thresholds(self._solver_stabilization_threshs[robot_name])
            self._get_solver_info() # gets again solver info for articulation, so that it's possible to debug if
            # the operation was successful

    def _print_envs_info(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            task_info = f"[{robot_name}]" + "\n"
            Journal.log(self.__class__.__name__,
                "_print_envs_info",
                task_info,
                LogType.STAT,
                throw_when_excep = True)
    
    def _fill_robot_info_from_world(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._robot_bodynames[robot_name] = self._robots_art_views[robot_name].body_names
            self._robot_n_links[robot_name] = self._robots_art_views[robot_name].num_bodies
            self._robot_n_dofs[robot_name] = self._robots_art_views[robot_name].num_dof
            self._robot_dof_names[robot_name] = self._robots_art_views[robot_name].dof_names
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        pass
    
    def _init_contact_sensors(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            # creates base contact sensor (which is then cloned)
            if self.omni_contact_sensors[robot_name] is not None:
                self.omni_contact_sensors[robot_name].create_contact_sensors(
                                                        self._world,
                                                        envs_namespace=self._env_opts["envs_ns"])
    
    def _init_robots_state(self):

        self._calc_robot_distrib()

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
        
            # root p (measured, previous, default)
            self._root_p[robot_name] =
            self._root_p_prev[robot_name] = 
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] =
            # root q (measured, previous, default)
            self._root_q[robot_name] =
            self._root_q_prev[robot_name] = 
            self._root_q_default[robot_name] = 
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] =
            self._jnts_q_prev[robot_name] =
            self._jnts_q_default[robot_name] = torch.full((self._jnts_q[robot_name].shape[0], 
                                                           self._jnts_q[robot_name].shape[1]), 
                                                            0.0, 
                                                            dtype=self._dtype, 
                                                            device=self._device)
            
            # root v (measured, default)
            self._root_v[robot_name] = 

            self._root_v_default[robot_name] = torch.full((self._root_v[robot_name].shape[0], self._root_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            # root omega (measured, default)
            self._root_omega[robot_name] = 
            self._root_omega_default[robot_name] = torch.full((self._root_omega[robot_name].shape[0], self._root_omega[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            # joints v (measured, default)
            self._jnts_v[robot_name] = 
            self._jnts_v_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                0.0, 
                                                dtype=self._dtype, 
                                                device=self._device)
            self._jnts_eff_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                    0.0, 
                                                    dtype=self._dtype, 
                                                    device=self._device)
            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            # self._update_root_offsets(robot_name)

    def current_tstep(self):
        pass
    
    def current_time(self):
        pass
    
    def physics_dt(self):
        pass
    
    def rendering_dt(self):
        pass
    
    def set_physics_dt(self, physics_dt:float):
        pass
    
    def set_rendering_dt(self, rendering_dt:float):
        pass
    
    def _robot_jnt_names(self, robot_name: str):
        pass
