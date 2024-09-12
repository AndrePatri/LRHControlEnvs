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

import torch
import numpy as np

from typing import Union, Tuple, Dict, List

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

from omni_robo_gym.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel

from lrhc_control.envs.lrhc_remote_env_base import LRhcEnvBase
from omni_robo_gym.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from adarl_ros.adapters.XbotMjAdapter import XbotMjAdapter
from xbot2_mujoco.PyXbotMjSimEnv import LoadingUtils

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

        if not len(robot_names)==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Parallel simulation is not supported yet!",
            LogType.EXCEP,
            throw_when_excep = True)

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

        xmj_opts["headless"] = False
        xmj_opts["init_timesteps"] = 0
        xmj_opts["xmj_files_dir"]=None
        xmj_opts["xmj_timeout"]=1000

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
            
            self._xmj_helper = LoadingUtils(self._name)
            xmj_files_dir=self._env_opts["xmj_files_dir"]
            self._xmj_helper.set_simopt_path(xmj_files_dir+"/sim_opt.xml")
            self._xmj_helper.set_world_path(xmj_files_dir+"/world.xml")
            self._xmj_helper.set_sites_path(xmj_files_dir+"/sites.xml")
            
            self._xmj_helper.set_urdf_path(self._urdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_srdf_path(self._srdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_xbot_config_path(self._jnt_imp_config_paths[self._robot_names[0]])
            self._xmj_helper.generate()
            self._mj_xml_path = self.loader.xml_path()

            self._xmj_adapter=XbotMjAdapter(model_fpath=self._mj_xml_path,
                model_name=self._robot_names[0],
                xbot2_config_path=self._jnt_imp_config_paths[self._robot_names[0]],
                headless=self._env_opts["headless"],
                init_steps=self._env_opts["init_timesteps"],
                timeout_ms=self._env_opts["xmj_timeout"],
                forced_ros_master_uri= None,
                maxObsDelay=float("+inf"),
                blocking_observation=False,
                is_floating_base=True,
                reference_frame="world",
                torch_device=self._device,
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=100.0,
                allow_fallback=True,
                enable_filters=True)
            self._xmj_adapter.startup()
            to_monitor=[]
            jnt_names_sim=self._robot_jnt_names()
            for jnt in range(len(jnt_names_sim)):
                to_monitor.append((self._robot_names[i],jnt_names_sim[jnt]))
            self._xmj_adapter.set_monitored_joints(to_monitor)
            self._xmj_adapter.set_impedance_controlled_joints(to_monitor)

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
        
    def _render_sim(self, mode="human"):
        pass

    def _close(self):
        pass
    
    def _step_sim(self): 
        pass

    def _pre_step(self): 
        super()._pre_step()
        self._xmj_adapter.setJointsImpedanceCommand(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd())

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = XMjJntImpCntrl(xbot_adapter=self._xmj_adapter,
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
        self._xmj_adapter.resetWorld()
    
    def _reset_state(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None,
            randomize: bool = False):

        rob_names = robot_names if (robot_names is not None) else self._robot_names

        for i in range(len(rob_names)):
            robot_name = rob_names[i]

            if randomize:
                self.randomize_yaw(robot_name=robot_name,env_indxs=None)
            self._move_root_to_defconfig()

        # we update the robots state 
        self._update_state_from_sim(env_indxs=env_indxs, 
                        robot_names=rob_names)

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
        
        for i in range(0, len(rob_names)):
            robot_name = rob_names[i]
            
            self._root_p[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().p)
            self._root_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().q)
            self._jnts_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q)
            if dt is None:
                # we get velocities from the simulation. This is not good since 
                # these can actually represent artifacts which do not have physical meaning.
                # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                self._root_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[0:3])             
                self._root_omega[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[3:6])  
                self._jnts_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v)

            else:
                # differentiate numerically
                if not reset:                    
                    self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
                                                    self._root_p_prev[robot_name]) / dt 
                    self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q[robot_name], 
                                                                self._root_q_prev[robot_name], 
                                                                dt)
                    self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                                                    self._jnts_q_prev[robot_name]) / dt
                else:
                    # to avoid issues when differentiating numerically
                    self._root_v[robot_name][:, :].zero_()
                    self._root_omega[robot_name][:, :].zero_()
                    self._jnts_v[robot_name][:, :].zero_()
                # update "previous" data for numerical differentiation
                self._root_p_prev[robot_name][:, :] = self._root_p[robot_name]
                self._root_q_prev[robot_name][:, :] = self._root_q[robot_name]
                self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

            self._jnts_eff[robot_name][env_indxs, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff)
    
    def _move_jnts_to_homing(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._xmj_adapter.xmj_env().move_to_homing_now()
                
    def _move_root_to_defconfig(self):
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
            self._xmj_adapter.xmj_env().set_pi(self._root_p_default[robot_name].numpy())
            self._xmj_adapter.xmj_env().set_qi(self._root_q_default[robot_name].numpy())
        self._xmj_adapter.step() # perform a sim step to update state
            
    def _get_solver_info(self):
        raise NotImplementedError()

    def _print_envs_info(self):
        raise NotImplementedError()
    
    def _fill_robot_info_from_world(self):
        raise NotImplementedError()
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        raise NotImplementedError()
    
    def _init_contact_sensors(self):
        raise NotImplementedError()
    
    def _init_robots_state(self):

        self._calc_robot_distrib()

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
        
            # root p (measured, previous, default)
            self._root_p[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().p.copy())
            self._root_p_prev[robot_name] = self._root_p[robot_name].clone()
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = self._root_p[robot_name].clone()
            # root q (measured, previous, default)
            self._root_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().q.copy())
            self._root_q_prev[robot_name] = self._root_q[robot_name].clone()
            self._root_q_default[robot_name] = self._root_q[robot_name].clone()
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q.copy())
            self._jnts_q_prev[robot_name] = self._jnts_q[robot_name].clone()
            self._jnts_q_default[robot_name] = self._jnts_q[robot_name].clone()
            
            # root v (measured, default)
            self._root_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[0:3])

            self._root_v_default[robot_name] = self._root_v[robot_name].clone()

            # root omega (measured, default)
            self._root_omega[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[3:6])
            self._root_omega_default[robot_name] = self._root_omega[robot_name].clone()

            # joints v (measured, default)
            self._jnts_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v.copy())
            self._jnts_v_default[robot_name] = self._jnts_v[robot_name].clone()
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff.copy())
            self._jnts_eff_default[robot_name] = self._jnts_eff[robot_name].clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            # self._update_root_offsets(robot_name)

    def current_tstep(self):
        return self._xmj_adapter.xmj_env().step_counter
    
    def current_time(self):
        return self._xmj_adapter.getEnvTimeFromReset()
    
    def physics_dt(self):
        return self._xmj_adapter.xmj_env().physics_dt
    
    def rendering_dt(self):
        return self._xmj_adapter.xmj_env().physics_dt
    
    def set_physics_dt(self, physics_dt:float):
        raise NotImplementedError()
    
    def set_rendering_dt(self, rendering_dt:float):
        raise NotImplementedError()
    
    def _robot_jnt_names(self, robot_name: str):
        return self._xmj_adapter.xmj_env().jnt_names()
