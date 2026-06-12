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

import os
import math
import xml.etree.ElementTree as ET

import torch
import numpy as np

from typing import Dict, List
from typing_extensions import override

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc_envs.utils.math_utils import quat_to_omega

from aug_mpc_envs.utils.height_sensor import HeightGridSensor
from aug_mpc_envs.utils.xmj_jnt_imp_cntrl import XMjJntImpCntrl
from adarl.adapters.XbotMjAdapter import XbotMjAdapter
from xbot2_mujoco.PyXbotMjSim import LoadingUtils
from mpc_hive.utilities.math_utils_torch import world2base_frame,world2base_frame3D

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase

class XMjSimEnv(AugMPCWorldInterfaceBase):

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
            "Multi-robot simulation is not supported yet!",
            LogType.EXCEP,
            throw_when_excep = True)

        if not num_envs==1:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Parallel simulation is not supported yet!",
            LogType.EXCEP,
            throw_when_excep = True)
            
        if use_gpu:
            Journal.log(self.__class__.__name__,
            "__init__",
            "Only CPU simulation is supported!",
            LogType.EXCEP,
            throw_when_excep = True)

        self._xmj_adapter_init_tsteps=n_init_step
        
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
            override_low_lev_controller=override_low_lev_controller)
        # BaseTask.__init__(self,name=self._name,offset=None)

    def is_running(self):
        xbot_control_running=self._xmj_adapter.is_xbot_control_running()
        if not xbot_control_running:
            Journal.log(self.__class__.__name__,
            "_is_running",
            "XBot2/ZMQ control plugin is not running",
            LogType.EXCEP,
            throw_when_excep = False)

        sim_running=self._xmj_adapter.sim_is_running()
        if not sim_running:
            Journal.log(self.__class__.__name__,
            "_is_running",
            "simulation is not running",
            LogType.EXCEP,
            throw_when_excep = False)

        return xbot_control_running and sim_running and self._isrunning
    
    def _pre_setup(self):
        
        self._render = (not self._env_opts["headless"])
        self._height_sensors = {}
        self._height_imgs = {}
        self._height_field_data = None

    def _parse_env_opts(self):
        xmj_opts={}
        xmj_opts["use_gpu"]=False
        xmj_opts["state_from_xbot"]=True
        xmj_opts["device"]="cpu"
        xmj_opts["sim_device"]="cpu" if xmj_opts["use_gpu"] else "cpu"
        xmj_opts["physics_dt"]=1e-3
        xmj_opts["rendering_dt"]=xmj_opts["physics_dt"]
        xmj_opts["render_to_file"]=False
        xmj_opts["render_fps"]=60
        xmj_opts["substeps"]=1 # number of physics steps to be taken for for each rendering step
        xmj_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        xmj_opts["use_diff_vels"] = False

        xmj_opts["headless"] = False
        xmj_opts["xmj_files_dir"]=None
        xmj_opts["xmj_timeout"]=30000
        xmj_opts["xbot2_filter_prof"]="medium"
        xmj_opts["xbot2_sense_timeout_s"]=2.0

        xmj_opts["base_linkname"]="base_link"
        xmj_opts["spawning_height"]=1.0 # same default as the Isaac interface; set explicitly per robot.
        # Set to None to fall back to the model/MJCF base height instead.

        xmj_opts["use_mpc_pos_for_robot"]=False # default to using pos from sim
        xmj_opts["use_rel_q_from_startup"]=True
        xmj_opts["height_map_resolution"]=0.05
        xmj_opts["height_map_margin"]=0.5
        xmj_opts["height_sensor_forward_offset"]=0.0
        xmj_opts["height_sensor_lateral_offset"]=0.0
        xmj_opts["generate_stepup_terrain"]=False
        xmj_opts["stepup_terrain_size"]=100.0
        xmj_opts["stepup_stairs_ratio"]=0.99
        xmj_opts["stepup_platform_size"]=50.0
        xmj_opts["stepup_step_height_lb"]=0.08
        xmj_opts["stepup_step_height_ub"]=0.15
        xmj_opts["stepup_step_width_lb"]=0.4
        xmj_opts["stepup_step_width_ub"]=1.5
        xmj_opts["stepup_n_steps"]=25
        xmj_opts["stepup_area_factor"]=0.7
        xmj_opts["stepup_res_low"]=0.1
        xmj_opts["stepup_res_high"]=0.03
        xmj_opts["stepup_position"]=np.array([0.0, 0.0, 0.0])
        xmj_opts["stepup_random_n_steps"]=False
        xmj_opts["stepup_wall_height"]=2.0
        xmj_opts["stepup_seed"]=None

        xmj_opts["ramp_to_homing"]=True
        xmj_opts["ramp_impedances"]=True
        xmj_opts["jnt_imp_ramp_time"]=1.0
        xmj_opts["jnt_pos_ramp_time"]=4.0
        xmj_opts["jnt_imp_ramp_time_onclose"]=2.0

        xmj_opts.update(self._env_opts) # update defaults with provided opts
        if xmj_opts["spawning_height"] is not None:
            xmj_opts["spawning_height"] = float(xmj_opts["spawning_height"])
        xmj_opts["xbot2_sense_timeout_s"] = float(xmj_opts["xbot2_sense_timeout_s"])
        xmj_opts["rendering_dt"]=1/xmj_opts["render_fps"]        
        xmj_opts["height_sensor_pixels"]=int(xmj_opts["height_sensor_pixels"])
        xmj_opts["height_sensor_resolution"]=float(xmj_opts["height_sensor_resolution"])
        xmj_opts["enable_height_sensor"]=bool(xmj_opts.get("enable_height_sensor", False))
        # keep visualization/state anchored to simulator ground truth when height sensor is on
        if xmj_opts["enable_height_sensor"]:
            xmj_opts["use_mpc_pos_for_robot"]=False
        
        xmj_opts["run_cluster_bootstrap"] = True # to avoid initial jumps

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

    @override
    def _setup(self):

        # last thing called before spinning
        setup_ok=super()._setup()

        self._isrunning=True

        return setup_ok
    
    def _configure_scene(self):

        # environment 
        self._fix_base = [False] * len(self._robot_names)
        self._self_collide = [False] * len(self._robot_names)
        self._merge_fixed = [True] * len(self._robot_names)

        # prepare world XML (optionally augment with procedurally generated terrain)
        self._world_xml_path = self._prepare_world_xml()

        # pre-compute static height map from world.xml (used by height sensor)
        if self._env_opts["enable_height_sensor"]:
            self._height_field_data = self._build_static_heightmap_from_world(world_path=self._world_xml_path)
        
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
            self._patch_generated_urdf_for_mujoco(
                urdf_path=self._urdf_dump_paths[robot_name])
            
            self._xmj_helper = LoadingUtils(self._name)
            xmj_files_dir=self._env_opts["xmj_files_dir"]
            if xmj_files_dir is None:
                Journal.log(self.__class__.__name__,
                    "_configure_scene",
                    "xmj_files_dir is None. It should be a valid path to where sim_opt.xml, world.xml and sites.xml files are.",
                    LogType.EXCEP,
                    throw_when_excep = True)
            self._xmj_helper.set_simopt_path(xmj_files_dir+"/sim_opt.xml")
            self._xmj_helper.set_world_path(self._world_xml_path)
            self._xmj_helper.set_sites_path(xmj_files_dir+"/sites.xml")
            
            self._xmj_helper.set_urdf_path(self._urdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_srdf_path(self._srdf_dump_paths[self._robot_names[0]])
            self._xmj_helper.set_xbot_config_path(self._jnt_imp_config_paths[self._robot_names[0]])
            self._xmj_helper.generate()
            self._mj_xml_path = self._xmj_helper.xml_path()

            self._xmj_adapter=XbotMjAdapter(model_fpath=self._mj_xml_path,
                model_name=self._robot_names[0],
                xbot2_config_path=self._jnt_imp_config_paths[self._robot_names[0]],
                stepLength_sec=self._env_opts["physics_dt"],
                headless=self._env_opts["headless"],
                init_steps=self._xmj_adapter_init_tsteps,
                timeout_ms=self._env_opts["xmj_timeout"],
                forced_ros_master_uri= None,
                maxObsDelay=float("+inf"),
                blocking_observation=False,
                is_floating_base=True,
                reference_frame="world",
                torch_device=torch.device(self._device),
                fallback_cmd_stiffness=200.0,
                fallback_cmd_damping=100.0,
                allow_fallback=True,
                enable_filters=True,
                base_link=self._env_opts["base_linkname"],
                root_spawn_height=self._env_opts["spawning_height"],
                render_to_file=self._env_opts["render_to_file"],
                render_fps=self._env_opts["render_fps"],
                sense_timeout_s=self._env_opts["xbot2_sense_timeout_s"])
            # self._xmj_adapter.build_scenario()
            with open(self._urdf_dump_paths[self._robot_names[0]], "r", encoding="utf-8") as f:
                urdf_str = f.read()
            with open(self._srdf_dump_paths[self._robot_names[0]], "r", encoding="utf-8") as f:
                srdf_str = f.read()
            self._xmj_adapter.startup(urdf=urdf_str,
                            srdf=srdf_str)
            self._xmj_adapter.position_ramp_time=self._env_opts["jnt_pos_ramp_time"] # [s]
            self._xmj_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time"] # [s]
            self._xmj_adapter.set_filters(set_enabled=True, 
                profile_name=self._env_opts["xbot2_filter_prof"])

            to_monitor=[]
            jnt_names_sim=self._robot_jnt_names(robot_name=robot_name)
            for jnt in range(len(jnt_names_sim)):
                to_monitor.append((self._robot_names[i],jnt_names_sim[jnt]))
            self._xmj_adapter.set_monitored_joints(to_monitor)
            self._xmj_adapter.set_impedance_controlled_joints(to_monitor)

            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing sim pre-setup...",
                        LogType.STAT,
                        throw_when_excep = True)

            if self._env_opts["enable_height_sensor"]:
                # height grid sensor (static height map parsed from world.xml)
                self._height_sensors[robot_name] = HeightGridSensor(
                    terrain_utils=self._height_field_data,
                    grid_size=int(self._env_opts["height_sensor_pixels"]),
                    resolution=float(self._env_opts["height_sensor_resolution"]),
                    forward_offset=float(self._env_opts["height_sensor_forward_offset"]),
                    lateral_offset=float(self._env_opts["height_sensor_lateral_offset"]),
                    n_envs=self._num_envs,
                    device=self._device,
                    dtype=self._dtype)
     
            self._reset_sim()
            self._fill_robot_info_from_world() 
            # initializes robot state data
            self._init_robots_state()
            # update solver options 
            self._print_envs_info() # debug print

            self.scene_setup_completed = True

        # self._rospy_startime=rospy.get_time()

    @override
    def _xrdf_cmds(self, robot_name:str):
        cmds=super()._xrdf_cmds(robot_name=robot_name)
        for i, s in enumerate(cmds):
            if "floating_joint:=" in s: # mujoco needs a floating joint
                cmds[i] = "floating_joint:=true" 
        return cmds

    def _patch_generated_urdf_for_mujoco(self, urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        patched = 0

        for geometry in root.findall(".//visual/geometry"):
            mesh = geometry.find("mesh")
            if mesh is None:
                continue

            filename = mesh.attrib.get("filename", "")
            if not filename.endswith("/realsense/d435.dae") and not filename.endswith("d435.dae"):
                continue

            geometry.remove(mesh)
            ET.SubElement(geometry, "box", {"size": "0.09 0.025 0.02505"})
            patched += 1

        if patched:
            tree.write(urdf_path, encoding="utf-8", xml_declaration=True)
            Journal.log(self.__class__.__name__,
                        "_patch_generated_urdf_for_mujoco",
                        f"Patched {patched} Realsense D435 visual mesh(es) in generated URDF.",
                        LogType.STAT,
                        throw_when_excep=True)

    def _render_sim(self, mode="human"):
        pass

    def _close(self):
        if not hasattr(self, "_xmj_adapter"):
            return

        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]
            adapter_started = getattr(self._xmj_adapter, "_started", False)

            # set filters to safe
            if adapter_started:
                self._xmj_adapter.set_filters(set_enabled=True,
                    profile_name="safe")

            # resets jnt imp gain to the startups with a ramp
            self._xmj_adapter.impedance_ramp_time=self._env_opts["jnt_imp_ramp_time_onclose"] # setting slower
            self._env_opts["ramp_to_homing"]=False # skip homing when closing
            
            # read last pos ref from jnt imp control before reset

            # ramp since impedances will generally be ramped up when cleaning up
            if adapter_started and robot_name in self._jnt_imp_controllers:
                self._reset_jnt_imp_control(robot_name=robot_name) # will set jnt imp gains to initial vals and
                # pos ref to homing and apply them with the adapter

            self._isrunning=False

        self._xmj_adapter.close()

    @override
    def _apply_cmds_to_jnt_imp_control(self, robot_name:str):

        super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name) # write to interface jnt imp control
        self._xmj_adapter.setJointsImpedanceCommand(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd()) # just set cmds, will be applied when stepping world
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
            self._xmj_adapter.apply_joint_ref_with_ramp(pvesd, tolerance=0.1)
        else: # set p ref to current value to avoid jumps (pref or meas. p?)
            reset_ref=self._p_ref_reset[robot_name]
            # reset_ref=self._jnts_q[robot_name]
            self._jnt_imp_controllers[robot_name].set_refs(
                pos_ref=reset_ref,
                robot_indxs = None)
            super()._apply_cmds_to_jnt_imp_control(robot_name=robot_name)
            
        if self._env_opts["ramp_impedances"]: # ramp impedances
            pvesd = self._jnt_imp_controllers[robot_name].get_pvesd()
            self._xmj_adapter.apply_joint_impedances_with_ramp(pvesd, tolerance=0.1) # ramps impeances

    def _step_world(self): 
        time_elapsed=self._xmj_adapter.step()
        if not (abs(time_elapsed-self.physics_dt())<1e-6):
            Journal.log(self.__class__.__name__,
                "_step_world",
                f"simulation stepped of {time_elapsed} [s], while expected one should be {self.physics_dt()} [s]",
                LogType.EXCEP,
                throw_when_excep = True)
        
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

    def _reset_sim(self):
        self._xmj_adapter.resetWorld()
    
    @override
    def _set_startup_jnt_imp_gains(self,
            robot_name:str, 
            env_indxs: torch.Tensor = None):
        super()._set_startup_jnt_imp_gains(robot_name=robot_name,env_indxs=env_indxs)
        # apply jnt imp cmds to xbot immediately to avoid robot
        self._xmj_adapter.apply_joint_impedances(self._jnt_imp_controllers[self._robot_names[0]].get_pvesd())

    def _reset_state(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            randomize: bool = False):

        if randomize:
            self._randomize_yaw(robot_name=robot_name,env_indxs=None)
            self._set_root_to_defconfig(robot_name=robot_name)
        
        self._reset_sim() # moves robot to homing and default root config in sim, so that we can read these values as initial state for the robot before applying any randomization
        
    def _read_root_state_from_robot(self,
            robot_name: str,
            env_indxs: torch.Tensor = None,
            ):
        
        if (not self._env_opts["state_from_xbot"]):
            self._get_root_state(numerical_diff=False,
                    env_indxs=env_indxs,
                    robot_name=robot_name)
        else:
            # raise NotImplementedError("Getting root state from xbot not implemented yet !")
            self._get_root_state_xbot(numerical_diff=False,
                    env_indxs=env_indxs,
                    robot_name=robot_name)

        # height grid sensor readout
        if robot_name in self._height_sensors:
            # always use simulator ground-truth pose for the height sensor, even when
            # we fall back to MPC pose for the rest of the pipeline (use_mpc_pos_for_robot)
            if self._env_opts["use_mpc_pos_for_robot"]:
                sim_p = torch.from_numpy(self._xmj_adapter.xmj_env().p).reshape(self._num_envs, -1).to(self._dtype)
                sim_q = torch.from_numpy(self._xmj_adapter.xmj_env().q).reshape(self._num_envs, -1).to(self._dtype)
                pos_src = sim_p if env_indxs is None else sim_p[env_indxs]
                quat_src = sim_q if env_indxs is None else sim_q[env_indxs]
            else:
                pos_src = self._root_p[robot_name] if env_indxs is None else self._root_p[robot_name][env_indxs]
                quat_src = self._root_q[robot_name] if env_indxs is None else self._root_q[robot_name][env_indxs]
            heights = self._height_sensors[robot_name].read(pos_src, quat_src)*1.0
            if env_indxs is None:
                self._height_imgs[robot_name] = heights
            else:
                self._height_imgs[robot_name][env_indxs] = heights.clone()

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
        
        self._root_p[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().p).reshape(self._num_envs, -1).to(self._dtype)
        self._root_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().q).reshape(self._num_envs, -1).to(self._dtype)

        dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate

        if not numerical_diff:
            # we get velocities from the simulation. This is not good since 
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            self._root_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)             
            self._root_omega[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[3:6]).reshape(self._num_envs, -1).to(self._dtype)        
            
            # for now obtain root a numerically
            self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
                                            self._root_v_prev[robot_name][env_indxs, :]) / dt 
            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
            
            self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
            self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]
            
        else:
            # differentiate numerically
            self._root_v[robot_name][:, :] = (self._root_p[robot_name] - \
                                            self._root_p_prev[robot_name]) / dt 
            self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name], 
                                                        self._root_q[robot_name], 
                                                        dt)

            self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
                                                self._root_v_prev[robot_name][env_indxs, :]) / dt 
            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 
            
            # update "previous" data for numerical differentiation
            self._root_p_prev[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_prev[robot_name][:, :] = self._root_q[robot_name]
            self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
            self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]

        if base_loc:
            # rotate robot twist in base local
            twist_w=torch.cat((self._root_v[robot_name], 
                self._root_omega[robot_name]), 
                dim=1)
            twist_base_loc=torch.cat((self._root_v_base_loc[robot_name], 
                self._root_omega_base_loc[robot_name]), 
                dim=1)
            world2base_frame(t_w=twist_w,q_b=self._root_q[robot_name],t_out=twist_base_loc)
            self._root_v_base_loc[robot_name]=twist_base_loc[:, 0:3]
            self._root_omega_base_loc[robot_name]=twist_base_loc[:, 3:6]

            # rotate robot a in base local
            a_w=torch.cat((self._root_a[robot_name], 
                self._root_alpha[robot_name]), 
                dim=1)
            a_base_loc=torch.cat((self._root_a_base_loc[robot_name], 
                self._root_alpha_base_loc[robot_name]), 
                dim=1)
            world2base_frame(t_w=a_w,q_b=self._root_q[robot_name],t_out=a_base_loc)
            self._root_a_base_loc[robot_name]=a_base_loc[:, 0:3]
            self._root_alpha_base_loc[robot_name]=a_base_loc[:, 3:6]

            world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
                v_out=self._gravity_normalized_base_loc[robot_name])

    def _get_root_state_xbot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        base_loc: bool = True):

        # update IMU and get base link state (assumed to return torch tensors)
        self._xmj_adapter.read_imu_data()
        frame_name, q, omega, linacc = self._xmj_adapter.get_base_link_state()

        # position handling (same as before)
        if self._env_opts["use_mpc_pos_for_robot"]:
            actions = self.cluster_servers[robot_name].get_actions()
            rhc_p = actions.root_state.get(data_type="p", gpu=self._use_gpu)
            self._root_p[robot_name][:, :] = rhc_p
        else:
            # in sim we get pos from sim
            self._root_p[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().p).reshape(self._num_envs, -1).to(self._dtype)

        # store raw IMU quaternion (ensure dtype/device). Startup yaw-rel projection
        # for MPC state publishing is now handled in the base world interface.
        self._root_q[robot_name][:, :] = torch.from_numpy(q).reshape(self._num_envs, -1).to(self._dtype)
        
        dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate sim we are in sim and we can enforce a constant rate

        if not numerical_diff:
            # we get velocities from the simulation. This is not good since 
            # these can actually represent artifacts which do not have physical meaning.
            # It's better to obtain them by differentiation to avoid issues with controllers, etc...
            # self._root_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().twist[0:3]).reshape(self._num_envs, -1).to(self._dtype)   
            self._root_omega[robot_name][:, :] = torch.from_numpy(omega).reshape(self._num_envs, -1).to(self._dtype)        
            
            self._root_a[robot_name][env_indxs, :] = torch.from_numpy(linacc).reshape(self._num_envs, -1).to(self._dtype)  

            self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                            self._root_omega_prev[robot_name][env_indxs, :]) / dt 

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
        
        dt= self.physics_dt() if self._override_low_lev_controller else self._cluster_dt[robot_name]

        self._jnts_q[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q).reshape(self._num_envs, -1).to(self._dtype)

        if not numerical_diff:
            self._jnts_v[robot_name][:, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v).reshape(self._num_envs, -1).to(self._dtype)     
        else: 
            self._jnts_v[robot_name][:, :] = self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt
            
            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

        self._jnts_eff[robot_name][env_indxs, :] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff).reshape(self._num_envs, -1).to(self._dtype) 

    def _get_robots_jnt_state_xbot(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):

        jnt_state_from_xbot=self._xmj_adapter.getJointsState().T # [3(p, v, e)x n_jnts]

        self._jnts_q[robot_name][:, :] = jnt_state_from_xbot[0,:]

        dt= None
        if numerical_diff:
            dt= self.physics_dt() if self._override_low_lev_controller else self._cluster_dt[robot_name]

        if dt is None:
            self._jnts_v[robot_name][:, :] = jnt_state_from_xbot[1,:]
        else: 
            self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name] - \
                self._jnts_q_prev[robot_name]) / dt
            
            self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name]

        self._jnts_eff[robot_name][env_indxs, :] = jnt_state_from_xbot[2,:]

    def _set_jnts_to_homing(self, robot_name: str):
        del robot_name
        self._xmj_adapter.move_to_homing_now()
                
    def _set_root_to_defconfig(self, robot_name: str):
        self._xmj_adapter.xmj_env().set_pi(self._root_p_default[robot_name].numpy())
        self._xmj_adapter.xmj_env().set_qi(self._root_q_default[robot_name].numpy())
        # self._xmj_adapter.xmj_env().move_base_to_now()

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
            self._root_p[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().p.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._root_p_prev[robot_name] = self._root_p[robot_name].clone()
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = self._root_p[robot_name].clone()
            # root q (measured, previous, default)
            self._root_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().q.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._root_q_prev[robot_name] = self._root_q[robot_name].clone()
            self._root_q_default[robot_name] = self._root_q[robot_name].clone()

            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_q.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_q_prev[robot_name] = self._jnts_q[robot_name].clone()
            self._jnts_q_default[robot_name] = self._jnts_q[robot_name].clone()
            
            # root v (measured, default)
            self._root_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[0:3]).reshape(self._num_envs, -1).to(self._dtype)
            self._root_v_base_loc[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_prev[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_default[robot_name] = self._root_v[robot_name].clone()

            # root omega (measured, default)
            self._root_omega[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().twist.copy()[3:6]).reshape(self._num_envs, -1).to(self._dtype)
            self._root_omega_prev[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_base_loc[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_default[robot_name] = self._root_omega[robot_name].clone()

            # root a (measured,)
            self._root_a[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_a_base_loc[robot_name] = torch.full_like(self._root_a[robot_name], fill_value=0.0)
            self._root_alpha[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_alpha_base_loc[robot_name] = torch.full_like(self._root_alpha[robot_name], fill_value=0.0)

            # height grid sensor storage
            grid_sz = int(self._env_opts["height_sensor_pixels"])
            self._height_imgs[robot_name] = torch.zeros((self._num_envs, grid_sz, grid_sz),
                                                        dtype=self._dtype,
                                                        device=self._device)

            # joints v (measured, default)
            self._jnts_v[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_v.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_v_default[robot_name] = self._jnts_v[robot_name].clone()
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.from_numpy(self._xmj_adapter.xmj_env().jnts_eff.copy()).reshape(self._num_envs, -1).to(self._dtype)
            self._jnts_eff_default[robot_name] = self._jnts_eff[robot_name].clone()

            self._root_pos_offsets[robot_name] = torch.zeros((self._num_envs, 3), 
                                device=self._device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self._num_envs, 4), 
                                device=self._device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion

            # self._update_root_offsets(robot_name)

            self._p_ref_reset[robot_name] = self._jnts_q[robot_name].clone() # last p ref for set to jnt imp control

    def current_tstep(self):
        return self._xmj_adapter.xmj_env().step_counter
    
    def world_time(self, robot_name: str):
        return self._xmj_adapter.getEnvTimeFromReset()
    
    def physics_dt(self):
        return self._xmj_adapter.xmj_env().physics_dt
    
    def rendering_dt(self):
        return self._env_opts["rendering_dt"]
        # return self._xmj_adapter.xmj_env().physics_dt
    
    def set_physics_dt(self, physics_dt:float):
        raise NotImplementedError()
    
    def set_rendering_dt(self, rendering_dt:float):
        raise NotImplementedError()
    
    def _robot_jnt_names(self, robot_name: str):
        return self._xmj_adapter.jnt_names()

    def _prepare_world_xml(self):
        """Optionally augment the base world.xml with procedural step-up tiles and return the path to use."""
        world_dir = self._env_opts.get("xmj_files_dir", None)
        if world_dir is None:
            Journal.log(self.__class__.__name__,
                "_prepare_world_xml",
                "xmj_files_dir not provided: using default world.xml.",
                LogType.WARN,
                throw_when_excep = False)
            return None

        base_world_path = os.path.join(world_dir, "world.xml")
        if not self._env_opts.get("generate_stepup_terrain", False):
            return base_world_path

        if not os.path.isfile(base_world_path):
            Journal.log(self.__class__.__name__,
                "_prepare_world_xml",
                f"world.xml not found at {base_world_path}: cannot generate procedural terrain.",
                LogType.WARN,
                throw_when_excep = False)
            return base_world_path

        try:
            tree = ET.parse(base_world_path)
            root = tree.getroot()
        except Exception as exc:
            Journal.log(self.__class__.__name__,
                "_prepare_world_xml",
                f"Failed parsing {base_world_path}: {exc}",
                LogType.WARN,
                throw_when_excep = False)
            return base_world_path

        worldbody = root.find("worldbody")
        if worldbody is None:
            Journal.log(self.__class__.__name__,
                "_prepare_world_xml",
                f"No <worldbody> found in {base_world_path}: cannot inject procedural terrain.",
                LogType.WARN,
                throw_when_excep = False)
            return base_world_path

        step_boxes = self._generate_stepup_boxes()
        if len(step_boxes) == 0:
            Journal.log(self.__class__.__name__,
                "_prepare_world_xml",
                "Procedural step-up generation returned no boxes: using base world.xml.",
                LogType.WARN,
                throw_when_excep = False)
            return base_world_path

        # remove previous autogenerated body if present
        for child in list(worldbody):
            if child.tag == "body" and child.get("name") == "auto_stepup_prim":
                worldbody.remove(child)
            # drop floor plane when generating stepup terrain to avoid double ground
            if child.tag == "geom" and child.get("name") == "floor":
                worldbody.remove(child)

        step_body = ET.SubElement(worldbody, "body", {"name": "auto_stepup_prim", "pos": "0 0 0"})
        for idx, box in enumerate(step_boxes):
            size = box["size"]
            pos = box["pos"]
            name = box.get("name", f"stepup_{idx}")
            geom_attrib = {
                "name": name,
                "type": "box",
                "size": " ".join([f"{v:.5f}" for v in size]),
                "pos": " ".join([f"{v:.5f}" for v in pos]),
                "quat": "1 0 0 0",
                "material": "groundplane",
                "group": "2",
                "contype": "1",
                "conaffinity": "1"
            }
            ET.SubElement(step_body, "geom", geom_attrib)

        out_path = os.path.join(world_dir, "world_autogen.xml")
        tree.write(out_path, encoding="utf-8", xml_declaration=True)

        Journal.log(self.__class__.__name__,
            "_prepare_world_xml",
            f"Generated procedural step-up terrain -> {out_path}",
            LogType.STAT,
            throw_when_excep = False)

        return out_path

    def _generate_stepup_boxes(self):
        """Create step-up style boxes matching Isaac stepup_prim terrain generation."""
        opts = self._env_opts
        terrain_size = float(opts.get("stepup_terrain_size", 100.0))
        stairs_ratio = float(opts.get("stepup_stairs_ratio", 0.99))
        platform_size = float(opts.get("stepup_platform_size", 50.0))
        step_height_lb = float(opts.get("stepup_step_height_lb", 0.08))
        step_height_ub = float(opts.get("stepup_step_height_ub", step_height_lb))
        min_step_width = opts.get("stepup_step_width_lb", None)
        max_step_width = opts.get("stepup_step_width_ub", None)
        n_steps = max(1, int(opts.get("stepup_n_steps", 25)))
        area_factor = float(opts.get("stepup_area_factor", 0.7))
        wall_height = float(opts.get("stepup_wall_height", 2.0))
        res_low = float(opts.get("stepup_res_low", 0.1))
        res_high = float(opts.get("stepup_res_high", 0.03))
        pos = np.array(opts.get("stepup_position", [0.0, 0.0, 0.0]), dtype=float)
        seed = opts.get("stepup_seed", None)
        random_n_steps = bool(opts.get("stepup_random_n_steps", False))

        step_height_ub = max(step_height_lb, step_height_ub)
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        use_high_res = (stairs_ratio > 0.0) or (n_steps > 1)
        horizontal_scale = res_high if use_high_res else res_low
        area_factor = max(min(area_factor, 0.9999), 1e-3)

        boxes = []

        ground_thickness = 0.1
        base_size = np.array([terrain_size / 2.0, terrain_size / 2.0, ground_thickness / 2.0])
        base_pos = pos.copy()
        boxes.append({"name": "stepup_base", "size": base_size, "pos": base_pos})

        terrain_low_corner = pos + np.array([-terrain_size / 2.0, -terrain_size / 2.0, 0.0])
        ground_top = base_pos[2] + base_size[2]

        n_tiles_x = max(1, int(math.ceil(terrain_size / platform_size)))
        n_tiles_y = max(1, int(math.ceil(terrain_size / platform_size)))

        use_step_width = (min_step_width is not None and max_step_width is not None)
        if use_step_width:
            min_step_width = max(0.0, float(min_step_width))
            max_step_width = max(min_step_width, float(max_step_width))

        for ix in range(n_tiles_x):
            for iy in range(n_tiles_y):
                start_x_m = ix * platform_size
                end_x_m = min(terrain_size, (ix + 1) * platform_size)
                start_y_m = iy * platform_size
                end_y_m = min(terrain_size, (iy + 1) * platform_size)

                size_x = end_x_m - start_x_m
                size_y = end_y_m - start_y_m
                if size_x <= 0.0 or size_y <= 0.0:
                    continue

                if rng.random() >= stairs_ratio:
                    continue

                tile_center_x = terrain_low_corner[0] + start_x_m + 0.5 * size_x
                tile_center_y = terrain_low_corner[1] + start_y_m + 0.5 * size_y

                if n_steps <= 1:
                    step_height = float(rng.uniform(step_height_lb, step_height_ub))
                    center_z = ground_top + 0.5 * step_height
                    boxes.append({
                        "name": f"stepup_tile_{ix}_{iy}",
                        "size": np.array([size_x / 2.0, size_y / 2.0, step_height / 2.0]),
                        "pos": np.array([tile_center_x, tile_center_y, center_z])
                    })
                    continue

                shrink_factor = math.sqrt(area_factor)
                steps_for_tile = int(rng.integers(1, n_steps + 1)) if random_n_steps else n_steps

                accumulated_height = 0.0
                curr_start_x = start_x_m
                curr_start_y = start_y_m
                curr_size_x = size_x
                curr_size_y = size_y

                for level in range(steps_for_tile):
                    if use_step_width:
                        stair_w = float(rng.uniform(min_step_width, max_step_width))
                        level_size_x = curr_size_x - 2.0 * stair_w
                        level_size_y = curr_size_y - 2.0 * stair_w
                    else:
                        level_size_x = size_x * (shrink_factor ** level)
                        level_size_y = size_y * (shrink_factor ** level)

                    if level_size_x <= horizontal_scale or level_size_y <= horizontal_scale:
                        break

                    if use_step_width:
                        level_start_x = curr_start_x + stair_w
                        level_start_y = curr_start_y + stair_w
                    else:
                        level_start_x = start_x_m + 0.5 * (size_x - level_size_x)
                        level_start_y = start_y_m + 0.5 * (size_y - level_size_y)

                    level_height = float(rng.uniform(step_height_lb, step_height_ub))
                    accumulated_height += level_height
                    level_center_x = terrain_low_corner[0] + level_start_x + 0.5 * level_size_x
                    level_center_y = terrain_low_corner[1] + level_start_y + 0.5 * level_size_y
                    level_center_z = ground_top + (accumulated_height - 0.5 * level_height)

                    boxes.append({
                        "name": f"stepup_tile_{ix}_{iy}_lvl{level}",
                        "size": np.array([level_size_x / 2.0, level_size_y / 2.0, level_height / 2.0]),
                        "pos": np.array([level_center_x, level_center_y, level_center_z])
                    })

                    if use_step_width:
                        curr_start_x = level_start_x
                        curr_start_y = level_start_y
                        curr_size_x = level_size_x
                        curr_size_y = level_size_y

        if wall_height > 0.0:
            wall_thickness = 0.1
            wall_z = ground_top + 0.5 * wall_height
            half_size = terrain_size / 2.0
            boxes.append({
                "name": "stepup_wall_xp",
                "size": np.array([wall_thickness / 2.0, half_size, wall_height / 2.0]),
                "pos": np.array([pos[0] + half_size + wall_thickness / 2.0, pos[1], wall_z])
            })
            boxes.append({
                "name": "stepup_wall_xm",
                "size": np.array([wall_thickness / 2.0, half_size, wall_height / 2.0]),
                "pos": np.array([pos[0] - half_size - wall_thickness / 2.0, pos[1], wall_z])
            })
            boxes.append({
                "name": "stepup_wall_yp",
                "size": np.array([half_size, wall_thickness / 2.0, wall_height / 2.0]),
                "pos": np.array([pos[0], pos[1] + half_size + wall_thickness / 2.0, wall_z])
            })
            boxes.append({
                "name": "stepup_wall_ym",
                "size": np.array([half_size, wall_thickness / 2.0, wall_height / 2.0]),
                "pos": np.array([pos[0], pos[1] - half_size - wall_thickness / 2.0, wall_z])
            })

        return boxes

    def _build_static_heightmap_from_world(self, world_path: str = None):
        """Parse world.xml and build a static heightfield using box geoms only."""
        if world_path is None:
            world_dir = self._env_opts.get("xmj_files_dir", None)
            if world_dir is None:
                Journal.log(self.__class__.__name__,
                    "_build_static_heightmap_from_world",
                    "xmj_files_dir not provided: heightmap sensor will be flat.",
                    LogType.WARN,
                    throw_when_excep = False)
                return None
            world_path = os.path.join(world_dir, "world.xml")

        if not os.path.isfile(world_path):
            Journal.log(self.__class__.__name__,
                "_build_static_heightmap_from_world",
                f"world.xml not found at {world_path}: heightmap sensor will be flat.",
                LogType.WARN,
                throw_when_excep = False)
            return None

        try:
            tree = ET.parse(world_path)
            root = tree.getroot()
        except Exception as exc:
            Journal.log(self.__class__.__name__,
                "_build_static_heightmap_from_world",
                f"Failed parsing {world_path}: {exc}",
                LogType.WARN,
                throw_when_excep = False)
            return None

        worldbody = root.find("worldbody")
        if worldbody is None:
            Journal.log(self.__class__.__name__,
                "_build_static_heightmap_from_world",
                f"No <worldbody> found in {world_path}: heightmap sensor will be flat.",
                LogType.WARN,
                throw_when_excep = False)
            return None

        boxes = []

        def parse_vec(attr_val, default):
            if attr_val is None:
                return np.array(default, dtype=float)
            vals = [float(x) for x in attr_val.strip().split()]
            if len(vals) == 0:
                return np.array(default, dtype=float)
            return np.array(vals, dtype=float)

        def parse_quat(attr_val):
            return parse_vec(attr_val, [1.0, 0.0, 0.0, 0.0])

        def quat_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=float)

        def quat_to_rot(q):
            w, x, y, z = q
            ww, xx, yy, zz = w*w, x*x, y*y, z*z
            wx, wy, wz = w*x, w*y, w*z
            xy, xz, yz = x*y, x*z, y*z
            return np.array([
                [ww + xx - yy - zz, 2*(xy - wz),     2*(xz + wy)],
                [2*(xy + wz),     ww - xx + yy - zz, 2*(yz - wx)],
                [2*(xz - wy),     2*(yz + wx),     ww - xx - yy + zz]
            ], dtype=float)

        def quat_rotate(q, v):
            # rotate vector v by quaternion q (w, x, y, z)
            qvec = q[1:]
            uv = np.cross(qvec, v)
            uuv = np.cross(qvec, uv)
            return v + 2.0 * (q[0] * uv + uuv)

        def collect_from_element(elem, parent_pos, parent_quat):
            local_pos = parse_vec(elem.get("pos"), [0.0, 0.0, 0.0])
            local_quat = parse_quat(elem.get("quat"))

            world_pos = parent_pos + quat_rotate(parent_quat, local_pos)
            world_quat = quat_multiply(parent_quat, local_quat)

            for geom in elem.findall("geom"):
                if geom.get("type", "").lower() != "box":
                    continue
                size = parse_vec(geom.get("size"), [0.0, 0.0, 0.0])
                geom_pos = parse_vec(geom.get("pos"), [0.0, 0.0, 0.0])
                geom_quat = parse_quat(geom.get("quat"))
                center = world_pos + quat_rotate(world_quat, geom_pos)
                quat_abs = quat_multiply(world_quat, geom_quat)
                boxes.append((center, size, quat_abs))

            for child in elem.findall("body"):
                collect_from_element(child, world_pos, world_quat)

        # top-level geoms directly inside worldbody (with world frame)
        collect_from_element(worldbody, np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float))

        if len(boxes) == 0:
            Journal.log(self.__class__.__name__,
                "_build_static_heightmap_from_world",
                "No box geometries found in world.xml: heightmap sensor will be flat.",
                LogType.WARN,
                throw_when_excep = False)
            return None

        # compute global bounds from all box corners
        min_x = float("+inf")
        max_x = float("-inf")
        min_y = float("+inf")
        max_y = float("-inf")
        boxes_bounds = []

        for center, size, quat_abs in boxes:
            rot = quat_to_rot(quat_abs)
            # all 8 corners
            corners = []
            for sx in (-size[0], size[0]):
                for sy in (-size[1], size[1]):
                    for sz in (-size[2], size[2]):
                        local = np.array([sx, sy, sz], dtype=float)
                        corners.append(center + rot @ local)
            corners = np.array(corners)
            cmin = corners.min(axis=0)
            cmax = corners.max(axis=0)
            min_x = min(min_x, cmin[0])
            max_x = max(max_x, cmax[0])
            min_y = min(min_y, cmin[1])
            max_y = max(max_y, cmax[1])
            boxes_bounds.append((cmin, cmax))

        margin = float(self._env_opts.get("height_map_margin", 0.0))
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        resolution = float(self._env_opts.get("height_map_resolution", 0.1))

        if not (resolution > 0.0):
            Journal.log(self.__class__.__name__,
                "_build_static_heightmap_from_world",
                f"Invalid height_map_resolution={resolution}, falling back to 0.1.",
                LogType.WARN,
                throw_when_excep = False)
            resolution = 0.1

        extent_x = max(max_x - min_x, resolution)
        extent_y = max(max_y - min_y, resolution)

        n_rows = int(math.ceil(extent_x / resolution)) + 1
        n_cols = int(math.ceil(extent_y / resolution)) + 1

        heightfield = np.zeros((n_rows, n_cols), dtype=np.float32)
        origin = np.array([min_x, min_y, 0.0], dtype=np.float64)

        for (cmin, cmax) in boxes_bounds:
            x_start = max(0, int(math.floor((cmin[0] - min_x) / resolution)))
            x_end = min(n_rows, int(math.ceil((cmax[0] - min_x) / resolution)) + 1)
            y_start = max(0, int(math.floor((cmin[1] - min_y) / resolution)))
            y_end = min(n_cols, int(math.ceil((cmax[1] - min_y) / resolution)) + 1)
            h_val = cmax[2]
            heightfield[x_start:x_end, y_start:y_end] = np.maximum(heightfield[x_start:x_end, y_start:y_end], h_val)

        class StaticHeightField:
            def __init__(self, hf, horiz_scale, pos, orient):
                self.heightfield_world = hf
                self._horizontal_scale = horiz_scale
                self._vertical_scale = 1.0
                self.position = pos
                self.orientation = orient

        height_data = StaticHeightField(heightfield, resolution, origin, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))

        info = f"Parsed static heightmap from {world_path}: grid {heightfield.shape} at res {resolution} m."
        Journal.log(self.__class__.__name__,
            "_build_static_heightmap_from_world",
            info,
            LogType.STAT,
            throw_when_excep = False)

        return height_data
