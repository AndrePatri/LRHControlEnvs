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
from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.world import World

import omni.kit

import numpy as np
import torch

from omni.importer.urdf import _urdf
from omni.isaac.core.utils.prims import move_prim
from omni.isaac.cloner import GridCloner
import omni.isaac.core.utils.prims as prim_utils

# from omni.isaac.sensor import ContactSensor

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.scenes.scene import Scene

from omni_robo_gym.utils.jnt_imp_cntrl import OmniJntImpCntrl
from omni_robo_gym.utils.homing import OmniRobotHomer
from omni_robo_gym.utils.contact_sensor import OmniContactSensors

from omni_robo_gym.utils.terrains import RlTerrains
from omni_robo_gym.utils.math_utils import quat_to_omega, quaternion_difference, rel_vel

from abc import abstractmethod
from typing import List, Dict

from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

class IsaacTask(BaseTask):

    def __init__(self, 
                name: str,
                integration_dt: float,
                robot_names: List[str],
                robot_urdf_paths: List[str],
                robot_srdf_paths: List[str],
                contact_prims: Dict[str, List] = None,
                contact_offsets: Dict[str, Dict[str, np.ndarray]] = None,
                sensor_radii: Dict[str, Dict[str, np.ndarray]] = None,
                num_envs = 1,
                device = "cuda", 
                cloning_offset: np.array = None,
                fix_base: List[bool] = None,
                self_collide: List[bool] = None,
                merge_fixed: List[bool] = None,
                replicate_physics: bool = True,
                solver_position_iteration_count: int = 4,
                solver_velocity_iteration_count: int = 1,
                solver_stabilization_thresh: float = 1e-5,
                offset=None, 
                env_spacing = 5.0, 
                spawning_radius = 1.0,
                use_flat_ground = True,
                default_jnt_stiffness = 300.0,
                default_jnt_damping = 20.0,
                default_wheel_stiffness = 0.0,
                default_wheel_damping = 10.0,
                override_art_controller = False,
                dtype = torch.float64,
                debug_enabled: bool = False,
                verbose = False,
                use_diff_velocities = False,
                dump_basepath: str = "/tmp") -> None:

        self.torch_dtype = dtype
        
        self._descr_dump_path = dump_basepath + "/" + f"{self.__class__.__name__}"

        self._debug_enabled = debug_enabled

        self._verbose = verbose

        self.use_diff_velocities = use_diff_velocities

        self.num_envs = num_envs

        self._override_art_controller = override_art_controller
        
        self._integration_dt = integration_dt # just used for contact reporting
        
        self.torch_device = torch.device(device) # defaults to "cuda" ("cpu" also valid)

        self.using_gpu = False
        if self.torch_device == torch.device("cuda"):
            self.using_gpu = True
                
        if fix_base is None:
            self._fix_base = [False] * len(self.robot_names)
        else:
            # check dimension consistency
            if len(fix_base) != len(self._robot_names):
                exception = "The provided fix_base list of boolean must match the length " + \
                    "of the provided robot package names"
                raise Exception(exception)
            self._fix_base = fix_base 
        
        if self_collide is None:
            self._self_collide = [False] * len(self.robot_names)
        else:
            # check dimension consistency
            if len(self_collide) != len(self._robot_names):
                exception = "The provided self_collide list of boolean must match the length " + \
                    "of the provided robot package names"
                raise Exception(exception)
            self._self_collide = self_collide 
        
        if merge_fixed is None:
            self._merge_fixed = [False] * len(self.robot_names)
        else:
            # check dimension consistency
            if len(merge_fixed) != len(self._robot_names):
                exception = "The provided merge_fixed list of boolean must match the length " + \
                    "of the provided robot package names"
                raise Exception(exception)
            
            self._merge_fixed = merge_fixed 

        self._urdf_paths = {}
        self._srdf_paths = {}
        self._robots_art_views = {}
        self._robots_articulations = {}
        self._robots_geom_prim_views = {}
        
        self._solver_position_iteration_count = solver_position_iteration_count # solver position iteration count
        # -> higher number makes simulation more accurate
        self._solver_velocity_iteration_count = solver_velocity_iteration_count
        self._solver_stabilization_thresh = solver_stabilization_thresh # threshold for kin. energy below which an articulatiion
        # "goes to sleep", i.e. it's not simulated anymore until some action wakes him up
        # potentially, each robot could have its own setting for the solver (not supported yet)
        self._solver_position_iteration_counts = {}
        self._solver_velocity_iteration_counts = {}
        self._solver_stabilization_threshs = {}

        self.robot_bodynames =  {}
        self.robot_n_links =  {}
        self.robot_n_dofs =  {}
        self.robot_dof_names =  {}

        self._root_p = {}
        self._root_q = {}
        self._jnts_q = {} 
        self._root_p_prev = {} # used for num differentiation
        self._root_q_prev = {} # used for num differentiation
        self._jnts_q_prev = {} # used for num differentiation
        self._root_p_default = {} 
        self._root_q_default = {}
        self._jnts_q_default = {}

        self._root_v = {}
        self._root_v_default = {}
        self._root_omega = {}
        self._root_omega_default = {}
        self._jnts_v = {}
        self._jnts_v_default = {}
        self._jnts_eff = {}
        self._jnts_eff_default = {}

        self._root_pos_offsets = {} 
        self._root_q_offsets = {} 

        self.distr_offset = {} # decribed how robots within each env are distributed
 
        self.jnt_imp_controllers = {}

        self.homers = {} 
        
        # default jnt impedance settings
        self.default_jnt_stiffness = default_jnt_stiffness
        self.default_jnt_damping = default_jnt_damping
        self.default_wheel_stiffness = default_wheel_stiffness
        self.default_wheel_damping = default_wheel_damping
        
        self.use_flat_ground = use_flat_ground
        
        self.spawning_radius = spawning_radius # [m] -> default distance between roots of robots in a single 
        # environment 
        self._calc_robot_distrib() # computes the offsets of robots withing each env.
        
        self._env_ns = "/World/envs"
        self._env_spacing = env_spacing # [m]
        self._template_env_ns = self._env_ns + "/env_0"

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self._env_ns)
        prim_utils.define_prim(self._template_env_ns)
        self._envs_prim_paths = self._cloner.generate_paths(self._env_ns + "/env", 
                                                self.num_envs)
        self._cloning_offset = cloning_offset
        if self._cloning_offset is None:
            self._cloning_offset = np.array([[0, 0, 0]] * self.num_envs)

        self._replicate_physics = replicate_physics

        self._world_initialized = False

        self._ground_plane_prim_path = "/World/terrain"

        self._world = None
        self._world_scene = None
        self._world_physics_context = None

        self.omni_contact_sensors = {}
        for robot_name in self.robot_names:
            self.omni_contact_sensors[robot_name]=None
        self.contact_prims = contact_prims
        if self.contact_prims is not None:
            for robot_name in contact_prims:
                if not (self.contact_prims[robot_name] is None):
                    self.omni_contact_sensors[robot_name] = OmniContactSensors(
                                        name = robot_name, 
                                        n_envs = self.num_envs, 
                                        contact_prims = contact_prims, 
                                        contact_offsets = contact_offsets, 
                                        sensor_radii = sensor_radii,
                                        device = self.torch_device, 
                                        dtype = self.torch_dtype,
                                        enable_debug=self._debug_enabled)

        # trigger __init__ of parent class
        BaseTask.__init__(self,
                        name=name, 
                        offset=offset)

        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class

    def update_jnt_imp_control_gains(self, 
                    robot_name: str,
                    jnt_stiffness: float, 
                    jnt_damping: float, 
                    wheel_stiffness: float, 
                    wheel_damping: float,
                    env_indxs: torch.Tensor = None):

        # updates joint imp. controller with new impedance values
        
        if self._debug_enabled:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "update_jnt_imp_control_gains",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)
                if self.using_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                            "update_jnt_imp_control_gains",
                            f"updating joint impedances " + for_robots,
                            LogType.STAT,
                            throw_when_excep = True)
        # set jnt imp gains for the whole robot
        if env_indxs is None:           
            gains_pos = torch.full((self.num_envs, \
                                    self.jnt_imp_controllers[robot_name].n_dofs), 
                        jnt_stiffness, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
            gains_vel = torch.full((self.num_envs, \
                                    self.jnt_imp_controllers[robot_name].n_dofs), 
                        jnt_damping, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
        else:
            gains_pos = torch.full((env_indxs.shape[0], \
                                    self.jnt_imp_controllers[robot_name].n_dofs), 
                        jnt_stiffness, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
            gains_vel = torch.full((env_indxs.shape[0], \
                                    self.jnt_imp_controllers[robot_name].n_dofs), 
                        jnt_damping, 
                        device = self.torch_device, 
                        dtype=self.torch_dtype)
        self.jnt_imp_controllers[robot_name].set_gains(
                pos_gains = gains_pos,
                vel_gains = gains_vel,
                robot_indxs = env_indxs)
        
        # in case of wheels
        wheels_indxs = self.jnt_imp_controllers[robot_name].get_jnt_idxs_matching(
                                name_pattern="wheel")
        if wheels_indxs is not None:
            if env_indxs is None:           
                # wheels are velocity-controlled
                wheels_pos_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                            wheel_stiffness, 
                                            device = self.torch_device, 
                                            dtype=self.torch_dtype)
                wheels_vel_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                            wheel_damping, 
                                            device = self.torch_device, 
                                            dtype=self.torch_dtype)
            else:
                # wheels are velocity-controlled
                wheels_pos_gains = torch.full((env_indxs.shape[0], len(wheels_indxs)), 
                                            wheel_stiffness, 
                                            device = self.torch_device, 
                                            dtype=self.torch_dtype)
                
                wheels_vel_gains = torch.full((env_indxs.shape[0], len(wheels_indxs)), 
                                            wheel_damping, 
                                            device = self.torch_device, 
                                            dtype=self.torch_dtype)
            self.jnt_imp_controllers[robot_name].set_gains(
                    pos_gains = wheels_pos_gains,
                    vel_gains = wheels_vel_gains,
                    jnt_indxs=wheels_indxs,
                    robot_indxs = env_indxs)
        
    def update_root_offsets(self, 
                    robot_name: str,
                    env_indxs: torch.Tensor = None):
        
        if self._debug_enabled:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):                
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "update_root_offsets",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)    
                if self.using_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "update_root_offsets",
                    f"updating root offsets " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # only planar position used
        if env_indxs is None:
            self._root_pos_offsets[robot_name][:, 0:2]  = self._root_p[robot_name][:, 0:2]
            self._root_q_offsets[robot_name][:, :]  = self._root_q[robot_name]
        else:
            self._root_pos_offsets[robot_name][env_indxs, 0:2]  = self._root_p[robot_name][env_indxs, 0:2]
            self._root_q_offsets[robot_name][env_indxs, :]  = self._root_q[robot_name][env_indxs, :]

    def synch_default_root_states(self,
            robot_name: str = None,
            env_indxs: torch.Tensor = None):

        if self._debug_enabled:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    msg = "Provided env_indxs should be a torch tensor of indexes!"
                    Journal.log(self.__class__.__name__,
                        "synch_default_root_states",
                        msg,
                        LogType.EXCEP,
                        throw_when_excep = True)  
                if self.using_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
            if self._verbose:
                Journal.log(self.__class__.__name__,
                            "synch_default_root_states",
                            f"updating default root states " + for_robots,
                            LogType.STAT,
                            throw_when_excep = True)

        if env_indxs is None:
            self._root_p_default[robot_name][:, :] = self._root_p[robot_name]
            self._root_q_default[robot_name][:, :] = self._root_q[robot_name]
        else:
            self._root_p_default[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :]
            self._root_q_default[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]

    def post_initialization_steps(self):
        
        print("Performing post-initialization steps")

        self._world_initialized = True # used by other methods which nees to run
        # only when the world was initialized

        # populates robot info fields
        self._fill_robot_info_from_world() 

        # initializes homing managers
        self._init_homing_managers() 

        # initializes robot state data
        self._init_robots_state()
        
        # default robot state
        self._set_robots_default_jnt_config()
        self._set_robots_root_default_config()

        # initializes joint impedance controllers
        self._init_jnt_imp_control() 

        # update solver options 
        self._update_art_solver_options() 

        self.reset()

        self._custom_post_init()

        self._get_solver_info() # get again solver option before printing everything

        self._print_envs_info() # debug prints
    
    def apply_collision_filters(self, 
                                physicscene_path: str, 
                                coll_root_path: str):

        self._cloner.filter_collisions(physicsscene_path = physicscene_path,
                                collision_root_path = coll_root_path, 
                                prim_paths=self._envs_prim_paths, 
                                global_paths=[self._ground_plane_prim_path] # can collide with these prims
                    )

    def reset_jnt_imp_control(self, 
                robot_name: str,
                env_indxs: torch.Tensor = None):
        
        if self._debug_enabled:
            for_robots = ""
            if env_indxs is not None:
                if not isinstance(env_indxs, torch.Tensor):
                    Journal.log(self.__class__.__name__,
                        "reset_jnt_imp_control",
                        "Provided env_indxs should be a torch tensor of indexes!",
                        LogType.EXCEP,
                        throw_when_excep = True)
                if self.using_gpu:
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
                for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs)
                                
            if self._verbose:
                Journal.log(self.__class__.__name__,
                    "reset_jnt_imp_control",
                    f"resetting joint impedances " + for_robots,
                    LogType.STAT,
                    throw_when_excep = True)

        # resets all internal data, refs to defaults
        self.jnt_imp_controllers[robot_name].reset(robot_indxs = env_indxs)

        # restore current state
        if env_indxs is None:
            self.jnt_imp_controllers[robot_name].update_state(pos = self._jnts_q[robot_name][:, :], 
                vel = self._jnts_v[robot_name][:, :],
                eff = None,
                robot_indxs = None)
        else:
            self.jnt_imp_controllers[robot_name].update_state(pos = self._jnts_q[robot_name][env_indxs, :], 
                vel = self._jnts_v[robot_name][env_indxs, :],
                eff = None,
                robot_indxs = env_indxs)
        
        # restore default gains
        self.update_jnt_imp_control_gains(robot_name = robot_name,
                                jnt_stiffness = self.default_jnt_stiffness,
                                jnt_damping = self.default_jnt_damping,
                                wheel_stiffness = self.default_wheel_stiffness, 
                                wheel_damping = self.default_wheel_damping,
                                env_indxs = env_indxs)
        
        #restore jnt imp refs to homing            
        if env_indxs is None:                               
            self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing()[:, :],
                                                    robot_indxs = None)
        else:
            self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing()[env_indxs, :],
                                                            robot_indxs = env_indxs)

        # actually applies reset commands to the articulation
        # self.jnt_imp_controllers[robot_name].apply_cmds()          

    def set_world(self,
                world: World):
        
        if not isinstance(world, World):
            Journal.log(self.__class__.__name__,
                    "configure_scene",
                    "world should be an instance of omni.isaac.core.world.World!",
                    LogType.EXCEP,
                    throw_when_excep = True)
        
        self._world = world
        self._world_scene = self._world.scene
        self._world_physics_context = self._world.get_physics_context()

    def set_up_scene(self, 
                    scene: Scene):
        
        super().set_up_scene(scene)

    def configure_scene(self) -> None:

        # this is called automatically by the environment BEFORE
        # initializing the simulation

        if self._world is None:
            Journal.log(self.__class__.__name__,
                "configure_scene",
                "Did you call the set_world() method??",
                LogType.EXCEP,
                throw_when_excep = True)

        if not self.scene_setup_completed:
            for i in range(len(self.robot_names)):
                robot_name = self.robot_names[i]
                robot_pkg_name = self.robot_pkg_names[i]
                pkg_prefix_path = self.robot_pkg_prefix_path[i]
                fix_base = self._fix_base[i]
                self_collide = self._self_collide[i]
                merge_fixed = self._merge_fixed[i]
                self._generate_rob_descriptions(robot_name=robot_name, 
                                        robot_pkg_name=robot_pkg_name,
                                        pkg_prefix_path=pkg_prefix_path)
                self._import_urdf(robot_name, 
                                fix_base=fix_base, 
                                self_collide=self_collide, 
                                merge_fixed=merge_fixed)
            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "cloning environments...",
                        LogType.STAT,
                        throw_when_excep = True)
            self._cloner.clone(
                source_prim_path=self._template_env_ns,
                prim_paths=self._envs_prim_paths,
                replicate_physics=self._replicate_physics,
                position_offsets = self._cloning_offset
            ) # we can clone the environment in which all the robos are
            Journal.log(self.__class__.__name__,
                        "set_up_scene",
                        "finishing scene setup...",
                        LogType.STAT,
                        throw_when_excep = True)
            for i in range(len(self.robot_names)):
                robot_name = self.robot_names[i]
                self._robots_art_views[robot_name] = ArticulationView(name = robot_name + "ArtView",
                                                            prim_paths_expr = self._env_ns + "/env_.*"+ "/" + robot_name + "/base_link", 
                                                            reset_xform_properties=False)
                self._robots_articulations[robot_name] = self._world_scene.add(self._robots_art_views[robot_name])
                # self._robots_geom_prim_views[robot_name] = GeometryPrimView(name = robot_name + "GeomView",
                #                                                 prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name,
                #                                                 # prepare_contact_sensors = True
                #                                             )
                # self._robots_geom_prim_views[robot_name].apply_collision_apis() # to be able to apply contact sensors
            
            if self.use_flat_ground:
                self._world_scene.add_default_ground_plane(z_position=0, 
                            name="terrain", 
                            prim_path= self._ground_plane_prim_path, 
                            static_friction=1.5, 
                            dynamic_friction=1.5, 
                            restitution=0.0)
            else:
                self.terrains = RlTerrains(get_current_stage())
                self.terrains.get_obstacles_terrain(terrain_size=40, 
                                            num_obs=100, 
                                            max_height=0.4, 
                                            min_size=0.5,
                                            max_size=5.0)
            # delete_prim(self._ground_plane_prim_path + "/SphereLight") # we remove the default spherical light
            
            # set default camera viewport position and target
            self._set_initial_camera_params()
            self.apply_collision_filters(self._world_physics_context.prim_path, 
                                "/World/collisions")
            # init contact sensors
            self._init_contact_sensors() # IMPORTANT: this has to be called
            # after calling the clone() method and initializing articulation views!!! 
            self._world.reset() # reset world to make art views available
            self.post_initialization_steps()

            self.scene_setup_completed = True
    
    def post_reset(self):
        pass

    def reset(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None,
            randomize: bool = False):

        # we first reset all target articulations to their default state
        rob_names = robot_names if (robot_names is not None) else self.robot_names
        # resets the state of target robot and env to the defaults
        self.reset_state(env_indxs=env_indxs, 
                    robot_names=rob_names,
                    randomize=randomize)
        # and jnt imp. controllers
        for i in range(len(rob_names)):
            self.reset_jnt_imp_control(robot_name=rob_names[i],
                                env_indxs=env_indxs)

    def _randomize_yaw(self,
            robot_name: str,
            env_indxs: torch.Tensor = None):

        root_q_default = self._root_q_default[robot_name]

        if env_indxs is None:
            env_indxs = torch.arange(root_q_default.shape[0])

        num_indices = env_indxs.shape[0]
        yaw_angles = torch.rand((num_indices,), 
                        device=root_q_default.device) * 2 * torch.pi  # uniformly distributed random angles
        
        # Compute cos and sin once
        cos_half = torch.cos(yaw_angles / 2)

        root_q_default[env_indxs, :] = torch.stack((cos_half, 
                                torch.zeros_like(cos_half),
                                torch.zeros_like(cos_half), 
                                torch.sin(yaw_angles / 2)), dim=1).reshape(num_indices, 4)
        
    def reset_state(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None,
            randomize: bool = False):

        rob_names = robot_names if (robot_names is not None) else self.robot_names
        if env_indxs is not None:
            if self._debug_enabled:
                if self.using_gpu:
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
                    self.randomize_yaw(robot_name=robot_name,env_indxs=env_indxs)

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
        self.get_states(env_indxs=env_indxs, 
                        robot_names=rob_names)

    def deactivate(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None):
        
        # deactivate jnt imp controllers for given robots and envs (makes the robot fall)
        rob_names = robot_names if (robot_names is not None) else self.robot_names
        for i in range(len(rob_names)):
            robot_name = rob_names[i]
            self.jnt_imp_controllers[robot_name].deactivate(robot_indxs = env_indxs)
                
    def close(self):
        pass

    def root_pos_offsets(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_pos_offsets[robot_name]
        else:
            return self._root_pos_offsets[robot_name][env_idxs, :]
    
    def root_q_offsets(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_q_offsets[robot_name]
        else:
            return self._root_q_offsets[robot_name][env_idxs, :]
    
    def root_p(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_p[robot_name]
        else:
            return self._root_p[robot_name][env_idxs, :]

    def root_p_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        rel_pos = torch.sub(self.root_p(robot_name=robot_name,
                                            env_idxs=env_idxs), 
                                self.root_pos_offsets(robot_name=robot_name, 
                                                        env_idxs=env_idxs))
        return rel_pos
    
    def root_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_q[robot_name]
        else:
            return self._root_q[robot_name][env_idxs, :]
    
    def root_q_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        rel_q = quaternion_difference(self.root_q_offsets(robot_name=robot_name, 
                                                        env_idxs=env_idxs), 
                            self.root_q(robot_name=robot_name,
                                            env_idxs=env_idxs))
        return rel_q
    
    def root_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_v[robot_name]
        else:
            return self._root_v[robot_name][env_idxs, :]

    def root_v_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        v_rel = rel_vel(offset_q0_q1=self.root_q_offsets(robot_name=robot_name, 
                                                        env_idxs=env_idxs),
                        v0=self.root_v(robot_name=robot_name, env_idxs=env_idxs))
        return v_rel
            
    def root_omega(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._root_omega[robot_name]
        else:
            return self._root_omega[robot_name][env_idxs, :]
    
    def root_omega_rel(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        omega_rel = rel_vel(offset_q0_q1=self.root_q_offsets(robot_name=robot_name, 
                                                        env_idxs=env_idxs),
                        v0=self.root_omega(robot_name=robot_name, env_idxs=env_idxs))
        return omega_rel

    def jnts_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):
        
        if env_idxs is None:
            return self._jnts_q[robot_name]
        else:
            return self._jnts_q[robot_name][env_idxs, :]

    def jnts_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            return self._jnts_v[robot_name]
        else:
            return self._jnts_v[robot_name][env_idxs, :]
    
    def jnts_eff(self,
            robot_name: str,
            env_idxs: torch.Tensor = None): # (measured) efforts

        if env_idxs is None:
            return self._jnts_eff[robot_name]
        else:
            return self._jnts_eff[robot_name][env_idxs, :]

    def integration_dt(self):
        return self._integration_dt

    @abstractmethod
    def _xrdf_cmds(self) -> Dict:

        # this has to be implemented by the user depending on the arguments
        # the xacro description of the robot takes. The output is a list 
        # of xacro commands.
        # Example implementation: 

        # def _xrdf_cmds():

        #   cmds = {}
        #   cmds{self.robot_names[0]} = []
        #   xrdf_cmd_vals = [True, True, True, False, False, True]

        #   legs = "true" if xrdf_cmd_vals[0] else "false"
        #   big_wheel = "true" if xrdf_cmd_vals[1] else "false"
        #   upper_body ="true" if xrdf_cmd_vals[2] else "false"
        #   velodyne = "true" if xrdf_cmd_vals[3] else "false"
        #   realsense = "true" if xrdf_cmd_vals[4] else "false"
        #   floating_joint = "true" if xrdf_cmd_vals[5] else "false" # horizon needs a floating joint

        #   cmds.append("legs:=" + legs)
        #   cmds.append("big_wheel:=" + big_wheel)
        #   cmds.append("upper_body:=" + upper_body)
        #   cmds.append("velodyne:=" + velodyne)
        #   cmds.append("realsense:=" + realsense)
        #   cmds.append("floating_joint:=" + floating_joint)

        #   return cmds
    
        pass

    @abstractmethod
    def pre_physics_step(self, 
                actions, 
                robot_name: str) -> None:
        
        # apply actions to simulated robot
        # to be overriden by child class depending
        # on specific needs
        pass

    def _generate_srdf(self, 
                robot_name: str, 
                robot_pkg_name: str,
                pkg_prefix_path: str):
        
        # we generate the URDF where the description package is located
        descr_path = pkg_prefix_path + f"/{robot_pkg_name}_srdf"
        srdf_path = descr_path + "/srdf"
        xacro_name = robot_pkg_name
        xacro_path = srdf_path + "/" + xacro_name + ".srdf.xacro"
        self._srdf_paths[robot_name] = self._descr_dump_path + "/" + robot_name + ".srdf"

        if self._xrdf_cmds() is not None:
            cmds = self._xrdf_cmds()[robot_name]
            if cmds is None:
                xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._srdf_paths[robot_name]]
            else:
                xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._srdf_paths[robot_name]]

        if self._xrdf_cmds() is None:
            xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._srdf_paths[robot_name]]

        import subprocess
        try:
            xacro_gen = subprocess.check_call(xacro_cmd)
        except:
            Journal.log(self.__class__.__name__,
                "_generate_urdf",
                "failed to generate " + robot_name + "\'S SRDF!!!",
                LogType.EXCEP,
                throw_when_excep = True)
        
    def _generate_urdf(self, 
                robot_name: str, 
                robot_pkg_name: str,
                pkg_prefix_path: str):

        # we generate the URDF where the description package is located
        descr_path = pkg_prefix_path + f"/{robot_pkg_name}_urdf"
        urdf_path = descr_path + "/urdf"
        xacro_name = robot_pkg_name
        xacro_path = urdf_path + "/" + xacro_name + ".urdf.xacro"
        self._urdf_paths[robot_name] = self._descr_dump_path + "/" + robot_name + ".urdf"
        
        if self._xrdf_cmds() is not None:
            cmds = self._xrdf_cmds()[robot_name]
            if cmds is None:
                xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._urdf_paths[robot_name]]
            else:
                xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._urdf_paths[robot_name]]
        if self._xrdf_cmds() is None:
            xacro_cmd = ["xacro"] + [xacro_path] + ["-o"] + [self._urdf_paths[robot_name]]

        import subprocess
        try:
            xacro_gen = subprocess.check_call(xacro_cmd)
            # we also generate an updated SRDF
        except:
            Journal.log(self.__class__.__name__,
                "_generate_urdf",
                "Failed to generate " + robot_name + "\'s URDF!!!",
                LogType.EXCEP,
                throw_when_excep = True)

    def _generate_rob_descriptions(self, 
                    robot_name: str, 
                    robot_pkg_name: str,
                    pkg_prefix_path: str):
        
        Journal.log(self.__class__.__name__,
                    "update_root_offsets",
                    "generating URDF for robot "+ f"{robot_name}, of type {robot_pkg_name}...",
                    LogType.STAT,
                    throw_when_excep = True)
        self._generate_urdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name,
                        pkg_prefix_path=pkg_prefix_path)
        Journal.log(self.__class__.__name__,
                    "update_root_offsets",
                    "generating SRDF for robot "+ f"{robot_name}, of type {robot_pkg_name}...",
                    LogType.STAT,
                    throw_when_excep = True)
        # we also generate SRDF files, which are useful for control
        self._generate_srdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name,
                        pkg_prefix_path=pkg_prefix_path)
        
    def _import_urdf(self, 
                robot_name: str,
                import_config: omni.importer.urdf._urdf.ImportConfig = _urdf.ImportConfig(), 
                fix_base = False, 
                self_collide = False, 
                merge_fixed = True):

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
        success, robot_prim_path_default = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self._urdf_paths[robot_name],
            import_config=import_config, 
        )
        robot_base_prim_path = self._template_env_ns + "/" + robot_name
        # moving default prim to base prim path for cloning
        move_prim(robot_prim_path_default, # from
                robot_base_prim_path) # to

        return success
    
    def _init_contact_sensors(self):
        for i in range(0, len(self.robot_names)):
            robot_name = self.robot_names[i]
            # creates base contact sensor (which is then cloned)
            if self.omni_contact_sensors[robot_name] is not None:
                self.omni_contact_sensors[robot_name].create_contact_sensors(
                                                        self._world, 
                                                        self._env_ns
                                                    )
                            
    def _init_robots_state(self):

        for i in range(0, len(self.robot_names)):

            robot_name = self.robot_names[i]
            pose = self._robots_art_views[robot_name].get_world_poses( 
                                clone = True) # tuple: (pos, quat)

            # root p (measured, previous, default)
            self._root_p[robot_name] = pose[0]  
            self._root_p_prev[robot_name] = torch.clone(pose[0])
            self._root_p_default[robot_name] = torch.clone(pose[0]) + self.distr_offset[robot_name]
            # root q (measured, previous, default)
            self._root_q[robot_name] = pose[1] # root orientation
            self._root_q_prev[robot_name] = torch.clone(pose[1])
            self._root_q_default[robot_name] = torch.clone(pose[1])
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            self._jnts_q_prev[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) 
            self._jnts_q_default[robot_name] = self.homers[robot_name].get_homing(clone=True)
            # root v (measured, default)
            self._root_v[robot_name] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocity
            self._root_v_default[robot_name] = torch.full((self._root_v[robot_name].shape[0], self._root_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self.torch_dtype, 
                                                        device=self.torch_device)
            # root omega (measured, default)
            self._root_omega[robot_name] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True) # root ang. velocity
            self._root_omega_default[robot_name] = torch.full((self._root_omega[robot_name].shape[0], self._root_omega[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self.torch_dtype, 
                                                        device=self.torch_device)
            # joints v (measured, default)
            self._jnts_v[robot_name] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True) # joint velocities
            self._jnts_v_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self.torch_dtype, 
                                                        device=self.torch_device)
            
            # joints efforts (measured, default)
            self._jnts_eff[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                0.0, 
                                                dtype=self.torch_dtype, 
                                                device=self.torch_device)
            self._jnts_eff_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                    0.0, 
                                                    dtype=self.torch_dtype, 
                                                    device=self.torch_device)
            self._root_pos_offsets[robot_name] = torch.zeros((self.num_envs, 3), 
                                device=self.torch_device) # reference position offses
            
            self._root_q_offsets[robot_name] = torch.zeros((self.num_envs, 4), 
                                device=self.torch_device)
            self._root_q_offsets[robot_name][:, 0] = 1.0 # init to valid identity quaternion
            self.update_root_offsets(robot_name)
            
    def _calc_robot_distrib(self):

        import math
        # we distribute robots in a single env. along the 
        # circumference of a circle of given radius
        n_robots = len(self.robot_names)
        offset_baseangle = 2 * math.pi / n_robots
        for i in range(n_robots):
            offset_angle = offset_baseangle * (i + 1) 
            robot_offset_wrt_center = torch.tensor([self.spawning_radius * math.cos(offset_angle), 
                                            self.spawning_radius * math.sin(offset_angle), 0], 
                    device=self.torch_device, 
                    dtype=self.torch_dtype)
            # list with n references to the original tensor
            tensor_list = [robot_offset_wrt_center] * self.num_envs
            self.distr_offset[self.robot_names[i]] = torch.stack(tensor_list, dim=0)

    def _get_robots_state(self, 
                env_indxs: torch.Tensor = None,
                robot_names: List[str] = None,
                dt: float = None, 
                reset: bool = False):
        
        rob_names = robot_names if (robot_names is not None) else self.robot_names
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
    
    def get_states(self,
                env_indxs: torch.Tensor = None,
                robot_names: List[str] = None):
        
        if self.use_diff_velocities:
            self._get_robots_state(dt = self.integration_dt(),
                            env_indxs = env_indxs,
                            robot_names = robot_names) # updates robot states
            # but velocities are obtained via num. differentiation
        else:
            self._get_robots_state(env_indxs = env_indxs,
                            robot_names = robot_names) # velocities directly from simulator (can 
            # introduce relevant artifacts, making them unrealistic)

    def _custom_post_init(self):
        # can be overridden by child class
        pass

    def _set_robots_default_jnt_config(self):
        # setting Isaac's internal defaults. Useful is resetting
        # whole scenes or views, but single env reset has to be implemented
        # manueally
        # we use the homing of the robots
        if (self._world_initialized):
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                homing = self.homers[robot_name].get_homing()
                self._robots_art_views[robot_name].set_joints_default_state(positions= homing, 
                                velocities = torch.zeros((homing.shape[0], homing.shape[1]), \
                                                    dtype=self.torch_dtype, device=self.torch_device), 
                                efforts = torch.zeros((homing.shape[0], homing.shape[1]), \
                                                    dtype=self.torch_dtype, device=self.torch_device))
        else:
            Journal.log(self.__class__.__name__,
                "_set_robots_default_jnt_config",
                "Before calling __set_robots_default_jnt_config(), you need to reset the World" + \
                            " at least once and call post_initialization_steps()",
                LogType.EXCEP,
                throw_when_excep = True)

    def _set_robots_root_default_config(self):
        if (self._world_initialized):
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                self._robots_art_views[robot_name].set_default_state(positions = self._root_p_default[robot_name], 
                            orientations = self._root_q_default[robot_name])
        else:
            Journal.log(self.__class__.__name__,
                "_generate_urdf",
                "Before calling _set_robots_root_default_config(), you need to reset the World" + \
                        " at least once and call post_initialization_steps()",
                LogType.EXCEP,
                throw_when_excep = True)

        return True
        
    def _get_solver_info(self):
        for i in range(0, len(self.robot_names)):
            robot_name = self.robot_names[i]
            self._solver_position_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_position_iteration_counts()
            self._solver_velocity_iteration_counts[robot_name] = self._robots_art_views[robot_name].get_solver_velocity_iteration_counts()
            self._solver_stabilization_threshs[robot_name] = self._robots_art_views[robot_name].get_stabilization_thresholds()
    
    def _update_art_solver_options(self):
        
        # sets new solver iteration options for specifc articulations
        self._get_solver_info() # gets current solver info for the articulations of the 
        # environments, so that dictionaries are filled properly
        
        if (self._world_initialized):
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                # increase by a factor
                self._solver_position_iteration_counts[robot_name] = torch.full((self.num_envs,), self._solver_position_iteration_count)
                self._solver_velocity_iteration_counts[robot_name] = torch.full((self.num_envs,), self._solver_velocity_iteration_count)
                self._solver_stabilization_threshs[robot_name] = torch.full((self.num_envs,), self._solver_stabilization_thresh)
                self._robots_art_views[robot_name].set_solver_position_iteration_counts(self._solver_position_iteration_counts[robot_name])
                self._robots_art_views[robot_name].set_solver_velocity_iteration_counts(self._solver_velocity_iteration_counts[robot_name])
                self._robots_art_views[robot_name].set_stabilization_thresholds(self._solver_stabilization_threshs[robot_name])
                self._get_solver_info() # gets again solver info for articulation, so that it's possible to debug if
                # the operation was successful
        else:
            Journal.log(self.__class__.__name__,
                "_set_robots_default_jnt_config",
                "Before calling update_art_solver_options(), you need to reset the World at least once!",
                LogType.EXCEP,
                throw_when_excep = True)                    
        
    def _print_envs_info(self):
        if (self._world_initialized):
            print("TASK INFO:")
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                task_info = f"[{robot_name}]" + "\n" + \
                    "bodies: " + str(self._robots_art_views[robot_name].body_names) + "\n" + \
                    "n. prims: " + str(self._robots_art_views[robot_name].count) + "\n" + \
                    "prims names: " + str(self._robots_art_views[robot_name].prim_paths) + "\n" + \
                    "n. bodies: " + str(self._robots_art_views[robot_name].num_bodies) + "\n" + \
                    "n. dofs: " + str(self._robots_art_views[robot_name].num_dof) + "\n" + \
                    "dof names: " + str(self._robots_art_views[robot_name].dof_names) + "\n" + \
                    "solver_position_iteration_counts: " + str(self._solver_position_iteration_counts[robot_name]) + "\n" + \
                    "solver_velocity_iteration_counts: " + str(self._solver_velocity_iteration_counts[robot_name]) + "\n" + \
                    "stabiliz. thresholds: " + str(self._solver_stabilization_threshs[robot_name])
                # print("dof limits: " + str(self._robots_art_views[robot_name].get_dof_limits()))
                # print("effort modes: " + str(self._robots_art_views[robot_name].get_effort_modes()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("dof max efforts: " + str(self._robots_art_views[robot_name].get_max_efforts()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("physics handle valid: " + str(self._robots_art_views[robot_name].is_physics_handle_valid())
                Journal.log(self.__class__.__name__,
                    "_print_envs_info",
                    task_info,
                    LogType.STAT,
                    throw_when_excep = True)
        else:
            Journal.log(self.__class__.__name__,
                        "_set_robots_default_jnt_config",
                        "Before calling __print_envs_info(), you need to reset the World at least once!",
                        LogType.EXCEP,
                        throw_when_excep = True)  

    def _fill_robot_info_from_world(self):

        if self._world_initialized:
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                self.robot_bodynames[robot_name] = self._robots_art_views[robot_name].body_names
                self.robot_n_links[robot_name] = self._robots_art_views[robot_name].num_bodies
                self.robot_n_dofs[robot_name] = self._robots_art_views[robot_name].num_dof
                self.robot_dof_names[robot_name] = self._robots_art_views[robot_name].dof_names
        else:
            Journal.log(self.__class__.__name__,
                "_fill_robot_info_from_world",
                "Before calling _fill_robot_info_from_world(), you need to reset the World at least once!",
                LogType.EXCEP,
                throw_when_excep = True)  

    def _init_homing_managers(self):
        
        if self._world_initialized:
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                self.homers[robot_name] = OmniRobotHomer(articulation=self._robots_art_views[robot_name], 
                                    srdf_path=self._srdf_paths[robot_name], 
                                    device=self.torch_device, 
                                    dtype=self.torch_dtype)
        else:
            exception = "you should reset the World at least once and call the " + \
                            "post_initialization_steps() method before initializing the " + \
                            "homing manager."
            Journal.log(self.__class__.__name__,
                "_init_homing_managers",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)  
    
    def _init_jnt_imp_control(self):
    
        if self._world_initialized:
            for i in range(0, len(self.robot_names)):
                robot_name = self.robot_names[i]
                # creates impedance controller
                self.jnt_imp_controllers[robot_name] = OmniJntImpCntrl(articulation=self._robots_art_views[robot_name],
                                            default_pgain = self.default_jnt_stiffness, # defaults
                                            default_vgain = self.default_jnt_damping,
                                            override_art_controller=self._override_art_controller,
                                            filter_dt = None, 
                                            filter_BW = 50,
                                            device= self.torch_device, 
                                            dtype=self.torch_dtype,
                                            enable_safety=True,
                                            enable_profiling=self._debug_enabled,
                                            urdf_path=self._urdf_paths[robot_name],
                                            debug_checks = self._debug_enabled)
                self.reset_jnt_imp_control(robot_name)
                
        else:
            exception = "you should reset the World at least once and call the " + \
                            "post_initialization_steps() method before initializing the " + \
                            "joint impedance controller."
            Journal.log(self.__class__.__name__,
                "_init_homing_managers",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")
