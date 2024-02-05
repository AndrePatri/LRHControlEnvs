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

import gymnasium
from gymnasium import spaces
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
from omni_robo_gym.utils.defs import Journal
from omni_robo_gym.utils.terrains import RlTerrains
from omni_robo_gym.utils.math_utils import quat_to_omega

from abc import abstractmethod
from typing import List, Dict

class CustomTask(BaseTask):

    def __init__(self, 
                name: str,
                integration_dt: float,
                robot_names: List[str],
                robot_pkg_names: List[str] = None, 
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
                debug_mode_jnt_imp = False) -> None:

        self.torch_dtype = dtype
        
        self.num_envs = num_envs

        self._override_art_controller = override_art_controller
        self._debug_mode_jnt_imp = debug_mode_jnt_imp

        self._integration_dt = integration_dt # just used for contact reporting
        
        self.torch_device = torch.device(device) # defaults to "cuda" ("cpu" also valid)

        self.using_gpu = False
        if self.torch_device == torch.device("cuda"):
            self.using_gpu = True
        
        self._journal = Journal()
        
        self.robot_names = robot_names # these are (potentially) custom names to 
        self.robot_pkg_names = robot_pkg_names # will be used to search for URDF and SRDF packages

        self.scene_setup_completed = False

        if self.robot_pkg_names is None:
            
            self.robot_pkg_names = self.robot_names # if not provided, robot_names are the same as robot_pkg_names
        
        else:
            
            # check dimension consistency
            if len(robot_names) != len(robot_pkg_names):

                exception = "The provided robot names list must match the length " + \
                    "of the provided robot package names"
                
                raise Exception(exception)
        
        if fix_base is None:

            self._fix_base = [False] * len(self.robot_names)
        
        else:

            # check dimension consistency
            if len(fix_base) != len(robot_pkg_names):

                exception = "The provided fix_base list of boolean must match the length " + \
                    "of the provided robot package names"
                
                raise Exception(exception)
            
            self._fix_base = fix_base 
        
        if self_collide is None:

            self._self_collide = [False] * len(self.robot_names)
        
        else:

            # check dimension consistency
            if len(self_collide) != len(robot_pkg_names):

                exception = "The provided self_collide list of boolean must match the length " + \
                    "of the provided robot package names"
                
                raise Exception(exception)
            
            self._self_collide = self_collide 
        
        if merge_fixed is None:

            self._merge_fixed = [False] * len(self.robot_names)
        
        else:

            # check dimension consistency
            if len(merge_fixed) != len(robot_pkg_names):

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

        self._jnts_eff_default = {}

        self._root_abs_offsets = {} 

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
        
        # if len(self._cloning_offset[:, 0]) != self.num_envs or \
        #     len(self._cloning_offset[0, :] != 3):
        
        #     warn = f"[{self.__class__.__name__}]" + \
        #                     f"[{self._journal.warning}]" + \
        #                     ": provided cloning offsets are not of the right shape." + \
        #                     " Resetting them to zero..."
        #     print(warn)

        #     self._cloning_offset = np.array([[0, 0, 0]] * self.num_envs)

        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1

        # a few class buffers to store RL-related states
        self.obs = torch.zeros((self.num_envs, self._num_observations))
        self.resets = torch.zeros((self.num_envs, 1))

        # set the action and observation space for RL
        self.action_space = spaces.Box(np.ones(self._num_actions) * -1.0, np.ones(self._num_actions) * 1.0)
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        )

        self._replicate_physics = replicate_physics

        self._world_initialized = False

        self._ground_plane_prim_path = "/World/terrain"

        self._world = None
        
        self.omni_contact_sensors = {}
        self.contact_prims = contact_prims
        for robot_name in contact_prims:
            
            self.omni_contact_sensors[robot_name] = OmniContactSensors(
                                name = robot_name, 
                                n_envs = self.num_envs, 
                                contact_prims = contact_prims, 
                                contact_offsets = contact_offsets, 
                                sensor_radii = sensor_radii,
                                device = self.torch_device, 
                                dtype = self.torch_dtype)

        # trigger __init__ of parent class
        BaseTask.__init__(self,
                        name=name, 
                        offset=offset)

        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class

    def update_root_offsets(self, 
                    robot_name: str,
                    env_indxs: torch.Tensor = None):
        
        for_robots = ""
        if env_indxs is not None:
            
            if not isinstance(env_indxs, torch.Tensor):
                
                msg = "Provided env_indxs should be a torch tensor of indexes!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]: " + msg)
                
            for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
        
        info = f"[{self.__class__.__name__}]" + f"[{self._journal.info}]: " + \
                        f"updating root offsets " + for_robots
        print(info)

        if env_indxs is None:

            self._root_abs_offsets[robot_name][:, 0:2]  = self._root_p[robot_name][:, 0:2]
        
        else:

            self._root_abs_offsets[robot_name][env_indxs, 0:2]  = self._root_p[robot_name][env_indxs, 0:2]

    def synch_default_root_states(self,
            robot_name: str = None,
            env_indxs: torch.Tensor = None):

        for_robots = ""
        if env_indxs is not None:
            
            if not isinstance(env_indxs, torch.Tensor):
                
                msg = "Provided env_indxs should be a torch tensor of indexes!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]: " + msg)
                
            for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
        
        info = f"[{self.__class__.__name__}]" + f"[{self._journal.info}]: " + \
                        f"updating default root states " + for_robots
        print(info)

        names = []
        
        if robot_name is None:

            names = self.robot_names
        
        else:

            names.append(robot_name)
            
        for i in range(0, len(names)):

            robot_name = names[i]

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

    def update_jnt_imp_control_gains(self, 
                    robot_name: str,
                    jnt_stiffness: float, 
                    jnt_damping: float, 
                    wheel_stiffness: float, 
                    wheel_damping: float,
                    env_indxs: torch.Tensor = None):

        # updates joint imp. controller with new impedance values
        
        for_robots = ""
        if env_indxs is not None:
            
            if not isinstance(env_indxs, torch.Tensor):
                
                msg = "Provided env_indxs should be a torch tensor of indexes!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]: " + msg)
                
            for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs.tolist())
        
        info = f"[{self.__class__.__name__}]" + f"[{self._journal.info}]: " + \
                        f"updating joint impedances " + for_robots
        print(info)
        
        wheels_indxs = self.jnt_imp_controllers[robot_name].get_jnt_idxs_matching(
                                name_pattern="wheel")
            
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

            # wheels are velocity-controlled
            wheels_pos_gains = torch.full((env_indxs.shape[0], len(wheels_indxs)), 
                                        wheel_stiffness, 
                                        device = self.torch_device, 
                                        dtype=self.torch_dtype)
            
            wheels_vel_gains = torch.full((env_indxs.shape[0], len(wheels_indxs)), 
                                        wheel_damping, 
                                        device = self.torch_device, 
                                        dtype=self.torch_dtype)
            
        success = self.jnt_imp_controllers[robot_name].set_gains(
                                    pos_gains = gains_pos,
                                    vel_gains = gains_vel,
                                    robot_indxs = env_indxs)
            
        if not all(success):
            
            warning = f"[{self.__class__.__name__}]" + f"[{self._journal.warning}]: " + \
            f"impedance controller could not set gains."

            print(warning)

        success_wheels = self.jnt_imp_controllers[robot_name].set_gains(
                                pos_gains = wheels_pos_gains,
                                vel_gains = wheels_vel_gains,
                                jnt_indxs=wheels_indxs,
                                robot_indxs = env_indxs)

        if not all(success_wheels):
            
            warning = f"[{self.__class__.__name__}]" + f"[{self._journal.warning}]: " + \
            f"impedance controller could not set wheel gains " + for_robots

            print(warning)
        
        info = f"[{self.__class__.__name__}]" + f"[{self._journal.info}]: " + \
            f"joint impedances updated " + for_robots
        
        print(info)
    
    def reset_jnt_imp_control(self, 
                robot_name: str,
                env_indxs: torch.Tensor = None):
        
        for_robots = ""
        if env_indxs is not None:
            
            if not isinstance(env_indxs, torch.Tensor):
                
                msg = "Provided env_indxs should be a torch tensor of indexes!"
            
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]: " + msg)
                
            for_robots = f"for robot {robot_name}, indexes: " + str(env_indxs)
        
        info = f"[{self.__class__.__name__}]" + f"[{self._journal.info}]: " + \
                        f"resetting joint impedances " + for_robots
        print(info)

        # resets all internal data, refs, 
        self.jnt_imp_controllers[robot_name].reset(robot_indxs = env_indxs)

        # we override internal default gains only for the wheels (which btw are usually
        # velocity controlled)

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
        
        self.update_jnt_imp_control_gains(robot_name = robot_name,
                                jnt_stiffness = self.default_jnt_stiffness,
                                jnt_damping = self.default_jnt_damping,
                                wheel_stiffness = self.default_wheel_stiffness, 
                                wheel_damping = self.default_wheel_damping,
                                env_indxs = env_indxs)
            
        try:
            
            if env_indxs is None:
                                                          
                self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing()[:, :],
                                                        robot_indxs = None)

            else:
                
                self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing()[env_indxs, :],
                                                                robot_indxs = env_indxs)
        
        except Exception:
            
            print(f"[{self.__class__.__name__}]" + f"[{self._journal.warning}]" +  f"[{self.init_imp_control.__name__}]" +\
            ": cannot set imp. controller reference to homing. Did you call the \"init_homing_managers\" method ?")

            pass      

        # actually applies reset commands to the articulation
        self.jnt_imp_controllers[robot_name].apply_cmds()          

    def set_world(self,
                world: World):
        
        self._world = world
        
    def set_up_scene(self, 
                    scene: Scene) -> None:

        # this is called automatically by the environment BEFORE
        # initializing the simulation

        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]
            robot_pkg_name = self.robot_pkg_names[i]

            fix_base = self._fix_base[i]
            self_collide = self._self_collide[i]
            merge_fixed = self._merge_fixed[i]

            self._generate_rob_descriptions(robot_name=robot_name, 
                                    robot_pkg_name=robot_pkg_name)

            self._import_urdf(robot_name, 
                            fix_base=fix_base, 
                            self_collide=self_collide, 
                            merge_fixed=merge_fixed)
        
        # init contact sensors
        self._init_contact_sensors() # IMPORTANT: this has to be called
        # before calling the clone() method!!! 
            
        print(f"[{self.__class__.__name__}]" + \
            f"[{self._journal.status}]" + \
            ": cloning environments...")

        self._cloner.clone(
            source_prim_path=self._template_env_ns,
            prim_paths=self._envs_prim_paths,
            replicate_physics=self._replicate_physics,
            position_offsets = self._cloning_offset
        ) # we can clone the environment in which all the robos are

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": finishing scene setup...")
        
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            self._robots_art_views[robot_name] = ArticulationView(name = robot_name + "ArtView",
                                                        prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name + "/base_link", 
                                                        reset_xform_properties=False)

            self._robots_articulations[robot_name] = scene.add(self._robots_art_views[robot_name])
            
            # self._robots_geom_prim_views[robot_name] = GeometryPrimView(name = robot_name + "GeomView",
            #                                                 prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name,
            #                                                 # prepare_contact_sensors = True
            #                                             )

            # self._robots_geom_prim_views[robot_name].apply_collision_apis() # to be able to apply contact sensors

        if self.use_flat_ground:

            scene.add_default_ground_plane(z_position=0, 
                        name="terrain", 
                        prim_path= self._ground_plane_prim_path, 
                        static_friction=0.5, 
                        dynamic_friction=0.5, 
                        restitution=0.8)
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
        
        self.scene_setup_completed = True

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": done")
    
    def post_reset(self):
        
        # post reset operations
            
        # for i in range(len(self.robot_names)):
            
        #     robot_name = self.robot_names[i]

        #     # resets articulations to their default state
        #     self._robots_art_views[robot_name].post_reset()
    
        pass

    def reset(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None):

        # we first reset all target articulations to their default state

        rob_names = robot_names if (robot_names is not None) else self.robot_names

        # resets the state of target robot and env to the defaults
        self.reset_state(env_indxs=env_indxs, 
                    robot_names=rob_names)
        
        # we then update the robots state (this should only be done
        # on target robots and envs)
        self._get_robots_state(dt = self._integration_dt, 
                        env_indxs=env_indxs, 
                        robot_names=rob_names,
                        reset = True)

        # and, based on that, we reset the jnt imp. controller
        for i in range(len(rob_names)):
            
            self.reset_jnt_imp_control(robot_name=rob_names[i],
                                env_indxs=env_indxs)

    def reset_state(self,
            env_indxs: torch.Tensor = None,
            robot_names: List[str] =None):

        rob_names = robot_names if (robot_names is not None) else self.robot_names

        if env_indxs is not None:

            for i in range(len(rob_names)):

                robot_name = rob_names[i]
        
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

    def get_observations(self):
        
        # retrieve data from the simulation

        pass

    def calculate_metrics(self) -> None:
        
        # compute any metric to be fed to the agent

        pass

    def is_done(self) -> None:
        
        pass

    def close(self):

        pass

    def root_offsets(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:

            return self._root_abs_offsets[robot_name]
        
        else:
            
            return self._root_abs_offsets[robot_name][env_idxs, :]
    
    def root_p(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:
            
            return self._root_p[robot_name]

        else:

            return self._root_p[robot_name][env_idxs, :]

    def root_q(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:

            return self._root_q[robot_name]
            
        else:
            
            return self._root_q[robot_name][env_idxs, :]

    def root_v(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:

            return self._root_v[robot_name]
            
        else:
            
            return self._root_v[robot_name][env_idxs, :]

    def root_omega(self,
            robot_name: str,
            env_idxs: torch.Tensor = None):

        if env_idxs is None:

            return self._root_omega[robot_name]
            
        else:

            return self._root_omega[robot_name][env_idxs, :]

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
                robot_pkg_name: str):
        
        # we generate the URDF where the description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        descr_path = rospackage.get_path(robot_pkg_name + "_srdf")
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

            raise Exception(f"[{self.__class__.__name__}]" 
                            + f"[{self._journal.exception}]" + 
                            ": failed to generate " + robot_name + "\'S SRDF!!!")
        
    def _generate_urdf(self, 
                robot_name: str, 
                robot_pkg_name: str):

        # we generate the URDF where the description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        descr_path = rospackage.get_path(robot_pkg_name + "_urdf")
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
            
            # we also generate an updated SRDF (used by controllers)

        except:

            raise Exception(f"[{self.__class__.__name__}]" + 
                            f"[{self._journal.exception}]" + 
                            ": failed to generate " + robot_name + "\'s URDF!!!")

    def _generate_rob_descriptions(self, 
                    robot_name: str, 
                    robot_pkg_name: str):
        
        self._descr_dump_path = "/tmp/" + f"{self.__class__.__name__}"
        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": generating URDF for robot "+ 
              f"{robot_name}, of type {robot_pkg_name}...")

        self._generate_urdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name)

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": generating SRDF for robot "+ 
              f"{robot_name}, of type {robot_pkg_name}...")

        # we also generate SRDF files, which are useful for control
        self._generate_srdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name)
        
        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": done")

    def _import_urdf(self, 
                robot_name: str,
                import_config: omni.importer.urdf._urdf.ImportConfig = _urdf.ImportConfig(), 
                fix_base = False, 
                self_collide = False, 
                merge_fixed = True):

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": importing robot URDF")

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

        print(f"[{self.__class__.__name__}]" + f"[{self._journal.status}]" + ": done")

        return success
    
    def _init_contact_sensors(self):
        
        for robot_name in self.contact_prims:
            
            # creates base contact sensor (which is then cloned)
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
            
            self._jnts_eff_default[robot_name] = torch.full((self._jnts_v[robot_name].shape[0], self._jnts_v[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self.torch_dtype, 
                                                        device=self.torch_device)
            
            self._root_abs_offsets[robot_name] = torch.zeros((self.num_envs, 3), 
                                device=self.torch_device) # reference clone positions
            # on the ground plane (init to 0)

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
                        
                        # self._jnts_v[robot_name][:, :].zero_()

                    else:
                        
                        # to avoid issues when differentiating numerically

                        self._root_v[robot_name][env_indxs, :].zero_()

                        self._root_omega[robot_name][env_indxs, :].zero_()
                        
                        self._jnts_v[robot_name][env_indxs, :].zero_()
            
                    # update "previous" data for numerical differentiation

                    self._root_p_prev[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :] 
                    self._root_q_prev[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]
                    self._jnts_q_prev[robot_name][env_indxs, :] = self._jnts_q[robot_name][env_indxs, :]

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

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + \
                        "Before calling __set_robots_default_jnt_config(), you need to reset the World" + \
                        " at least once and call post_initialization_steps()")

    def _set_robots_root_default_config(self):
        
        if (self._world_initialized):

            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self._robots_art_views[robot_name].set_default_state(positions = self._root_p_default[robot_name], 
                            orientations = self._root_q_default[robot_name])
            
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + \
                        "Before calling _set_robots_root_default_config(), you need to reset the World" + \
                        " at least once and call post_initialization_steps()")
        

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

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + \
                            "Before calling update_art_solver_options(), you need to reset the World at least once!")
        
    def _print_envs_info(self):
        
        if (self._world_initialized):
            
            print("TASK INFO:")

            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                print(f"[{robot_name}]")
                print("bodies: " + str(self._robots_art_views[robot_name].body_names))
                print("n. prims: " + str(self._robots_art_views[robot_name].count))
                print("prims names: " + str(self._robots_art_views[robot_name].prim_paths))
                print("n. bodies: " + str(self._robots_art_views[robot_name].num_bodies))
                print("n. dofs: " + str(self._robots_art_views[robot_name].num_dof))
                print("dof names: " + str(self._robots_art_views[robot_name].dof_names))
                print("solver_position_iteration_counts: " + str(self._solver_position_iteration_counts[robot_name]))
                print("solver_velocity_iteration_counts: " + str(self._solver_velocity_iteration_counts[robot_name]))
                print("stabiliz. thresholds: " + str(self._solver_stabilization_threshs[robot_name]))
                
                # print("dof limits: " + str(self._robots_art_views[robot_name].get_dof_limits()))
                # print("effort modes: " + str(self._robots_art_views[robot_name].get_effort_modes()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("dof max efforts: " + str(self._robots_art_views[robot_name].get_max_efforts()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("physics handle valid: " + str(self._robots_art_views[robot_name].is_physics_handle_valid()))

        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + \
                            "Before calling __print_envs_info(), you need to reset the World at least once!")

    def _fill_robot_info_from_world(self):

        if self._world_initialized:
            
            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self.robot_bodynames[robot_name] = self._robots_art_views[robot_name].body_names
                self.robot_n_links[robot_name] = self._robots_art_views[robot_name].num_bodies
                self.robot_n_dofs[robot_name] = self._robots_art_views[robot_name].num_dof
                self.robot_dof_names[robot_name] = self._robots_art_views[robot_name].dof_names
        
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + \
                        "Before calling _fill_robot_info_from_world(), you need to reset the World at least once!")

    def _init_homing_managers(self):
        
        if self._world_initialized:

            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self.homers[robot_name] = OmniRobotHomer(articulation=self._robots_art_views[robot_name], 
                                    srdf_path=self._srdf_paths[robot_name], 
                                    device=self.torch_device, 
                                    dtype=self.torch_dtype)
                    
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + ": you should reset the World at least once and call the " + \
                            "post_initialization_steps() method before initializing the " + \
                            "homing manager."
                            )
    
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
                                            urdf_path=self._urdf_paths[robot_name],
                                            debug = self._debug_mode_jnt_imp)

                self.reset_jnt_imp_control(robot_name)
                
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self._journal.exception}]" + ": you should reset the World at least once and call the " + \
                            "post_initialization_steps() method before initializing the " + \
                            "joint impedance controller."
                            )
    
    def _set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")
