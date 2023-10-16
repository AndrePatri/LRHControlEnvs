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
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.rotations import euler_angles_to_quat 

import omni.kit

import gymnasium as gym
from gym import spaces
import numpy as np
import torch

from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.prims import move_prim
from omni.isaac.cloner import GridCloner
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.sensor import ContactSensor

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.scenes.scene import Scene

from omni_robo_gym.utils.jnt_imp_cntrl import OmniJntImpCntrl
from omni_robo_gym.utils.homing import OmniRobotHomer
from omni_robo_gym.utils.defs import Journal
from omni_robo_gym.utils.terrains import RlTerrains
from abc import abstractmethod
from typing import List, Dict

class CustomTask(BaseTask):

    def __init__(self, 
                name: str,
                robot_names: List[str],
                robot_pkg_names: List[str] = None, 
                num_envs = 1,
                device = "cuda", 
                cloning_offset: np.array = np.array([0.0, 0.0, 0.0]),
                replicate_physics: bool = True,
                offset=None, 
                env_spacing = 5.0, 
                spawning_radius = 1.0,
                use_flat_ground = True,
                default_jnt_stiffness = 300.0,
                default_jnt_damping = 20.0,
                dtype = torch.float64) -> None:

        self.torch_dtype = dtype
        
        self.num_envs = num_envs

        self.torch_device = torch.device(device) # defaults to "cuda" ("cpu" also valid)

        self.journal = Journal()
        
        self.robot_names = robot_names # these are (potentially) custom names to 
        self.robot_pkg_names = robot_pkg_names # will be used to search for URDF and SRDF packages

        if self.robot_pkg_names is None:
            
            self.robot_pkg_names = self.robot_names # if not provided, robot_names are the same as robot_pkg_names
        
        else:

            if len(robot_names) != len(robot_pkg_names):

                exception = "The provided robot names list must match the length " + \
                    "of the provided robot package names"
                
                raise Exception(exception)
            
        self._urdf_paths = {}
        self._srdf_paths = {}
        self._robots_art_views = {}
        self._robots_articulations = {}
        self.robot_bodynames =  {}
        self.robot_n_links =  {}
        self.robot_n_dofs =  {}
        self.robot_dof_names =  {}

        self.root_p = {}
        self.root_q = {}
        self.root_v = {}
        self.root_omega = {}
        self.jnts_q = {} 
        self.jnts_v = {}
        self.root_p_default = {} 
        self.root_q_default = {}
        
        self.root_abs_offsets = {}

        self.distr_offset = {} # decribed how robots within each env are distributed
 
        self.jnt_imp_controllers = {}

        self.homers = {} 
        
        self.contact_sensors = {}

        self.default_jnt_stiffness = default_jnt_stiffness
        self.default_jnt_damping = default_jnt_damping
        
        self.use_flat_ground = use_flat_ground
        
        self.spawning_radius = spawning_radius # [m] -> default distance between roots of robots in a single 
        # environment 
        self.calc_robot_distrib() # computes the offsets of robots withing each env.

        # environment cloing stuff
        
        self._env_ns = "/World/envs"
        self._env_spacing = env_spacing # [m]
        self._template_env_ns = self._env_ns + "/env_0"

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self._env_ns)

        prim_utils.define_prim(self._template_env_ns)
        self._envs_prim_paths = self._cloner.generate_paths(self._env_ns + "/env", 
                                                self.num_envs)
        if len(cloning_offset) != 3:
            cloning_offset = np.array([0.0, 0.0, 0.0])
            print(f"[{self.__class__.__name__}]" + f"[{self.journal.warning}]" + ":  the provided cloning_offset is not of the correct shape. A null offset will be used instead.")

        self._cloning_offset = cloning_offset
        
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

        self.contact_sensors = []

        # trigger __init__ of parent class
        BaseTask.__init__(self,
                        name=name, 
                        offset=offset)

        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class

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
                            + f"[{self.journal.exception}]" + 
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
                            f"[{self.journal.exception}]" + 
                            ": failed to generate " + robot_name + "\'s URDF!!!")

    def _generate_rob_descriptions(self, 
                    robot_name: str, 
                    robot_pkg_name: str):
        
        self._descr_dump_path = "/tmp/" + f"{self.__class__.__name__}"
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": generating URDF...")

        self._generate_urdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name)

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": generating SRDF...")

        # we also generate SRDF files, which are useful for control
        self._generate_srdf(robot_name=robot_name, 
                        robot_pkg_name=robot_pkg_name)
        
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": done")

    def _import_urdf(self, 
                robot_name: str,
                import_config: omni.isaac.urdf._urdf.ImportConfig = _urdf.ImportConfig()):

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": importing robot URDF")

        # we overwrite some settings which are bound to be fixed
        import_config.merge_fixed_joints = True # makes sim more stable
        # in case of fixed joints with light objects
        import_config.import_inertia_tensor = True
        import_config.fix_base = False
        import_config.self_collision = False

        _urdf.acquire_urdf_interface()
        
        # import URDF
        success, robot_prim_path_default = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self._urdf_paths[robot_name],
            import_config=import_config, 
        )

        robot_base_prim_path = self._template_env_ns + "/" + robot_name

        # moving def prim
        move_prim(robot_prim_path_default, # from
                robot_base_prim_path) # to

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": done")

        return success
    
    def init_root_abs_offsets(self, 
                    robot_name: str):
            
        self.root_abs_offsets[robot_name][:, 0:2]  = self.root_p[robot_name][:, 0:2]

    def _init_robots_state(self):

        for i in range(0, len(self.robot_names)):

            robot_name = self.robot_names[i]

            pose = self._robots_art_views[robot_name].get_world_poses( 
                                clone = True) # tuple: (pos, quat)
        
            self.root_p[robot_name] = pose[0]  

            self.root_q[robot_name] = pose[1] # root orientation

            self.root_v[robot_name] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocity
            
            self.root_omega[robot_name] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True) # root ang. velocity
            
            self.jnts_q[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            
            self.jnts_v[robot_name] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True) # joint velocities
            
            self.root_p_default[robot_name] = torch.clone(pose[0]) + self.distr_offset[robot_name]

            self.root_q_default[robot_name] = torch.clone(pose[1])

            self.root_abs_offsets[robot_name] = torch.zeros((self.num_envs, 3), 
                                device=self.torch_device) # reference clone positions
            # on the ground plane (init to 0)

            self.init_root_abs_offsets(robot_name)
            
    def synch_default_root_states(self):

        for i in range(0, len(self.robot_names)):

            robot_name = self.robot_names[i]

            self.root_p_default[robot_name][:, :] = self.root_p[robot_name]

            self.root_q_default[robot_name][:, :] = self.root_q[robot_name]

    def calc_robot_distrib(self):

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

    def _get_robots_state(self):
        
        for i in range(0, len(self.robot_names)):

            robot_name = self.robot_names[i]

            pose = self._robots_art_views[robot_name].get_world_poses( 
                                            clone = True) # tuple: (pos, quat)
            
            self.root_p[robot_name][:, :] = pose[0]  

            self.root_q[robot_name][:, :] = pose[1] # root orientation

            self.root_v[robot_name][:, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocity
            
            self.root_omega[robot_name][:, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True) # root ang. velocity
            
            self.jnts_q[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            
            self.jnts_v[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True) # joint velocities
            
    def world_was_initialized(self):

        self._world_initialized = True

    def set_robots_default_jnt_config(self):
        
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

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                        "Before calling _set_robots_default_jnt_config(), you need to reset the World" + \
                        " at least once and call _world_was_initialized()")

    def set_robots_root_default_config(self):
        
        if (self._world_initialized):

            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self._robots_art_views[robot_name].set_default_state(positions = self.root_p_default[robot_name], 
                            orientations = self.root_q_default[robot_name])
            
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                        "Before calling set_robots_root_default_config(), you need to reset the World" + \
                        " at least once and call _world_was_initialized()")
        

        return True
        
    def apply_collision_filters(self, 
                                physicscene_path: str, 
                                coll_root_path: str):

        self._cloner.filter_collisions(physicsscene_path = physicscene_path,
                                collision_root_path = coll_root_path, 
                                prim_paths=self._envs_prim_paths, 
                                global_paths=[self._ground_plane_prim_path] # can collide with these prims
                            )

    def print_envs_info(self):
        
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
                # print("dof limits: " + str(self._robots_art_views[robot_name].get_dof_limits()))
                # print("effort modes: " + str(self._robots_art_views[robot_name].get_effort_modes()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("dof max efforts: " + str(self._robots_art_views[robot_name].get_max_efforts()))
                # print("dof gains: " + str(self._robots_art_views[robot_name].get_gains()))
                # print("physics handle valid: " + str(self._robots_art_views[robot_name].is_physics_handle_valid()))

        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                            "Before calling _print_envs_info(), you need to reset the World at least once!")

    def fill_robot_info_from_world(self):

        if self._world_initialized:
            
            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self.robot_bodynames[robot_name] = self._robots_art_views[robot_name].body_names
                self.robot_n_links[robot_name] = self._robots_art_views[robot_name].num_bodies
                self.robot_n_dofs[robot_name] = self._robots_art_views[robot_name].num_dof
                self.robot_dof_names[robot_name] = self._robots_art_views[robot_name].dof_names
        
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                        "Before calling _get_robot_info_from_world(), you need to reset the World at least once!")

    def init_homing_managers(self):
        
        if self.world_was_initialized:

            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]

                self.homers[robot_name] = OmniRobotHomer(articulation=self._robots_art_views[robot_name], 
                                    srdf_path=self._srdf_paths[robot_name], 
                                    device=self.torch_device, 
                                    dtype=self.torch_dtype)
                    
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + ": you should reset the World at least once and call the " + \
                            "world_was_initialized() method before initializing the " + \
                            "homing manager."
                            )
        
    def init_imp_control(self, 
                default_jnt_pgain = 300.0, 
                default_jnt_vgain = 20.0, 
                default_wheel_pgain = 0.0, # by default wheels are supposed to be controlled in velocity mode
                default_wheel_vgain = 10.0):

        if self.world_was_initialized:
            
            for i in range(0, len(self.robot_names)):

                robot_name = self.robot_names[i]
                
                self.jnt_imp_controllers[robot_name] = OmniJntImpCntrl(articulation=self._robots_art_views[robot_name],
                                                default_pgain = default_jnt_pgain, 
                                                default_vgain = default_jnt_vgain,
                                                device= self.torch_device, 
                                                dtype=self.torch_dtype)

                # we override internal default gains for the wheels, which are usually
                # velocity controlled
                wheels_indxs = self.jnt_imp_controllers[robot_name].get_jnt_idxs_matching(name_pattern="wheel")

                if wheels_indxs.numel() != 0: # the robot has wheels

                    wheels_pos_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                                default_wheel_pgain, 
                                                device = self.torch_device, 
                                                dtype=self.torch_dtype)
                    
                    wheels_vel_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                                default_wheel_vgain, 
                                                device = self.torch_device, 
                                                dtype=self.torch_dtype)

                    self.jnt_imp_controllers[robot_name].set_gains(pos_gains = wheels_pos_gains,
                                                vel_gains = wheels_vel_gains,
                                                jnt_indxs=wheels_indxs)
                            
                try:

                    self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing())
                
                except Exception:
                    
                    print(f"[{self.__class__.__name__}]" + f"[{self.journal.warning}]" +  f"[{self.init_imp_control.__name__}]" +\
                    ": cannot set imp. controller reference to homing. Did you call the \"init_homing_managers\" method ?")

                    pass                
                
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + ": you should reset the World at least once and call the " + \
                            "world_was_initialized() method before initializing the " + \
                            "joint impedance controller."
                            )
    
    def init_contact_sensors(self, 
                    world: omni.isaac.core.world.world.World):
          
        
        for i in range(len(self._envs_prim_paths)):
                        
            for j in range(0, len(self.robot_names)):
                
                robot_name = self.robot_names[j]
                
                prim_path = self._env_ns + f"/env_{0}" + "/" + robot_name + "/wheel_1" + "/contact_sensor"
                # prim_utils.define_prim(prim_path)

                self.contact_sensors[robot_name] = []

                self.contact_sensors[robot_name].append(
                            world.scene.add(
                                ContactSensor(
                                    prim_path=prim_path,
                                    name=f"{robot_name}_contact_sensor{i}".format(i),
                                    min_threshold=0,
                                    max_threshold=10000000,
                                    radius=0.1, 
                                    translation=np.zeros((1, 3))
                                )
                            )
                        )

                self.contact_sensors[robot_name][i].add_raw_contact_data_to_frame()

    def set_up_scene(self, 
                    scene: Scene) -> None:

        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]
            robot_pkg_name = self.robot_pkg_names[i]

            self._generate_rob_descriptions(robot_name=robot_name, 
                                    robot_pkg_name=robot_pkg_name)

            self._import_urdf(robot_name)
        
        pos_offsets = np.zeros((self.num_envs, 3))
        for i in range(0, self.num_envs):
            pos_offsets[i, :] = self._cloning_offset
        
        print(f"[{self.__class__.__name__}]" + \
            f"[{self.journal.status}]" + \
            ": cloning environments...")
        
        self._cloner.clone(
            source_prim_path=self._template_env_ns,
            prim_paths=self._envs_prim_paths,
            replicate_physics=self._replicate_physics,
            position_offsets = pos_offsets
        ) # we can clone the environment in which all the robos are

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": finishing scene setup...")
        
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            self._robots_art_views[robot_name] = ArticulationView(name = robot_name,
                                                        prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name, 
                                                        reset_xform_properties=False)

            self._robots_articulations[robot_name] = scene.add(self._robots_art_views[robot_name])

            # self._robots_geom_view = GeometryPrimView(
            #                     prim_paths_expr = self._env_ns + f"/env*" + "/" + self._robot_prim_name + "/wheel_1", 
            #                     # name=self._robot_name + "geom_views", 
            #                     # # collisions = torch.tensor([None] * self.num_envs), 
            #                     # track_contact_forces = False, 
            #                     prepare_contact_sensors = False
            #                     ) # geometry view (useful to enable contact reporting)
            # self._robots_geom_view.apply_collision_apis() # random data with GPU pipeline!!!

            # self._robots_geometries = scene.add(self._robots_geom_view)

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
        self.set_initial_camera_params()
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": done")

    def set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        
        # post reset operations
        
        pass

    def reset(self, 
            env_ids=None):
        
        self.set_robots_default_jnt_config()
        self.set_robots_root_default_config()

        self._get_robots_state() # updates robot states
        
        for i in range(len(self.robot_names)):
            
            robot_name = self.robot_names[i]

            self._robots_art_views[robot_name].set_velocities(torch.zeros((self.num_envs, 
                                                        6), device=self.torch_device))

            self._robots_art_views[robot_name].post_reset()

            self.jnt_imp_controllers[robot_name].set_refs(pos_ref=self.homers[robot_name].get_homing())
            self.jnt_imp_controllers[robot_name].apply_refs()

    @abstractmethod
    def pre_physics_step(self, 
                actions, 
                robot_name: str) -> None:
        
        # apply actions to simulated robot

        pass

    def get_observations(self):
        
        # retrieve data from the simulation

        pass

    def calculate_metrics(self) -> None:
        
        # compute any metric to be fed to the agent

        pass

    def is_done(self) -> None:
        
        pass
