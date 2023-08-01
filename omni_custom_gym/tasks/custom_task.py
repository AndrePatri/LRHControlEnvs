from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.core.utils.rotations import euler_angles_to_quat 

import omni.kit

import gymnasium as gym
from gym import spaces
import numpy as np
import torch
import math

from omni.isaac.urdf import _urdf
from omni.isaac.core.utils.prims import move_prim
from omni.isaac.cloner import GridCloner
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.urdf._urdf import UrdfJointTargetType

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.core.scenes.scene import Scene

from omni_custom_gym.utils.jnt_imp_cntrl import OmniJntImpCntrl
from omni_custom_gym.utils.homing import OmniRobotHomer

from abc import ABC, abstractmethod
from typing import List

class CustomTask(BaseTask):
    def __init__(self, 
                name: str,
                robot_name = "MyRobot", 
                num_envs = 1,
                device = "cuda", 
                cloning_offset: np.array = np.array([0.0, 0.0, 0.0]),
                replicate_physics: bool = True,
                offset=None, 
                env_spacing = 5.0) -> None:

        self.info = "info"
        self.status = "status"
        self.warning = "warning"
        self.exception = "exception"
        
        self._robot_name = robot_name # will be used to search for URDF and SRDF packages

        # cloning stuff
        self.num_envs = num_envs
        self._env_ns = "/World/envs"
        self._env_spacing = env_spacing # [m]
        self._template_env_ns = self._env_ns + "/env_0"

        self._cloner = GridCloner(spacing=self._env_spacing)
        self._cloner.define_base_env(self._env_ns)

        prim_utils.define_prim(self._template_env_ns)
        self._envs_prim_paths = self._cloner.generate_paths(self._env_ns + "/env", 
                                                self.num_envs)

        # task-specific parameters
        if len(cloning_offset) != 3:
            cloning_offset = np.array([0.0, 0.0, 0.0])
            print(f"[{self.__class__.__name__}]" + f"[{self.warning}]" + ":  the provided cloning_offset is not of the correct shape. A null offset will be used instead.")

        self._cloning_offset = cloning_offset

        self._position = np.array([0, 0, 0.8])
        self._orientation = euler_angles_to_quat(np.array([0, 0, 0]), degrees = True)

        # values used for defining RL buffers
        self._num_observations = 4
        self._num_actions = 1
        self.torch_device = torch.device(device) # defaults to "cuda" ("cpu" also valid)

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

        self.robot_bodynames = []
        self.robot_n_links = -1
        self.robot_n_dofs = -1
        self.robot_dof_names = []

        self._ground_plane_prim_path = "/World/ground_plane"

        # trigger __init__ of parent class
        BaseTask.__init__(self,
                        name=name, 
                        offset=offset)
        
        self._jnt_imp_controller = None 

        self._homer = None 

        self.xrdf_cmd_vals = [] # by default empty, needs to be overriden by
        # child class
    
    @abstractmethod
    def _xrdf_cmds(self, 
                   vals: List[bool]) -> List[str]:

        pass
    
    def _generate_srdf(self):
        
        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        descr_path = rospackage.get_path(self._robot_name + "_srdf")
        srdf_path = descr_path + "/srdf"
        xacro_name = self._robot_name
        xacro_path = srdf_path + "/" + xacro_name + ".srdf.xacro"
        self._srdf_path = self._descr_dump_path + "/" + xacro_name + ".srdf"

        cmds = self._xrdf_cmds()
        if cmds is None:

            cmds = []

        import subprocess
        try:
            
            xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._srdf_path]
            xacro_gen = subprocess.check_call(xacro_cmd)

        except:

            raise Exception(f"[{self.__class__.__name__}]" 
                            + f"[{self.exception}]" + 
                            ": failed to generate " + self._robot_name + "\'S SRDF!!!")
        
    def _generate_urdf(self):

        # we generate the URDF where the Kyon description package is located
        import rospkg
        rospackage = rospkg.RosPack()
        descr_path = rospackage.get_path(self._robot_name + "_urdf")
        urdf_path = descr_path + "/urdf"
        xacro_name = self._robot_name
        xacro_path = urdf_path + "/" + xacro_name + ".urdf.xacro"
        self._urdf_path = self._descr_dump_path + "/" + xacro_name + ".urdf"
        
        cmds = self._xrdf_cmds()
        if cmds is None:

            cmds = []

        import subprocess
        try:

            xacro_cmd = ["xacro"] + [xacro_path] + cmds + ["-o"] + [self._urdf_path]
            xacro_gen = subprocess.check_call(xacro_cmd)
            
            # we also generate an updated SRDF (used by controllers)

        except:

            raise Exception(f"[{self.__class__.__name__}]" + 
                            f"[{self.exception}]" + 
                            ": failed to generate " + self._robot_name+ "\'s URDF!!!")

    def _generate_description(self):
        
        self._descr_dump_path = "/tmp/" + f"{self.__class__.__name__}"
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": generating URDF...")
        self._generate_urdf()
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": generating SRDF...")
        # we also generate SRDF files, which are useful for control
        self._generate_srdf()
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": done")

    def _import_urdf(self, 
                    import_config: omni.isaac.urdf._urdf.ImportConfig = _urdf.ImportConfig()):

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": importing robot URDF")

        self._urdf_import_config = import_config
        # we overwrite some settings which are bound to be fixed
        self._urdf_import_config.merge_fixed_joints = True # makes sim more stable
        # in case of fixed joints with light objects
        self._urdf_import_config.import_inertia_tensor = True
        self._urdf_import_config.fix_base = False
        self._urdf_import_config.self_collision = False

        self._urdf_interface = _urdf.acquire_urdf_interface()

        self._robot_prim_name = self._robot_name
        
        # import URDF
        success, robot_prim_path_default = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=self._urdf_path,
            import_config=import_config, 
        )
        self._robot_base_prim_path = self._template_env_ns + "/" + self._robot_prim_name
        move_prim(robot_prim_path_default, self._robot_base_prim_path)# we move the prim
        # from the default one of the URDF importer to the prescribed one

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": done")

        return success
    
    def _get_robots_state(self):
        
        pose = self._robots_art_view.get_world_poses(indices = None, 
                                        clone = False) # tuple: (pos, quat)
        
        self.root_p = pose[0] # root position 

        self.root_q = pose[1] # root orientation

        self.root_v = self._robots_art_view.get_linear_velocities(indices = None, 
                                        clone = False) # root lin. velocity
        
        self.root_omega = self._robots_art_view.get_angular_velocities(indices = None, 
                                        clone = False) # root ang. velocity
        
        self.jnts_q = self._robots_art_view.get_joint_positions(indices = None, 
                                        joint_indices = None, 
                                        clone = False) # joint positions 
        
        self.jnts_v = self._robots_art_view.get_joint_velocities(indices = None, 
                                        joint_indices = None, 
                                        clone = False) # joint velocities
        
        # self.velocities = self._robots_art_view.get_velocities(indices = None, 
        #                                 clone = True) # [n_envs x 6]; 0:3 lin vel; 3:6 ang vel 

    def world_was_initialized(self):

        self._world_initialized = True

    def set_robot_default_jnt_config(self):
        
        if (self._world_initialized):

            self._robots_art_view.set_joints_default_state(positions= self._homer.get_homing())
            
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + \
                        "Before calling _set_robot_default_jnt_config(), you need to reset the World" + \
                        " at least once and call _world_was_initialized()")

    def set_robot_root_default_config(self):
        
        # To be implemented

        return True

    # def override_pd_controller_gains(self):
        
    #     # all gains set to 0 so that it's possible to 
    #     # attach to the articulation a custom joint controller (e.g. jnt impedance), 
    #     # on top of the default articulation pd controller

    #     self.joint_kps_envs = torch.zeros((self.num_envs, self.robot_n_dofs))
    #     self.joint_kds_envs = torch.zeros((self.num_envs, self.robot_n_dofs)) 

    #     self._robots_art_view.set_gains(kps= self.joint_kps_envs, 
    #                                     kds= self.joint_kds_envs)
        
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
            print("Envs bodies: " + str(self._robots_art_view.body_names))
            print("n. prims: " + str(self._robots_art_view.count))
            print("prims names: " + str(self._robots_art_view.prim_paths))
            print("n. bodies: " + str(self._robots_art_view.num_bodies))
            print("n. dofs: " + str(self._robots_art_view.num_dof))
            print("dof names: " + str(self._robots_art_view.dof_names))
            # print("dof limits: " + str(self._robots_art_view.get_dof_limits()))
            # print("effort modes: " + str(self._robots_art_view.get_effort_modes()))
            # print("dof gains: " + str(self._robots_art_view.get_gains()))
            # print("dof max efforts: " + str(self._robots_art_view.get_max_efforts()))
            # print("dof gains: " + str(self._robots_art_view.get_gains()))
            # print("physics handle valid: " + str(self._robots_art_view.is_physics_handle_valid()))

        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + \
                            "Before calling _print_envs_info(), you need to reset the World at least once!")

    def fill_robot_info_from_world(self):

        if (self._world_initialized):

            self.robot_bodynames = self._robots_art_view.body_names
            self.robot_n_links = self._robots_art_view.num_bodies
            self.robot_n_dofs = self._robots_art_view.num_dof
            self.robot_dof_names = self._robots_art_view.dof_names
        
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + \
                        "Before calling _get_robot_info_from_world(), you need to reset the World at least once!")

    def init_homing_manager(self):

        if self.world_was_initialized:

            self._homer = OmniRobotHomer(articulation=self._robots_art_view, 
                                srdf_path=self._srdf_path, 
                                device=self.torch_device)
            
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + ": you should reset the World at least once and call the " + \
                            "world_was_initialized() method before initializing the " + \
                            "homing manager."
                            )
        
    def init_imp_control(self, 
                default_jnt_pgain = 300.0, 
                default_jnt_vgain = 30.0, 
                default_wheel_pgain = 0.0, 
                default_wheel_vgain = 10.0):

        if self.world_was_initialized:

            self._jnt_imp_controller = OmniJntImpCntrl(articulation=self._robots_art_view,
                                                default_pgain = default_jnt_pgain, 
                                                default_vgain = default_jnt_vgain,
                                                device= self.torch_device)

            # we override internal default gains for the wheels, which are usually
            # velocity controlled
            wheels_indxs = self._jnt_imp_controller.get_jnt_idxs_matching(name_pattern="wheel")

            wheels_pos_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                        default_wheel_pgain, 
                                        device = self.torch_device, 
                                        dtype=torch.float32)
            
            wheels_vel_gains = torch.full((self.num_envs, len(wheels_indxs)), 
                                        default_wheel_vgain, 
                                        device = self.torch_device, 
                                        dtype=torch.float32)

            self._jnt_imp_controller.set_gains(pos_gains = wheels_pos_gains,
                            vel_gains = wheels_vel_gains,
                            jnt_indxs=wheels_indxs)

            if self._homer is not None:

                self._jnt_imp_controller.set_refs(pos_ref=self._homer.get_homing())
            
            else:
                
                print(f"[{self.__class__.__name__}]" + f"[{self.warning}]" +  f"[{self.init_imp_control.__name__}]" +\
                    ": cannot set imp. controller reference to homing. Did you call the \"init_homing_manager\" method ?")
        else:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + ": you should reset the World at least once and call the " + \
                            "world_was_initialized() method before initializing the " + \
                            "joint impedance controller."
                            )
        
    def set_up_scene(self, 
                    scene: Scene) -> None:

        self._generate_description()

        self._import_urdf()
        
        pos_offsets = np.zeros((self.num_envs, 3))
        for i in range(0, self.num_envs):
            pos_offsets[i, :] = self._cloning_offset
        
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": cloning environments")
        envs_positions = self._cloner.clone(
            source_prim_path=self._template_env_ns,
            prim_paths=self._envs_prim_paths,
            replicate_physics=self._replicate_physics,
            position_offsets = pos_offsets
        ) # robot is now at the default env prim --> we can clone the environment
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": done")

        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": finishing scene setup...")
        self._robots_art_view = ArticulationView(self._env_ns + "/env*"+ "/" + self._robot_prim_name, 
                                reset_xform_properties=False)

        self._robots_articulations = scene.add(self._robots_art_view)

        scene.add_default_ground_plane(z_position=0, 
                            name="ground_plane", 
                            prim_path= self._ground_plane_prim_path, 
                            static_friction=0.5, 
                            dynamic_friction=0.5, 
                            restitution=0.8)
        
        # delete_prim(self._ground_plane_prim_path + "/SphereLight") # we remove the default spherical light
        
        # set default camera viewport position and target
        self.set_initial_camera_params()
        print(f"[{self.__class__.__name__}]" + f"[{self.status}]" + ": done")

    def set_initial_camera_params(self, 
                                camera_position=[10, 10, 3], 
                                camera_target=[0, 0, 0]):
        
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")

    def post_reset(self):
        
        # post reset operations
        
        pass

    def reset(self, env_ids=None):
        
        # reset simulation 

        pass

    def pre_physics_step(self, actions) -> None:
        
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
