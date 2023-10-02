# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp
import os
import carb
import gymnasium as gym 
import torch
from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict
import numpy as np
from omni_custom_gym.utils.defs import Journal

class RobotVecEnv(gym.Env):
    """ This class provides a base interface for connecting RL policies with task implementations.
        APIs provided in this interface follow the interface in gymnasium.Env.
        This class also provides utilities for initializing simulation apps, creating the World,
        and registering a task.
    """

    def __init__(
        self, 
        headless: bool, 
        sim_device: int = 0, 
        enable_livestream: bool = False, 
        enable_viewport: bool = False
    ) -> None:
        """ Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
        """

        self.journal = Journal()

        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnicustomgym.kit'
        # experience = ""
        if headless:

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": will run in headless mode")
            
            if enable_livestream:
                experience = ""
            elif enable_viewport:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnicustomgym.headless.render.kit'
                # experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.render.kit'
            else:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnicustomgym.headless.kit'
                # experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp({"headless": headless,
                                            "physics_gpu": sim_device}, 
                                            experience=experience)

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using IsaacSim experience file @ " + experience)

        carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)
        self._render = not headless or enable_livestream or enable_viewport
        self.sim_frame_count = 0

        if enable_livestream:

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": livestream enabled")

            from omni.isaac.core.utils.extensions import enable_extension

            self._simulation_app.set_setting("/app/livestream/enabled", True)
            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")
            enable_extension("omni.services.streaming.manager")

    def set_task(self, 
                task, 
                backend="torch", 
                sim_params=None, 
                init_sim=True) -> None:

        """ Creates a World object and adds Task to World. 
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            backend (str): Backend to use for task. Can be "numpy" or "torch". Defaults to "numpy".
            sim_params (dict): Simulation parameters for physics settings. Defaults to None.
            init_sim (Optional[bool]): Automatically starts simulation. Defaults to True.
        """

        ## we first set up the World ##
        from omni.isaac.core.world import World

        device = torch.device("cpu") # defaults to CPU, unless this is set in sim_params
        if sim_params and "use_gpu_pipeline" in sim_params:
            if sim_params["use_gpu_pipeline"]:
                device = torch.device("cuda") # 
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using device: " + str(device))
        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using backend: " + backend)

        if (sim_params is None):
            
            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": no sim params provided -> defaults will be used")
            sim_params = {}

        # defaults for integration and rendering dt
        if not("integration_dt" in sim_params):
    
            sim_params["integration_dt"] = 1.0/60.0

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using default integration_dt of " + 
                sim_params["integration_dt"] + " s.")
            
        if not("rendering_dt" in sim_params):

            sim_params["rendering_dt"] = 1.0/60.0

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using default rendering_dt of " + 
                sim_params["rendering_dt"] + " s.")

        self._world = World(
            stage_units_in_meters=1.0, 
            physics_dt=sim_params["integration_dt"], 
            rendering_dt=sim_params["rendering_dt"],
            backend=backend,
            device=str(device),
            physics_prim_path="/physicsScene", 
            sim_params=sim_params
        )

        self._sim_params = sim_params

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.status}]" + ": creating task " + task.name + "\n")

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + "[world]:")
        print("use_gpu_pipeline: " + str(sim_params["use_gpu_pipeline"]))
        print("device: " + str(device))
        print("backend: " + str(backend))
        print("integration_dt: " + str(sim_params["integration_dt"]))
        print("rendering_dt: " + str(sim_params["rendering_dt"]))

        ## we get the physics context to expose additional low-level ##
        # settings of the simulation
        self._physics_context = self._world.get_physics_context() 
        self._physics_scene_path = self._physics_context.prim_path
        self._physics_context.enable_gpu_dynamics(True)
        self._physics_context.enable_stablization(True)
        self._physics_scene_prim = self._physics_context.get_current_physics_scene_prim()
        self._solver_type = self._physics_context.get_solver_type()

        # we set parameters, depending on sim_params dict
        if "gpu_max_rigid_contact_count" in sim_params:
                self._physics_context.set_gpu_max_rigid_contact_count(sim_params["gpu_max_rigid_contact_count"])
        if "gpu_max_rigid_patch_count" in sim_params:
                self._physics_context.set_gpu_max_rigid_patch_count(sim_params["gpu_max_rigid_patch_count"])
        if "gpu_found_lost_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_found_lost_pairs_capacity(sim_params["gpu_found_lost_pairs_capacity"])
        if "gpu_found_lost_aggregate_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(sim_params["gpu_found_lost_aggregate_pairs_capacity"])
        if "gpu_total_aggregate_pairs_capacity" in sim_params:
                self._physics_context.set_gpu_total_aggregate_pairs_capacity(sim_params["gpu_total_aggregate_pairs_capacity"])
        if "gpu_max_soft_body_contacts" in sim_params:
                self._physics_context.set_gpu_max_soft_body_contacts(sim_params["gpu_max_soft_body_contacts"])
        if "gpu_max_particle_contacts" in sim_params:
                self._physics_context.set_gpu_max_particle_contacts(sim_params["gpu_max_particle_contacts"])
        if "gpu_heap_capacity" in sim_params:
                self._physics_context.set_gpu_heap_capacity(sim_params["gpu_heap_capacity"])
        if "gpu_temp_buffer_capacity" in sim_params:
                self._physics_context.set_gpu_temp_buffer_capacity(sim_params["gpu_temp_buffer_capacity"])
        if "gpu_max_num_partitions" in sim_params:
                self._physics_context.set_gpu_max_num_partitions(sim_params["gpu_max_num_partitions"])

        # overwriting defaults
        # self._physics_context.set_gpu_max_rigid_contact_count(2 * self._physics_context.get_gpu_max_rigid_contact_count())
        # self._physics_context.set_gpu_max_rigid_patch_count(2 * self._physics_context.get_gpu_max_rigid_patch_count())
        # self._physics_context.set_gpu_found_lost_pairs_capacity(2 * self._physics_context.get_gpu_found_lost_pairs_capacity())
        # self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(20 * self._physics_context.get_gpu_found_lost_aggregate_pairs_capacity())
        # self._physics_context.set_gpu_total_aggregate_pairs_capacity(20 * self._physics_context.get_gpu_total_aggregate_pairs_capacity())
        # self._physics_context.set_gpu_heap_capacity(2 * self._physics_context.get_gpu_heap_capacity())
        # self._physics_context.set_gpu_temp_buffer_capacity(20 * self._physics_context.get_gpu_heap_capacity())
        # self._physics_context.set_gpu_max_num_partitions(20 * self._physics_context.get_gpu_temp_buffer_capacity())

        # GPU buffers
        self._gpu_max_rigid_contact_count = self._physics_context.get_gpu_max_rigid_contact_count()
        self._gpu_max_rigid_patch_count = self._physics_context.get_gpu_max_rigid_patch_count()
        self._gpu_found_lost_pairs_capacity = self._physics_context.get_gpu_found_lost_pairs_capacity()
        self._gpu_found_lost_aggregate_pairs_capacity = self._physics_context.get_gpu_found_lost_aggregate_pairs_capacity()
        self._gpu_total_aggregate_pairs_capacity = self._physics_context.get_gpu_total_aggregate_pairs_capacity()
        self._gpu_max_soft_body_contacts = self._physics_context.get_gpu_max_soft_body_contacts()
        self._gpu_max_particle_contacts = self._physics_context.get_gpu_max_particle_contacts()
        self._gpu_heap_capacity = self._physics_context.get_gpu_heap_capacity()
        self._gpu_temp_buffer_capacity = self._physics_context.get_gpu_temp_buffer_capacity()
        # self._gpu_max_num_partitions = physics_context.get_gpu_max_num_partitions() # BROKEN->method does not exist

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + "[physics context]:")
        print("gpu_max_rigid_contact_count: " + str(self._gpu_max_rigid_contact_count))
        print("gpu_max_rigid_patch_count: " + str(self._gpu_max_rigid_patch_count))
        print("gpu_found_lost_pairs_capacity: " + str(self._gpu_found_lost_pairs_capacity))
        print("gpu_found_lost_aggregate_pairs_capacity: " + str(self._gpu_found_lost_aggregate_pairs_capacity))
        print("gpu_total_aggregate_pairs_capacity: " + str(self._gpu_total_aggregate_pairs_capacity))
        print("gpu_max_soft_body_contacts: " + str(self._gpu_max_soft_body_contacts))
        print("gpu_max_particle_contacts: " + str(self._gpu_max_particle_contacts))
        print("gpu_heap_capacity: " + str(self._gpu_heap_capacity))
        print("gpu_temp_buffer_capacity: " + str(self._gpu_temp_buffer_capacity))

        self._scene = self._world.scene

        from omni.usd import get_context
        self._stage = get_context().get_stage()

        from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
           
        # add lighting
        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(500)

        self._world.add_task(task)
        self.task = task

        # filter collisions between envs
        self.task.apply_collision_filters(self._physics_scene_path, 
                                "/World/collisions")
        
        self._num_envs = self.task.num_envs

        self.observation_space = self.task.observation_space
        self.action_space = self.task.action_space

        if sim_params and "enable_viewport" in sim_params:
            self._render = sim_params["enable_viewport"]

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + "[render]: " + str(self._render))

        if init_sim:

            self._world.reset() # after the first reset we get get all quantities 
            # from the scene 
            
            self.task.world_was_initialized() # we signal the task 
            # that the first reset was called -> all info is now available
            # to be retrieved

            self.task.fill_robot_info_from_world() # populates robot info fields
            # in task

            self.task.init_homing_managers() 

            self.task._init_robots_state()

            # self.task.set_robots_default_jnt_config()
            # self.task.set_robots_root_default_config()
            
            self.task.init_imp_control(default_jnt_pgain = self.task.default_jnt_stiffness, 
                            default_jnt_vgain = self.task.default_jnt_damping) # initialized the impedance controller

            self.task.reset()

            # self.task.init_contact_sensors(self._world)
            self.task.print_envs_info() # debug prints

    def render(self, mode="human") -> None:
        """ Step the renderer.

        Args:
            mode (str): Select mode of rendering based on OpenAI environments.
        """

        if mode == "human":
            self._world.render()
        else:
            gym.Env.render(self, mode=mode)
        return

    def close(self) -> None:
        """ Closes simulation.
        """

        # bypass USD warnings on stage close
        self._simulation_app.close()
        return

    def seed(self, seed=-1):
        """ Sets a seed. Pass in -1 for a random seed.

        Args:
            seed (int): Seed to set. Defaults to -1.
        Returns:
            seed (int): Seed that was set.
        """

        from omni.isaac.core.utils.torch.maths import set_seed

        return set_seed(seed)

    @abstractmethod
    def step(self, 
            index: int,
            actions = None) -> Tuple[Union[np.ndarray, torch.Tensor], 
                                    Union[np.ndarray, torch.Tensor],
                                    Union[np.ndarray, torch.Tensor],
                                    Dict]:
        """ Basic implementation for stepping simulation. 
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            actions (Union[numpy.ndarray, torch.Tensor]): Actions buffer from policy.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rewards(Union[numpy.ndarray, torch.Tensor]): Buffer of rewards data.
            dones(Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            info(dict): Dictionary of extras data.
        """
        
        pass
    
    @abstractmethod
    def reset(self):
        """ Usually resets the task and updates observations +
        # other custom operations. """

        pass

    @property
    def num_envs(self):
        """ Retrieves number of environments.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs
