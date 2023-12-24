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
from omni.isaac.kit import SimulationApp
import os
import signal

import carb
import gymnasium as gym 

import torch

from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict

import numpy as np

from omni_robo_gym.utils.defs import Journal

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

        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.kit'
        # experience = ""
        if headless:

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": will run in headless mode")
            
            if enable_livestream:
                 
                experience = ""
            
            elif enable_viewport:
                 
                raise Exception(f"[{self.__class__.__name__}]" + f"[{self.journal.exception}]" + \
                            ": using viewport is not supported yet.")
                
            else:
                 
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.headless.kit'
                # experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp({"headless": headless,
                                            "physics_gpu": sim_device}, 
                                            experience=experience)

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using IsaacSim experience file @ " + experience)

        # carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

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
        
        # handle ctrl+c event
        signal.signal(signal.SIGINT, self.signal_handler)

        self._render = not headless or enable_livestream or enable_viewport
        self._record = False
        self.sim_frame_count = 0
        self._world = None
        self.metadata = None

        self.gpu_pipeline_enabled = False
    
    def signal_handler(self, sig, frame):
        self.close()

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

        from omni.isaac.core.world import World

        # parse device based on sim_param settings
        if sim_params and "sim_device" in sim_params:
            device = sim_params["sim_device"]
        else:
            device = "cpu"
            physics_device_id = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
            gpu_id = 0 if physics_device_id < 0 else physics_device_id
            if sim_params and "use_gpu_pipeline" in sim_params:
                # GPU pipeline must use GPU simulation
                if sim_params["use_gpu_pipeline"]:
                    device = "cuda:" + str(gpu_id)
            elif sim_params and "use_gpu" in sim_params:
                if sim_params["use_gpu"]:
                    device = "cuda:" + str(gpu_id)
        
        self.gpu_pipeline_enabled = sim_params["use_gpu_pipeline"]

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using device: " + str(device))

        if (sim_params is None):
            
            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": no sim params provided -> defaults will be used")
            sim_params = {}

        # defaults for integration and rendering dt
        if not("integration_dt" in sim_params):
    
            sim_params["integration_dt"] = 1.0/60.0

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using default integration_dt of " + 
                sim_params["integration_dt"] + " s.")
            
        if not("rendering_dt" in sim_params):

            sim_params["rendering_dt"] = sim_params["integration_dt"]

            print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + ": using default rendering_dt of " + 
                sim_params["rendering_dt"] + " s.")

        self._world = World(
            stage_units_in_meters=1.0, 
            physics_dt=sim_params["integration_dt"], 
            rendering_dt=sim_params["rendering_dt"], # dt between rendering steps. Note: rendering means rendering a frame of 
            # the current application and not only rendering a frame to the viewports/ cameras. 
            # So UI elements of Isaac Sim will be refereshed with this dt as well if running non-headless
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

        self._world._current_tasks = dict() # resets registered tasks
        self._world.add_task(task)
        self._task = task
        self._task.set_world(self._world)
        self._num_envs = self._task.num_envs

        # filter collisions between envs
        self._task.apply_collision_filters(self._physics_scene_path, 
                                "/World/collisions")
        
        self.observation_space = self._task.observation_space
        self.action_space = self._task.action_space

        if sim_params and "enable_viewport" in sim_params:
            self._render = sim_params["enable_viewport"]

        print(f"[{self.__class__.__name__}]" + f"[{self.journal.info}]" + "[render]: " + str(self._render))

        if init_sim:

            print("Jijijijjijijijijiji")
            self._world.reset() # after the first reset we get get all quantities 
            # from the scene 

            print("AUUHYHGYBIMOIIUBIBIBIUbn")
            self._task.post_initialization_steps() # performs initializations 
            # steps after the fisrt world reset was called

    def render(self, mode="human") -> None:
        """ Step the renderer.

        Args:
            mode (str): Select mode of rendering based on OpenAI environments.
        """

        if mode == "human":
            self._world.render()
            return None
        elif mode == "rgb_array":
            # check if viewport is enabled -- if not, then complain because we won't get any data
            if not self._render or not self._record:
                raise RuntimeError(
                    f"Cannot render '{mode}' when rendering is not enabled. Please check the provided"
                    "arguments to the environment class at initialization."
                )
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            gym.Env.render(self, mode=mode)
            return None

    def create_viewport_render_product(self, resolution=(1280, 720)):
        """Create a render product of the viewport for rendering."""

        try:
            import omni.replicator.core as rep

            # create render product
            self._render_product = rep.create.render_product("/OmniverseKit_Persp", resolution)
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            self._record = True
        except Exception as e:
            carb.log_info("omni.replicator.core could not be imported. Skipping creation of render product.")
            carb.log_info(str(e))

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
    
    @property
    def simulation_app(self):
        """Retrieves the SimulationApp object.

        Returns:
            simulation_app(SimulationApp): SimulationApp.
        """
        return self._simulation_app

    @property
    def world(self):
        """Retrieves the World object for simulation.

        Returns:
            world(World): Simulation World.
        """
        return self._world

    @property
    def task(self):
        """Retrieves the task.

        Returns:
            task(BaseTask): Task.
        """
        return self._task

    @property
    def render_enabled(self):
        """Whether rendering is enabled.

        Returns:
            render(bool): is render enabled.
        """
        return self._render
