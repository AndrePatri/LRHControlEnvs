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

import torch

from abc import ABC, abstractmethod
from typing import Union, Tuple, Dict

from SharsorIPCpp.PySharsorIPC import VLevel
from SharsorIPCpp.PySharsorIPC import LogType
from SharsorIPCpp.PySharsorIPC import Journal

import numpy as np

# import gymnasium as gym 
    
# class IsaacSimEnv(gym.Env):
class IsaacSimEnv():

    def __init__(
        self, 
        headless: bool, 
        sim_device: int = 0, 
        enable_livestream: bool = False, 
        enable_viewport: bool = False,
        debug = False
    ) -> None:
        """ Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
        """

        self.debug = debug
                
        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.kit'
        # experience = ""
        if headless:
            
            info = f"Will run in headless mode."

            Journal.log(self.__class__.__name__,
                "__init__",
                info,
                LogType.STAT,
                throw_when_excep = True)
                            
            if enable_livestream:
                 
                experience = ""
            
            elif enable_viewport:
                
                exception = f"Using viewport is not supported yet."

                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            else:
                 
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.omnirobogym.headless.kit'
                # experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

        self._simulation_app = SimulationApp({"headless": headless,
                                            "physics_gpu": sim_device}, 
                                            experience=experience)

        info = "Using IsaacSim experience file @ " + experience

        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
        
        # carb.settings.get_settings().set("/persistent/omnihydra/useSceneGraphInstancing", True)

        if enable_livestream:

            info = "Livestream enabled"

            Journal.log(self.__class__.__name__,
                "__init__",
                info,
                LogType.STAT,
                throw_when_excep = True)

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
        self.step_counter = 0 # step counter
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

        info = "Using device: " + str(device)

        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
            
        if (sim_params is None):
            
            info = f"No sim params provided -> defaults will be used."

            Journal.log(self.__class__.__name__,
                "set_task",
                info,
                LogType.STAT,
                throw_when_excep = True)
            
            sim_params = {}

        # defaults for integration and rendering dt
        if not("physics_dt" in sim_params):
    
            sim_params["physics_dt"] = 1.0/60.0

            dt = sim_params["physics_dt"]

            info = f"Using default integration_dt of {dt} s."

            Journal.log(self.__class__.__name__,
                "set_task",
                info,
                LogType.STAT,
                throw_when_excep = True)
                        
        if not("rendering_dt" in sim_params):

            sim_params["rendering_dt"] = sim_params["physics_dt"]
            
            dt = sim_params["rendering_dt"]

            info = f"Using default rendering_dt of {dt} s."

            Journal.log(self.__class__.__name__,
                "set_task",
                info,
                LogType.STAT,
                throw_when_excep = True)

        self._world = World(
            stage_units_in_meters=1.0, 
            physics_dt=sim_params["physics_dt"], 
            rendering_dt=sim_params["rendering_dt"], # dt between rendering steps. Note: rendering means rendering a frame of 
            # the current application and not only rendering a frame to the viewports/ cameras. 
            # So UI elements of Isaac Sim will be refereshed with this dt as well if running non-headless
            backend=backend,
            device=str(device),
            physics_prim_path="/physicsScene", 
            set_defaults = False, # set to True to use the defaults settings [physics_dt = 1.0/ 60.0, 
            # stage units in meters = 0.01 (i.e in cms), rendering_dt = 1.0 / 60.0, gravity = -9.81 m / s 
            # ccd_enabled, stabilization_enabled, gpu dynamics turned off, 
            # broadcast type is MBP, solver type is TGS]
            sim_params=sim_params
        )

        self._sim_params = sim_params

        big_info = "[World] Creating task " + task.name + "\n" + \
            "use_gpu_pipeline: " + str(sim_params["use_gpu_pipeline"]) + "\n" + \
            "device: " + str(device) + "\n" +\
            "backend: " + str(backend) + "\n" +\
            "integration_dt: " + str(sim_params["physics_dt"]) + "\n" + \
            "rendering_dt: " + str(sim_params["rendering_dt"]) + "\n" \

        Journal.log(self.__class__.__name__,
            "set_task",
            big_info,
            LogType.STAT,
            throw_when_excep = True)

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

        big_info2 = "[physics context]:" + "\n" + \
            "gpu_max_rigid_contact_count: " + str(self._gpu_max_rigid_contact_count) + "\n" + \
            "gpu_max_rigid_patch_count: " + str(self._gpu_max_rigid_patch_count) + "\n" + \
            "gpu_found_lost_pairs_capacity: " + str(self._gpu_found_lost_pairs_capacity) + "\n" + \
            "gpu_found_lost_aggregate_pairs_capacity: " + str(self._gpu_found_lost_aggregate_pairs_capacity) + "\n" + \
            "gpu_total_aggregate_pairs_capacity: " + str(self._gpu_total_aggregate_pairs_capacity) + "\n" + \
            "gpu_max_soft_body_contacts: " + str(self._gpu_max_soft_body_contacts) + "\n" + \
            "gpu_max_particle_contacts: " + str(self._gpu_max_particle_contacts) + "\n" + \
            "gpu_heap_capacity: " + str(self._gpu_heap_capacity) + "\n" + \
            "gpu_temp_buffer_capacity: " + str(self._gpu_temp_buffer_capacity)
        
        Journal.log(self.__class__.__name__,
            "set_task",
            big_info2,
            LogType.STAT,
            throw_when_excep = True)

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
        
        if sim_params and "enable_viewport" in sim_params:
            self._render = sim_params["enable_viewport"]

        Journal.log(self.__class__.__name__,
            "set_task",
            "[render]: " + str(self._render),
            LogType.STAT,
            throw_when_excep = True)

        if init_sim:

            self._world.reset() # after the first reset we get get all quantities 
            # from the scene 

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

                exception = f"Cannot render '{mode}' when rendering is not enabled. Please check the provided" + \
                    "arguments to the environment class at initialization."

                Journal.log(self.__class__.__name__,
                    "__init__",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            return rgb_data[:, :, :3]
        else:
            
            # gym.Env.render(self, mode=mode)
    
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

        if self._simulation_app.is_running():
            
            self._simulation_app.close()
        
        return

    @abstractmethod
    def step(self, 
            actions = None):
                                     
        """ Basic implementation for stepping simulation"""
        
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
    def get_world(self):
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
