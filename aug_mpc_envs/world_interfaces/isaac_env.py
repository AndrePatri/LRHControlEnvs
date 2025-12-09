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
from isaacsim import SimulationApp

import carb

import os
import math

import torch
import numpy as np

from typing import Dict, List

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc_envs.utils.math_utils import quat_to_omega
from aug_mpc_envs.utils.height_grid_visualizer import HeightGridVisualizer
from aug_mpc_envs.utils.height_sensor import HeightGridSensor

from aug_mpc.world_interfaces.world_interface_base import AugMPCWorldInterfaceBase
from mpc_hive.utilities.math_utils_torch import world2base_frame,world2base_frame3D

class IsaacSimEnv(AugMPCWorldInterfaceBase):

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

    def is_running(self):
        return self._simulation_app.is_running()

    def _pre_setup(self):
        self._backend="torch"
        enable_livestream = self._env_opts["enable_livestream"]
        enable_viewport = self._env_opts["enable_viewport"]
        base_isaac_exp = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.aug_mpc_envs.kit'
        base_isaac_exp_headless = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.aug_mpc_envs.headless.kit'

        experience=base_isaac_exp
        if self._env_opts["headless"]:
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
                experience=base_isaac_exp_headless

        self._simulation_app = SimulationApp({"headless": self._env_opts["headless"]}, 
                                            experience=experience)
        # all imports depending on isaac sim kits have to be done after simulationapp
        # from omni.isaac.core.tasks.base_task import BaseTask
        self._import_isaac_pkgs()
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
            
            self._simulation_app.set_setting("/app/livestream/enabled", True)
            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")
            enable_extension("omni.services.streaming.manager")
        self._render = ((not self._env_opts["headless"]) or self._env_opts["render_to_file"]) or enable_livestream or enable_viewport

        self._record = False
        self._world = None
        self._physics_context = None
        self._scene = None
        self._task = None
        self._metadata = None    

        self._robots_art_views = {}
        self._blink_rigid_prim_views = {}
        self._robots_articulations = {}
        self._robots_geom_prim_views = {}
        self.omni_contact_sensors = {}

        self._solver_position_iteration_count=self._env_opts["solver_position_iteration_count"]
        self._solver_velocity_iteration_count=self._env_opts["solver_velocity_iteration_count"]
        self._solver_stabilization_thresh=self._env_opts["stabilization_threshold"]
        self._solver_position_iteration_counts={}
        self._solver_velocity_iteration_counts={}
        self._solver_stabilization_threshs={}
        self._robot_bodynames={}
        self._robot_n_links={}
        self._robot_n_dofs={}
        self._robot_dof_names={}
        self._distr_offset={} # decribed how robots within each env are distributed
        self._spawning_radius=self._env_opts["spawning_radius"] # [m] -> default distance between roots of robots in a single 
        self._height_sensors={}
        self._height_imgs={}
        self._height_vis={}
        self._height_vis_step={}
        self._height_vis_step={}
        self._height_vis={}
        
    def _import_isaac_pkgs(self):
        # we use global, so that we can create the simulation app inside (and so
        # access Isaac's kit) and also expose to all methods the imports
        global World, omni_kit, get_context, UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
        global enable_extension, set_camera_view, _urdf, move_prim, GridCloner, prim_utils
        global get_current_stage, Scene, ArticulationView, RigidPrimView, rep
        global OmniContactSensors, RlTerrains,OmniJntImpCntrl
        global PhysxSchema, UsdGeom
        global _sensor, _dynamic_control
        global get_prim_at_path

        from pxr import PhysxSchema, UsdGeom

        from omni.isaac.core.world import World
        from omni.usd import get_context
        from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.viewports import set_camera_view
        import omni.kit as omni_kit
        from omni.importer.urdf import _urdf
        from omni.isaac.core.utils.prims import move_prim
        from omni.isaac.cloner import GridCloner
        import omni.isaac.core.utils.prims as prim_utils
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.scenes.scene import Scene
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.prims import RigidPrimView

        import omni.replicator.core as rep

        from omni.isaac.core.utils.prims import get_prim_at_path

        from omni.isaac.sensor import _sensor

        from omni.isaac.dynamic_control import _dynamic_control

        from aug_mpc_envs.utils.contact_sensor import OmniContactSensors
        from aug_mpc_envs.utils.omni_jnt_imp_cntrl import OmniJntImpCntrl
        from aug_mpc_envs.utils.terrains import RlTerrains

    def _parse_env_opts(self):
        isaac_opts={}
        isaac_opts["envs_ns"]="/World/envs"
        isaac_opts["template_env_ns"]=isaac_opts["envs_ns"] + "/env_0"
        isaac_opts["base_linkname"]="base_link"
        isaac_opts["deduce_base_link"]=False
        isaac_opts["ground_plane_prim_path"]="/World/terrain"
        isaac_opts["physics_prim_path"]="/physicsScene"
        isaac_opts["use_gpu"]=True
        isaac_opts["use_gpu_pipeline"]=True
        isaac_opts["render_to_file"]=False
        isaac_opts["device"]="cuda"
        isaac_opts["is_fixed_base"]=False
        isaac_opts["merge_fixed_jnts"]=True
        isaac_opts["self_collide"]=True
        isaac_opts["sim_device"]="cuda" if isaac_opts["use_gpu"] else "cpu"
        isaac_opts["physics_dt"]=1e-3
        isaac_opts["rendering_dt"]=15*isaac_opts["physics_dt"]
        isaac_opts["static_friction"]=0.5
        isaac_opts["dynamic_friction"]=0.5
        isaac_opts["restitution"]=0.1
        isaac_opts["substeps"]=1 # number of physics steps to be taken for for each rendering step
        isaac_opts["gravity"] = np.array([0.0, 0.0, -9.81])
        isaac_opts["enable_scene_query_support"]=False
        isaac_opts["use_fabric"]=True# Enable/disable reading of physics buffers directly. Default is True.
        isaac_opts["replicate_physics"]=True
        # isaac_opts["worker_thread_count"]=4
        isaac_opts["solver_type"]=1 # 0: PGS, 1:TGS, defaults to TGS. PGS faster but TGS more stable
        isaac_opts["enable_stabilization"]=True
        # isaac_opts["bounce_threshold_velocity"] = 0.2
        # isaac_opts["friction_offset_threshold"] = 0.04
        # isaac_opts["friction_correlation_distance"] = 0.025
        # isaac_opts["enable_sleeping"] = True
        # Per-actor settings ( can override in actor_options )
        isaac_opts["solver_position_iteration_count"] = 4 # defaults to 4
        isaac_opts["solver_velocity_iteration_count"] = 3 # defaults to 1
        isaac_opts["sleep_threshold"] = 1e-5 # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
        # Allowed range [0, max_float).
        isaac_opts["stabilization_threshold"] = 1e-4
        # Per-body settings ( can override in actor_options )
        # isaac_opts["enable_gyroscopic_forces"] = True
        # isaac_opts["density"] = 1000 # density to be used for bodies that do not specify mass or density
        # isaac_opts["max_depenetration_velocity"] = 100.0
        # isaac_opts["solver_velocity_iteration_count"] = 1
        # GPU buffers settings
        isaac_opts["gpu_max_rigid_contact_count"] = 512 * 2048
        isaac_opts["gpu_max_rigid_patch_count"] = 80 * 2048
        isaac_opts["gpu_found_lost_pairs_capacity"] = 102400
        isaac_opts["gpu_found_lost_aggregate_pairs_capacity"] = 102400
        isaac_opts["gpu_total_aggregate_pairs_capacity"] = 102400
        # isaac_opts["gpu_max_soft_body_contacts"] = 1024 * 1024
        # isaac_opts["gpu_max_particle_contacts"] = 1024 * 1024
        # isaac_opts["gpu_heap_capacity"] = 64 * 1024 * 1024
        # isaac_opts["gpu_temp_buffer_capacity"] = 16 * 1024 * 1024
        # isaac_opts["gpu_max_num_partitions"] = 8

        isaac_opts["env_spacing"]=8.0
        isaac_opts["spawning_height"]=0.8
        isaac_opts["spawning_radius"]=1.0
        isaac_opts["spawn_height_check_half_extent"]=0.2
        isaac_opts["spawn_height_cushion"]=0.03

        isaac_opts["enable_height_vis"]=False
        isaac_opts["height_vis_radius"]=0.03
        isaac_opts["height_vis_update_period"]=1
        isaac_opts["collision_refinement_level"]=3  # increase cylinder tesselation for smoother wheel contacts
            
        isaac_opts["use_flat_ground"]=True
        isaac_opts["ground_type"]="random"
        isaac_opts["ground_size"]=300
        isaac_opts["terrain_border"]=isaac_opts["ground_size"]/2
        isaac_opts["dh_ground"]=0.03
        isaac_opts["contact_prims"] = []
        isaac_opts["sensor_radii"] = 0.1
        isaac_opts["contact_offsets"] = {}
        
        isaac_opts["enable_livestream"] = False
        isaac_opts["enable_viewport"] = False

        isaac_opts["use_diff_vels"] = False

        isaac_opts["use_random_pertub"]=False
        isaac_opts["pert_wrenches_max_duration"]=0.2
        isaac_opts["pert_wrenches_min_duration"]=0.05

        isaac_opts["pert_wrenches_weight_factor"]=0.3 # 0.5 -> 50% of full robot weight
        isaac_opts["max_lin_impulse_norm"]=isaac_opts["pert_wrenches_weight_factor"]*isaac_opts["pert_wrenches_max_duration"]
        isaac_opts["max_ang_impulse_lever"]=0.2 # [m]
        isaac_opts["max_ang_impulse_norm"]=isaac_opts["max_lin_impulse_norm"]*isaac_opts["max_ang_impulse_lever"]
        isaac_opts["lin_pert_max_wrt_weight"]=0.5
        isaac_opts["ang_pert_max_wrt_weight"]=0.1
        
        isaac_opts["pert_wrenches_rate"]=15.0 # on avergare 1 pert every pert_wrenches_rate seconds

        isaac_opts.update(self._env_opts) # update defaults with provided opts
        isaac_opts["rendering_freq"]=int(isaac_opts["rendering_dt"]/isaac_opts["physics_dt"])
        # isaac_opts["rendering_dt"]=isaac_opts["physics_dt"] # forcing rendering_dt==physics_dt
        # for some mystic reason simulation is infuenced by the rendering dt (why ??????)
        
        # modify things
        isaac_opts["cloning_offset"] = np.array([[0.0, 0.0, isaac_opts["spawning_height"]]]*self._num_envs)
        if not isaac_opts["use_gpu"]: # don't use GPU at all
            isaac_opts["use_gpu_pipeline"]=False
            isaac_opts["device"]="cpu"
            isaac_opts["sim_device"]="cpu"
        else: # use GPU
            if isaac_opts["use_gpu_pipeline"]:
                isaac_opts["device"]="cuda"
                isaac_opts["sim_device"]="cuda"
            else: # cpu pipeline
                isaac_opts["device"]="cpu"
                isaac_opts["sim_device"]="cpu"
                isaac_opts["use_gpu"]=False
        # isaac_opts["sim_device"]=isaac_opts["device"]
        # overwrite env opts in case some sim params were missing
        self._env_opts=isaac_opts

        # update device flag based on sim opts
        self._device=isaac_opts["device"]
        self._use_gpu=isaac_opts["use_gpu"]

    def _calc_robot_distrib(self):

        import math
        # we distribute robots in a single env. along the 
        # circumference of a circle of given radius
        n_robots = len(self._robot_names)
        offset_baseangle = 2 * math.pi / n_robots
        for i in range(n_robots):
            offset_angle = offset_baseangle * (i + 1) 
            robot_offset_wrt_center = torch.tensor([self._spawning_radius * math.cos(offset_angle), 
                                            self._spawning_radius * math.sin(offset_angle), 0], 
                    device=self._device, 
                    dtype=self._dtype)
            # list with n references to the original tensor
            tensor_list = [robot_offset_wrt_center] * self._num_envs
            self._distr_offset[self._robot_names[i]] = torch.stack(tensor_list, dim=0)

    def _init_world(self):

        self._cloner = GridCloner(spacing=self._env_opts["env_spacing"])
        self._cloner.define_base_env(self._env_opts["envs_ns"])
        prim_utils.define_prim(self._env_opts["template_env_ns"])
        self._envs_prim_paths = self._cloner.generate_paths(self._env_opts["envs_ns"] + "/env", 
                                                self._num_envs)
                
        # parse device based on sim_param settings
        
        info = "Using sim device: " + str(self._env_opts["sim_device"])
        Journal.log(self.__class__.__name__,
            "__init__",
            info,
            LogType.STAT,
            throw_when_excep = True)
        
        self._world = World(
            physics_dt=self._env_opts["physics_dt"], 
            rendering_dt=self._env_opts["physics_dt"],  # == physics dt (rendering is actually done manually by this env)
            backend=self._backend,
            device=str(self._env_opts["sim_device"]),
            physics_prim_path=self._env_opts["physics_prim_path"], 
            set_defaults = False, # set to True to use the defaults settings [physics_dt = 1.0/ 60.0, 
            # stage units in meters = 0.01 (i.e in cms), rendering_dt = 1.0 / 60.0, gravity = -9.81 m / s 
            # ccd_enabled, stabilization_enabled, gpu dynamics turned off, 
            # broadcast type is MBP, solver type is TGS]
            sim_params=self._env_opts
        )

        big_info = "[World] Creating Isaac simulation " + self._name + "\n" + \
            "use_gpu_pipeline: " + str(self._env_opts["use_gpu_pipeline"]) + "\n" + \
            "device: " + str(self._env_opts["sim_device"]) + "\n" +\
            "backend: " + str(self._backend) + "\n" +\
            "integration_dt: " + str(self.physics_dt()) + "\n" + \
            "rendering_dt: " + str(self.rendering_dt()) + "\n" 
        Journal.log(self.__class__.__name__,
            "_init_world",
            big_info,
            LogType.STAT,
            throw_when_excep = True)
        
        # we get the physics context to expose additional low-level ##
        # settings of the simulation
        self._physics_context = self._world.get_physics_context() 
        self._physics_scene_path = self._physics_context.prim_path
        # self._physics_context.enable_gpu_dynamics(True)
        self._physics_context.enable_stablization(True)
        self._physics_scene_prim = self._physics_context.get_current_physics_scene_prim()
        self._solver_type = self._physics_context.get_solver_type()

        if "gpu_max_rigid_contact_count" in self._env_opts:
                self._physics_context.set_gpu_max_rigid_contact_count(self._env_opts["gpu_max_rigid_contact_count"])
        if "gpu_max_rigid_patch_count" in self._env_opts:
                self._physics_context.set_gpu_max_rigid_patch_count(self._env_opts["gpu_max_rigid_patch_count"])
        if "gpu_found_lost_pairs_capacity" in self._env_opts:
                self._physics_context.set_gpu_found_lost_pairs_capacity(self._env_opts["gpu_found_lost_pairs_capacity"])
        if "gpu_found_lost_aggregate_pairs_capacity" in self._env_opts:
                self._physics_context.set_gpu_found_lost_aggregate_pairs_capacity(self._env_opts["gpu_found_lost_aggregate_pairs_capacity"])
        if "gpu_total_aggregate_pairs_capacity" in self._env_opts:
                self._physics_context.set_gpu_total_aggregate_pairs_capacity(self._env_opts["gpu_total_aggregate_pairs_capacity"])
        if "gpu_max_soft_body_contacts" in self._env_opts:
                self._physics_context.set_gpu_max_soft_body_contacts(self._env_opts["gpu_max_soft_body_contacts"])
        if "gpu_max_particle_contacts" in self._env_opts:
                self._physics_context.set_gpu_max_particle_contacts(self._env_opts["gpu_max_particle_contacts"])
        if "gpu_heap_capacity" in self._env_opts:
                self._physics_context.set_gpu_heap_capacity(self._env_opts["gpu_heap_capacity"])
        if "gpu_temp_buffer_capacity" in self._env_opts:
                self._physics_context.set_gpu_temp_buffer_capacity(self._env_opts["gpu_temp_buffer_capacity"])
        if "gpu_max_num_partitions" in self._env_opts:
                self._physics_context.set_gpu_max_num_partitions(self._env_opts["gpu_max_num_partitions"])

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
            "gpu_temp_buffer_capacity: " + str(self._gpu_temp_buffer_capacity) + "\n" + \
            "use_gpu_sim: " + str(self._world.get_physics_context().use_gpu_sim) + "\n" + \
            "use_gpu_pipeline: " + str(self._world.get_physics_context().use_gpu_pipeline) + "\n" + \
            "use_fabric: " + str(self._world.get_physics_context().use_fabric) + "\n" + \
            "world device: " + str(self._world.get_physics_context().device) + "\n" + \
            "physics context device: " + str(self._world.get_physics_context().device) + "\n" 

        Journal.log(self.__class__.__name__,
            "set_task",
            big_info2,
            LogType.STAT,
            throw_when_excep = True)

        self._scene = self._world.scene
        self._physics_context = self._world.get_physics_context()

        self._stage = get_context().get_stage()

        distantLight = UsdLux.DistantLight.Define(self._stage, Sdf.Path("/World/DistantLight"))
        distantLight.CreateIntensityAttr(500)

        self._configure_scene()

        # if "enable_viewport" in sim_params:
        #     self._render = sim_params["enable_viewport"]

    def _get_baselink_candidate(self, 
            robot_name: str):
        
        stage=get_current_stage()
        all_prims = [prim.GetPath().pathString for prim in stage.Traverse()]
        filtered_prims = [prim for prim in all_prims if f"/{robot_name}/" in prim]

        matching=min(filtered_prims, key=len) if filtered_prims else None

        return matching.split('/')[-1]

    def _configure_scene(self):
            
        # environment 
        self._fix_base = [self._env_opts["is_fixed_base"]] * len(self._robot_names)
        self._self_collide = [self._env_opts["self_collide"]]  * len(self._robot_names)
        self._merge_fixed = [self._env_opts["merge_fixed_jnts"]] * len(self._robot_names)
        
        Journal.log(self.__class__.__name__,
            "_configure_scene",
            "cloning environments...",
            LogType.STAT,
            throw_when_excep = True)
        
        self._ground_plane_prim_paths=[]
        self._ground_plane=None
        self.terrain_generator = None
        if not self._env_opts["use_flat_ground"]:
            # ensure terrain is large enough to contain all env clones
            spacing = float(self._env_opts["env_spacing"])
            num_envs = self._num_envs
            num_per_row = max(1, int(math.sqrt(num_envs)))
            num_rows = int(math.ceil(num_envs / num_per_row))
            num_cols = int(math.ceil(num_envs / num_rows))
            row_offset = 0.5 * spacing * (num_rows - 1)
            col_offset = 0.5 * spacing * (num_cols - 1)
            margin = spacing  # leave a full spacing as border cushion
            required_size = 2.0 * (max(row_offset, col_offset) + margin)

            if required_size > self._env_opts["ground_size"]:
                old_size = self._env_opts["ground_size"]
                self._env_opts["ground_size"] = required_size
                self._env_opts["terrain_border"] = self._env_opts["ground_size"] / 2.0
                Journal.log(self.__class__.__name__,
                            "_configure_scene",
                            f"Ground size increased from {old_size} to {required_size} to fit {num_envs} envs (spacing {spacing}).",
                            LogType.WARN,
                            throw_when_excep = True)

            min_height=-self._env_opts["dh_ground"]
            max_height=self._env_opts["dh_ground"]
            step=max_height-min_height
            if self._env_opts["ground_type"]=="random":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_random_unif"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_random_uniform_terrain(terrain_size=self._env_opts["ground_size"], 
                    min_height=min_height,
                    max_height=max_height,
                    step=step,
                    position=np.array([0.0, 0.0,0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"])
            elif self._env_opts["ground_type"]=="random_patches":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_random_unif_patches"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_random_patched_terrain(terrain_size=self._env_opts["ground_size"], 
                    min_height=min_height,
                    max_height=max_height,
                    step=step,
                    position=np.array([0.0, 0.0,0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"],
                    patch_ratio=0.8,
                    patch_size=10
                    )
            elif self._env_opts["ground_type"]=="slopes":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_slopes"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_sloped_terrain(terrain_size=self._env_opts["ground_size"], 
                    slope=-0.5,
                    position=np.array([0.0, 0.0,0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"]
                    )
            elif self._env_opts["ground_type"]=="stairs":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_stairs"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_stairs_terrain(terrain_size=self._env_opts["ground_size"],      
                    position=np.array([0.0, 0.0,0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"],
                    )
            elif self._env_opts["ground_type"]=="stepup":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_stepup"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_stepup_terrain(
                    terrain_size=self._env_opts["ground_size"], 
                    stairs_ratio=0.3,
                    min_steps=1,
                    max_steps=1,
                    pyramid_platform_size=15.0,
                    position=np.array([0.0, 0.0, 0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"],
                    step_height=0.15
                    )
            elif self._env_opts["ground_type"]=="stepup_prim":
                terrain_prim_path=self._env_opts["ground_plane_prim_path"]+"_stepup_prim"
                self._ground_plane_prim_paths.append(terrain_prim_path)
                self.terrain_generator = RlTerrains(get_current_stage(), prim_path=terrain_prim_path)
                self._ground_plane=self.terrain_generator.create_stepup_prim_terrain(
                    terrain_size=self._env_opts["ground_size"], 
                    stairs_ratio=0.5,
                    platform_size=5.0,
                    step_height=0.10,
                    position=np.array([0.0, 0.0, 0.0]), 
                    static_friction=self._env_opts["static_friction"], 
                    dynamic_friction=self._env_opts["dynamic_friction"], 
                    restitution=self._env_opts["restitution"],
                    )
            else:
                ground_type=self._env_opts["ground_type"]
                Journal.log(self.__class__.__name__,
                    "_configure_scene",
                    f"Terrain type {ground_type} not supported. Will default to flat ground.",
                    LogType.EXCEP,
                    throw_when_excep = True)
                
            # add offsets to intial height depending on the terrain heightmap
            if self.terrain_generator is not None:
                stage = get_current_stage()
                up_axis = UsdGeom.GetStageUpAxis(stage)

                spacing = self._env_opts["env_spacing"]
                num_envs = self._num_envs

                num_per_row = max(1, int(np.sqrt(num_envs)))
                num_rows = int(np.ceil(num_envs / num_per_row))
                num_cols = int(np.ceil(num_envs / num_rows))

                row_offset = 0.5 * spacing * (num_rows - 1)
                col_offset = 0.5 * spacing * (num_cols - 1)

                offsets = np.array(self._env_opts["cloning_offset"], dtype=float)

                for env_idx in range(num_envs):
                    row = env_idx // num_cols
                    col = env_idx % num_cols
                    x = row_offset - row * spacing
                    y = col * spacing - col_offset

                    half_extent = self._env_opts.get("spawn_height_check_half_extent", 0.3)
                    if up_axis == UsdGeom.Tokens.z:
                        height = self.terrain_generator.get_max_height_in_rect(x, y, half_extent=half_extent)
                    else:
                        height = self.terrain_generator.get_max_height_in_rect(x, y, half_extent=half_extent)

                    offsets[env_idx][2] += height + self._env_opts["spawn_height_cushion"]

                self._env_opts["cloning_offset"] = offsets

        else:
            defaul_prim_path=self._env_opts["ground_plane_prim_path"]+"_default"
            self._ground_plane_prim_paths.append(defaul_prim_path)
            self._ground_plane=self._scene.add_default_ground_plane(z_position=0, 
                name="terrain", 
                prim_path=defaul_prim_path, 
                static_friction=self._env_opts["static_friction"], 
                dynamic_friction=self._env_opts["dynamic_friction"], 
                restitution=self._env_opts["restitution"])
            self._terrain_collision=None
            
            physics_material=UsdPhysics.MaterialAPI.Apply(self._ground_plane.prim)
            physics_material.CreateDynamicFrictionAttr(self._env_opts["dynamic_friction"])
            physics_material.CreateStaticFrictionAttr(self._env_opts["static_friction"])
            physics_material.CreateRestitutionAttr(self._env_opts["restitution"])
            # self._ground_plane.apply_physics_material(physics_material)

        self._terrain_collision=PhysxSchema.PhysxCollisionAPI.Get(get_current_stage(), self._ground_plane.prim_path)
        self._terrain_material=UsdPhysics.MaterialAPI.Get(get_current_stage(), self._ground_plane.prim_path)
        self._terrain_physix_material=PhysxSchema.PhysxMaterialAPI.Get(get_current_stage(), self._ground_plane.prim_path)

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
            self._import_urdf(robot_name, 
                            fix_base=fix_base, 
                            self_collide=self_collide, 
                            merge_fixed=merge_fixed)
            
            self._cloner.clone(
                source_prim_path=self._env_opts["template_env_ns"],
                prim_paths=self._envs_prim_paths,
                replicate_physics=self._env_opts["replicate_physics"],
                position_offsets=self._env_opts["cloning_offset"]
            ) # we can clone the environment in which all the robos are

            base_link_name=self._env_opts["base_linkname"]
            if self._env_opts["deduce_base_link"]:
                base_link_name=self._get_baselink_candidate(robot_name=robot_name)

            self._robots_art_views[robot_name] = ArticulationView(name = robot_name + "ArtView",
                                                        prim_paths_expr = self._env_opts["envs_ns"] + "/env_.*"+ "/" + robot_name + "/" + base_link_name, 
                                                        reset_xform_properties=False)
            self._robots_articulations[robot_name] = self._scene.add(self._robots_art_views[robot_name])

            # height grid sensor (terrain may be None if using flat ground)
            self._height_sensors[robot_name] = HeightGridSensor(
                terrain_utils=self.terrain_generator if not self._env_opts["use_flat_ground"] else None,
                grid_size=int(self._env_opts["height_sensor_pixels"]),
                resolution=float(self._env_opts["height_sensor_resolution"]),
                n_envs=self._num_envs,
                device=self._device,
                dtype=self._dtype)
            # ensure shared-data flags are set if a height sensor is active
            self._env_opts["height_sensor_pixels"] = int(self._env_opts["height_sensor_pixels"])
            self._env_opts["height_sensor_resolution"] = float(self._env_opts["height_sensor_resolution"])
            self._enable_height_shared = True
            self._height_sensor_pixels = self._env_opts["height_sensor_pixels"]
            self._height_sensor_resolution = self._env_opts["height_sensor_resolution"]
            self._height_vis_step[robot_name] = 0
            if self._env_opts["enable_height_vis"]:
                self._height_vis[robot_name] = HeightGridVisualizer(
                    robot_name=robot_name,
                    num_envs=self._num_envs,
                    grid_size=int(self._env_opts["height_sensor_pixels"]),
                    resolution=float(self._env_opts["height_sensor_resolution"]),
                    marker_radius=float(self._env_opts["height_vis_radius"]),
                    device=self._device,
                    dtype=self._dtype)

            self._blink_rigid_prim_views[robot_name] = RigidPrimView(prim_paths_expr=self._env_opts["envs_ns"] + "/env_.*"+ "/" + robot_name + "/" + base_link_name,
                                                    name = robot_name + "RigidPrimView") # base link prim views
            self._scene.add(self._blink_rigid_prim_views[robot_name]) # need to add so it is properly initialized when resetting world 

            # self._robots_geom_prim_views[robot_name] = GeometryPrimView(name = robot_name + "GeomView",
            #                                                 prim_paths_expr = self._env_ns + "/env*"+ "/" + robot_name,
            #                                                 # prepare_contact_sensors = True
            #                                             )
            # self._robots_geom_prim_views[robot_name].apply_collision_apis() # to be able to apply contact sensors

            # init contact sensors
            self._init_contact_sensors(robot_name=robot_name) # IMPORTANT: this has to be called
            # after calling the clone() method and initializing articulation views!!

        # self._set_contact_links_material(prim_path="/World/envs/env_0/kyon_no_wheels/lower_leg_1/collisions/mesh_1")
        # self._set_contact_links_material(prim_path="/World/envs/env_0/kyon_no_wheels/lower_leg_2/collisions/mesh_1")
        # self._set_contact_links_material(prim_path="/World/envs/env_0/kyon_no_wheels/lower_leg_3/collisions/mesh_1")
        # self._set_contact_links_material(prim_path="/World/envs/env_0/kyon_no_wheels/lower_leg_4/collisions/mesh_1")

        # filter collisions between default ground plane and custom terrains
        # self._cloner.filter_collisions(physicsscene_path = self._physics_context.prim_path,
        #     collision_root_path = "/World/terrain_collisions", 
        #     prim_paths=[self._ground_plane_prim_paths[1]], 
        #     global_paths=[self._ground_plane_prim_paths[0]]
        #     )

        # delete_prim(self._env_opts["ground_plane_prim_path"] + "/SphereLight") # we remove the default spherical light
        
        # set default camera viewport position and target
        camera_position=[4.2, 4.2, 1.5]
        camera_target=[0, 0, 0]
        self._set_initial_camera_params(camera_position=camera_position,
                camera_target=camera_target)
        if self._env_opts["render_to_file"]:
            self._render_camera = rep.create.camera(focal_length=12, 
                                name='rendering_camera',
                                clipping_range = (1, 40), 
                                position = camera_position,
                                look_at = camera_target)
            self._render_product  = rep.create.render_product(self._render_camera, 
                        (1280, 720), name='rendering_camera')
            self._render_writer = rep.WriterRegistry.get("BasicWriter")
            from datetime import datetime
            timestamp = datetime.now().strftime("h%H_m%M_s%S_%d_%m_%Y")
            self._render_writer.initialize(output_dir=f"/tmp/IsaacRenderings/{timestamp}", 
                        rgb=True)
            self._render_writer.attach([self._render_product])

        self.apply_collision_filters(self._physics_context.prim_path, 
                            "/World/collisions")

        self._reset_sim()

        self._fill_robot_info_from_world() 
        # initializes robot state data
        
        # update solver options 
        self._update_art_solver_options() 
        self._get_solver_info() # get again solver option before printing everything
        self._print_envs_info() # debug print

        # for n in range(self._n_init_steps): # run some initialization steps
        #     self._step_sim()

        self._init_robots_state()

        self.scene_setup_completed = True
        
        Journal.log(self.__class__.__name__,
            "set_up_scene",
            "finished scene setup...",
            LogType.STAT,
            throw_when_excep = True)
        
        self._is = _sensor.acquire_imu_sensor_interface()
        self._dyn_control=_dynamic_control.acquire_dynamic_control_interface()

    def _set_contact_links_material(self, prim_path: str):
        prim=get_prim_at_path(prim_path)
        physics_material=UsdPhysics.MaterialAPI.Apply(prim)
        physics_material.CreateDynamicFrictionAttr(0)
        physics_material.CreateStaticFrictionAttr(0)
        physics_material.CreateRestitutionAttr(1.0)
        physxMaterialAPI=PhysxSchema.PhysxMaterialAPI.Apply(prim)
        physxMaterialAPI.CreateFrictionCombineModeAttr().Set("multiply") # average, min, multiply, max 
        physxMaterialAPI.CreateRestitutionCombineModeAttr().Set("multiply")

    def _is_link(self, prim):
        return prim.GetTypeName() == 'Xform' 

    def _is_joint(self, prim):
        return prim.GetTypeName() == 'PhysicsRevoluteJoint'
    
    def _create_collision_group(self, group_path, link_paths):
        """
        Create a collision group under the given group_path that contains the links.
        Args:
            group_path (str): Path to create the collision group.
            link_paths (List[str]): List of link paths to include in this group.
        """
        collision_group = Sdf.PrimSpec(
            self._stage.GetRootLayer().GetPrimAtPath(group_path),
            group_path.split("/")[-1],
            Sdf.SpecifierDef,
            "PhysicsCollisionGroup"
        )
        # Add the links to the collision group
        for link_path in link_paths:
            includes_rel = Sdf.RelationshipSpec(collision_group, "collection:colliders:includes", False)
            includes_rel.targetPathList.Append(link_path)

    def _add_collision_filter(self, group_path, link1, link2):
        """
        Filters collision between two successive links.
        
        Args:
            group_path (str): Path of the collision group.
            link1 (str): Path of the first link.
            link2 (str): Path of the second link.
        """
        # Create a relationship to filter collisions between the two links
        filtered_groups = Sdf.RelationshipSpec(
            self._stage.GetPrimAtPath(group_path),
            "physics:filteredGroups",
            False
        )
        filtered_groups.targetPathList.Append(link1)
        filtered_groups.targetPathList.Append(link2)

    def _render_sim(self, mode="human"):

        if mode == "human":
            self._world.render()
            # optional height grid visualization
            if self._env_opts.get("enable_height_vis", False):
                for robot_name, vis in self._height_vis.items():
                    # use latest stored states
                    if robot_name not in self._height_imgs or robot_name not in self._height_sensors:
                        continue
                    heights = self._height_imgs.get(robot_name, None)
                    if heights is None or heights.numel() == 0:
                        continue
                    pos_src = self._root_p.get(robot_name, None)
                    quat_src = self._root_q.get(robot_name, None)
                    if pos_src is None or quat_src is None:
                        continue
                    step = self._height_vis_step.get(robot_name, 0)
                    period = max(1, int(self._env_opts.get("height_vis_update_period", 1)))
                    if step % period == 0:
                        try:
                            vis.update(
                                base_positions=pos_src,
                                base_quats=quat_src,
                                heights=heights)
                        except Exception as exc:
                            print(f"[height_vis] update failed for {robot_name}: {exc}")
                    self._height_vis_step[robot_name] = step + 1
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
            return None

    def _create_viewport_render_product(self, resolution=(1280, 720)):
        """Create a render product of the viewport for rendering."""

        try:

            # create render product
            self._render_product = rep.create.render_product("/OmniverseKit_Persp", resolution)
            # create rgb annotator -- used to read data from the render product
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self._rgb_annotator.attach([self._render_product])
            self._record = True
        except Exception as e:
            carb.log_info("omni.replicator.core could not be imported. Skipping creation of render product.")
            carb.log_info(str(e))

    def _close(self):
        if self._simulation_app.is_running():
            self._simulation_app.close()
    
    def _step_world(self): 
        self._world.step(render=False, step_sim=True)

        if (self._render) and (self.step_counter%self._env_opts["rendering_freq"]==0):
            # if self._env_opts["render_to_file"]:
            #     rep.orchestrator.step()
            self._render_sim() # manually trigger rendering (World.step(render=True) for some reason 
            # will step the simulation for a dt==rendering_dt)

    def _generate_jnt_imp_control(self, robot_name: str):
        
        jnt_imp_controller = OmniJntImpCntrl(articulation=self._robots_articulations[robot_name],
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
        self._world.reset(soft=False)
    
    def _reset_state(self,
        robot_name: str,
        env_indxs: torch.Tensor = None,
        randomize: bool = False):

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

            if randomize:
                self._randomize_yaw(robot_name=robot_name,env_indxs=None)

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
        self._read_root_state_from_robot(env_indxs=env_indxs, 
            robot_name=robot_name)
        self._read_jnts_state_from_robot(env_indxs=env_indxs,
            robot_name=robot_name)
    
    def _process_perturbations(self):

        # Iterate over each robot view
        for i in range(len(self._robot_names)):
            robot_name = self._robot_names[i]

            # Pre-fetch views for code clarity (references, not copies)
            active = self._pert_active[robot_name]
            steps_rem = self._pert_steps_remaining[robot_name]
            forces_world = self._pert_forces_world[robot_name]
            torques_world = self._pert_torques_world[robot_name]

            # --- 1. Update Active Counters (In-Place) ---
            if active.any():
                # In-place subtraction
                steps_rem[active] -= 1

            # --- 2. Reset Finished Perturbations (In-Place) ---
            # Logic: Active AND (Steps <= 0)
            # Note: Creating 'newly_ended' boolean mask is a tiny unavoidable allocation
            newly_ended = active & (steps_rem <= 0)

            if newly_ended.any():
                # Use masked_fill_ for in-place zeroing
                active.masked_fill_(newly_ended, False)
                forces_world[newly_ended, :]=0
                torques_world[newly_ended, :]=0
                steps_rem.masked_fill_(newly_ended, 0)

            # --- 3. Trigger New Perturbations ---

            # Reuse scratch buffer for probability check
            # Assumes self._pert_scratch is (num_envs, 1) pre-allocated
            self._pert_scratch[robot_name].uniform_(0.0, 1.0)

            # Check probs against threshold (Broadcasting (N,1) vs scalar)
            # Flatten scratch to (N,) to match 'active' mask
            trigger_mask = (self._pert_scratch[robot_name].flatten() < self._pert_wrenches_prob) & (~active)

            if trigger_mask.any():

                # Cache weights (references)
                weight = self._weights[robot_name]

                #   we now treat the configured "max_*_impulse_*" as the maximum **impulse** (N·s or N·m·s),
                #   and convert impulse -> force/torque by dividing by the sampled duration (seconds).
                #   Use per-robot weight scaling as before.
                lin_impulse_max = self._env_opts["max_lin_impulse_norm"] * weight
                ang_impulse_max = self._env_opts["max_ang_impulse_norm"] * weight

                # --- Force (Impulse) Direction Generation (Reuse _pert_lindir buffer) ---
                lindir = self._pert_lindir[robot_name]  # (N, 3)

                # 1. Fill with Standard Normal noise in-place
                lindir.normal_()

                # 2. Normalize in-place
                norms = torch.norm(lindir, dim=1, keepdim=True).clamp_min_(1e-6)
                lindir.div_(norms)

                # 3. Sample linear impulse magnitudes (reuse scratch)
                # scratch has shape (N,1) - uniform [0,1]
                self._pert_scratch[robot_name].uniform_(0.0, 1.0)
                # impulse vectors = unit_dir * (rand * lin_impulse_max)

                lindir.mul_(self._pert_scratch[robot_name] * lin_impulse_max)  # now contains linear impulses (N,3)

                # --- Angular (Impulse) Direction Generation (Reuse _pert_angdir buffer) ---
                angdir = self._pert_angdir[robot_name]  # (N, 3)

                # 1. Fill with Standard Normal noise
                angdir.normal_()

                # 2. Normalize
                norms = torch.norm(angdir, dim=1, keepdim=True).clamp_min_(1e-6)
                angdir.div_(norms)

                # 3. Sample angular impulse magnitudes (reuse scratch)
                self._pert_scratch[robot_name].uniform_(0.0, 1.0)
                angdir.mul_(self._pert_scratch[robot_name] * ang_impulse_max)  # now contains angular impulses (N,3)

                # --- Duration Generation (Reuse _pert_durations) ---
                # Keep integer steps sampling (same shape/device/dtype)
                self._pert_durations[robot_name] = torch.randint_like(
                    self._pert_durations[robot_name],
                    low=self._pert_min_steps,
                    high=self._pert_max_steps + 1
                )

                # --- convert to float
                duration_steps = self._pert_durations[robot_name].to(dtype=lindir.dtype, device=lindir.device)
                # duration in seconds (shape (N,))
                duration_seconds = duration_steps * self.physics_dt()
                # avoid divide-by-zero
                duration_seconds = duration_seconds.clamp_min_(1e-6)

                # compute per-step forces/torques = impulse / duration_seconds
                # lindir currently holds linear impulses, angdir holds angular impulses
                forces_to_apply = lindir / duration_seconds
                torques_to_apply = angdir / duration_seconds

                # --- Update State Buffers ---
                # Use boolean indexing to scatter only triggered values
                active[trigger_mask] = True
                steps_rem[trigger_mask] = self._pert_durations[robot_name][trigger_mask, :].flatten()
                forces_world[trigger_mask, :] = forces_to_apply[trigger_mask, :]
                torques_world[trigger_mask, :] = torques_to_apply[trigger_mask, :]

            # --- 4. Apply Wrenches (Vectorized) ---
            # Only call API if there are active perturbations to minimize overhead
            
            forces_world[~active, :]=0
            torques_world[~active, :]=0

            self._blink_rigid_prim_views[robot_name].apply_forces_and_torques_at_pos(
                forces=forces_world,
                torques=torques_world,
                positions=None, # body frame origin
                is_global=True
                )
                        
    def _pre_step(self):
        
        if self._env_opts["use_random_pertub"]:
            self._process_perturbations()

        super()._pre_step()

    def _pre_step_db(self):
        
        if self._env_opts["use_random_pertub"]:
            self._process_perturbations()

        super()._pre_step_db()

    def _update_contact_state(self, 
            robot_name: str, 
            env_indxs: torch.Tensor = None):
        
        super()._update_contact_state(robot_name, env_indxs)
        
        if self._env_opts["use_random_pertub"]:
            # write APPLIED perturbations to root wrench (mainly for debug)
            self.cluster_servers[robot_name].get_state().contact_wrenches_root.set(data=self._pert_forces_world[robot_name][env_indxs, :], 
                                    data_type="f",
                                    contact_name="root", 
                                    robot_idxs = env_indxs, 
                                    gpu=self._use_gpu)
            self.cluster_servers[robot_name].get_state().contact_wrenches_root.set(data=self._pert_torques_world[robot_name][env_indxs, :],
                                    data_type="t",
                                    contact_name="root", 
                                    robot_idxs = env_indxs, 
                                    gpu=self._use_gpu)
        
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

        if success:
            Journal.log(self.__class__.__name__,
                "_import_urdf",
                "Successfully importedf URDF into IsaacSim",
                LogType.STAT)
        else:
            Journal.log(self.__class__.__name__,
                "_import_urdf",
                "Failed to import URDF into IsaacSim",
                LogType.EXCEP,
                throw_when_excep = True)
        
        # moving default prim to base prim path for cloning
        move_prim(robot_prim_path_default, # from
                robot_base_prim_path) # to
        
        robot_base_prim = prim_utils.get_prim_at_path(robot_base_prim_path)
        print("Imported robot URDF: \n", prim_utils.get_prim_children(robot_base_prim))

        # improve collision tesselation for cylinders (e.g., wheels) if requested
        # self._apply_collision_refinement(robot_base_prim_path,
        #                                  self._env_opts["collision_refinement_level"])

        return success

    def _apply_collision_refinement(self, robot_base_prim_path: str, refinement_level: int):
        """Set refinement level on collision cylinders to avoid coarse faceting."""
        if refinement_level is None:
            return
        stage = get_current_stage()
        coll_prefix = robot_base_prim_path + "/collisions"
        count = 0
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            path_str = prim.GetPath().pathString
            if not path_str.startswith(coll_prefix):
                continue
            if prim.GetTypeName() == "Cylinder":
                attr = prim.GetAttribute("refinementLevel")
                if not attr.IsValid():
                    attr = prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
                attr.Set(int(refinement_level))
                count += 1
        Journal.log(self.__class__.__name__,
            "_apply_collision_refinement",
            f"Applied refinement level {refinement_level} to {count} cylinder collision prims under {coll_prefix}",
            LogType.STAT)

    def apply_collision_filters(self, 
                                physicscene_path: str, 
                                coll_root_path: str):

        self._cloner.filter_collisions(physicsscene_path = physicscene_path,
                                collision_root_path = coll_root_path, 
                                prim_paths=self._envs_prim_paths, 
                                global_paths=self._ground_plane_prim_paths # can collide with these prims
                                )

    def _read_root_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):

        self._get_root_state(numerical_diff=self._env_opts["use_diff_vels"],
                env_indxs=env_indxs,
                robot_name=robot_name)
        
        # height grid sensor readout
        if robot_name in self._height_sensors:
            pos_src = self._root_p[robot_name] if env_indxs is None else self._root_p[robot_name][env_indxs]
            quat_src = self._root_q[robot_name] if env_indxs is None else self._root_q[robot_name][env_indxs]
            heights = self._height_sensors[robot_name].read(pos_src, quat_src)
            if self._env_opts["use_flat_ground"]:
                heights.zero_()
            if env_indxs is None:
                self._height_imgs[robot_name] = heights
            else:
                # clone to avoid overlapping write/read views
                self._height_imgs[robot_name][env_indxs] = heights.clone()

            # print("height image")
            # print(self._height_imgs[robot_name][0, :, : ])

    def _read_jnts_state_from_robot(self,
        robot_name: str,
        env_indxs: torch.Tensor = None):
        
        self._get_robots_jnt_state(
            numerical_diff=self._env_opts["use_diff_vels"],
            env_indxs=env_indxs,
            robot_name=robot_name)
             
    def _get_root_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False,
        base_loc: bool = True):
        
        # reading = self._is.get_sensor_reading("/World/Cube/Imu_Sensor", 
        #     use_latest_data = True)

        dt=self._cluster_dt[robot_name] # getting diff state always at cluster rate

        # measurements from simulator are in world frame 
        if env_indxs is not None:
        
            pose = self._robots_art_views[robot_name].get_world_poses( 
                                            clone = True,
                                            indices=env_indxs) # tuple: (pos, quat), quat is [w, i, j, k] in Isaac4.2
            
            self._root_p[robot_name][env_indxs, :] = pose[0] 

            going_to_fly=self._root_p[robot_name][env_indxs, 0:2]>(self._env_opts["terrain_border"]-0.1)
            if going_to_fly.any():
                flying=going_to_fly.sum().item()
                warn = f"N. {flying} robots ({robot_name}) are about to go out of the terrain!!"
                Journal.log(self.__class__.__name__,
                    "_get_root_state",
                    warn,
                    LogType.WARN,
                    throw_when_excep = True)
            self._root_q[robot_name][env_indxs, :] = pose[1] # root orientation
            if not numerical_diff:
                # we get velocities from the simulation. This is not good since 
                # these can actually represent artifacts which do not have physical meaning.
                # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                self._root_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True,
                                            indices=env_indxs) # root lin. velocity               
                self._root_omega[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True,
                                            indices=env_indxs) # root ang. velocity
                
                # for now obtain root a numerically
                self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
                                                self._root_v_prev[robot_name][env_indxs, :]) / dt 
                self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                                self._root_omega_prev[robot_name][env_indxs, :]) / dt 
                
                self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
                self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]
            else:
                # differentiate numerically
                self._root_v[robot_name][env_indxs, :] = (self._root_p[robot_name][env_indxs, :] - \
                                                self._root_p_prev[robot_name][env_indxs, :]) / dt 
                self._root_omega[robot_name][env_indxs, :] = quat_to_omega( 
                                                            self._root_q_prev[robot_name][env_indxs, :], 
                                                            self._root_q[robot_name][env_indxs, :],
                                                            dt)

                self._root_a[robot_name][env_indxs, :] = (self._root_v[robot_name][env_indxs, :] - \
                                                self._root_v_prev[robot_name][env_indxs, :]) / dt 
                self._root_alpha[robot_name][env_indxs, :] = (self._root_omega[robot_name][env_indxs, :] - \
                                                self._root_omega_prev[robot_name][env_indxs, :]) / dt 

                # update "previous" data for numerical differentiation
                self._root_p_prev[robot_name][env_indxs, :] = self._root_p[robot_name][env_indxs, :] 
                self._root_q_prev[robot_name][env_indxs, :] = self._root_q[robot_name][env_indxs, :]
                self._root_v_prev[robot_name][env_indxs, :] = self._root_v[robot_name][env_indxs, :] 
                self._root_omega_prev[robot_name][env_indxs, :] = self._root_omega[robot_name][env_indxs, :]

        else:
            # updating data for all environments
            pose = self._robots_art_views[robot_name].get_world_poses( 
                                            clone = True) # tuple: (pos, quat)
            self._root_p[robot_name][:, :] = pose[0]  
            self._root_q[robot_name][:, :] = pose[1] # root orientation

            going_to_fly=self._root_p[robot_name][:, 0:2]>(self._env_opts["terrain_border"]-0.1)
            if going_to_fly.any():
                flying=going_to_fly.sum().item()
                warn = f"N. {flying} robots ({robot_name}) are about to go out of the terrain!!"
                Journal.log(self.__class__.__name__,
                    "_get_root_state",
                    warn,
                    LogType.WARN,
                    throw_when_excep = True)
                
            if not numerical_diff:
                # we get velocities from the simulation. This is not good since 
                # these can actually represent artifacts which do not have physical meaning.
                # It's better to obtain them by differentiation to avoid issues with controllers, etc...
                self._root_v[robot_name][:, :] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocity 
                self._root_omega[robot_name][:, :] = self._robots_art_views[robot_name].get_angular_velocities(
                                                clone = True) # root ang. velocity
                
                self._root_a[robot_name][:, :] = (self._root_v[robot_name][:, :] - \
                                                self._root_v_prev[robot_name][:, :]) / dt 
                self._root_alpha[robot_name][:, :] = (self._root_omega[robot_name][:, :] - \
                                                self._root_omega_prev[robot_name][:, :]) / dt 
                
                self._root_v_prev[robot_name][:, :] = self._root_v[robot_name][:, :] 
                self._root_omega_prev[robot_name][:, :]  = self._root_omega[robot_name][:, :]
            else: 
                # differentiate numerically
                self._root_v[robot_name][:, :] = (self._root_p[robot_name][:, :] - \
                                                self._root_p_prev[robot_name][:, :]) / dt 
                self._root_omega[robot_name][:, :] = quat_to_omega(self._root_q_prev[robot_name][:, :], 
                                                        self._root_q[robot_name][:, :], 
                                                        dt)

                self._root_a[robot_name][:, :] = (self._root_v[robot_name][:, :] - \
                                                self._root_v_prev[robot_name][:, :]) / dt 
                self._root_alpha[robot_name][:, :] = (self._root_omega[robot_name][:, :] - \
                                                self._root_omega_prev[robot_name][:, :]) / dt 
                
                # update "previous" data for numerical differentiation
                self._root_p_prev[robot_name][:, :] = self._root_p[robot_name][:, :] 
                self._root_q_prev[robot_name][:, :] = self._root_q[robot_name][:, :]
                self._root_v_prev[robot_name][:, :] = self._root_v[robot_name][:, :] 
                self._root_omega_prev[robot_name][:, :]  = self._root_omega[robot_name][:, :]
        
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
            
            # rotate gravity in base local
            world2base_frame3D(v_w=self._gravity_normalized[robot_name],q_b=self._root_q[robot_name],
                v_out=self._gravity_normalized_base_loc[robot_name])

    def _get_robots_jnt_state(self, 
        robot_name: str,
        env_indxs: torch.Tensor = None,
        numerical_diff: bool = False):
        
        dt= self.physics_dt() if self._override_low_lev_controller else self._cluster_dt[robot_name]
        
        # measurements from simulator are in world frame 
        if env_indxs is not None:
        
            self._jnts_q[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True,
                                            indices=env_indxs) # joint positions 
            if not numerical_diff:
                self._jnts_v[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True,
                                            indices=env_indxs) # joint velocities
            else:
                # differentiate numerically
                self._jnts_v[robot_name][env_indxs, :] = (self._jnts_q[robot_name][env_indxs, :] - \
                                                self._jnts_q_prev[robot_name][env_indxs, :]) / dt
                # update "previous" data for numerical differentiation
                self._jnts_q_prev[robot_name][env_indxs, :] = self._jnts_q[robot_name][env_indxs, :]

            self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                            clone = True,
                                            joint_indices=None,
                                            indices=env_indxs) # measured joint efforts (computed by joint force solver)

        else:
            self._jnts_q[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            if not numerical_diff:
                self._jnts_v[robot_name][:, :] = self._robots_art_views[robot_name].get_joint_velocities( 
                                                clone = True) # joint velocities
            else: 
                self._jnts_v[robot_name][:, :] = (self._jnts_q[robot_name][:, :] - \
                                                self._jnts_q_prev[robot_name][:, :]) / dt
                
                self._jnts_q_prev[robot_name][:, :] = self._jnts_q[robot_name][:, :]
            
            self._jnts_eff[robot_name][env_indxs, :] = self._robots_art_views[robot_name].get_measured_joint_efforts( 
                                            clone = True) # measured joint efforts (computed by joint force solver)

    def _get_contact_f(self, 
        robot_name: str, 
        contact_link: str,
        env_indxs: torch.Tensor) -> torch.Tensor:
        
        if self.omni_contact_sensors[robot_name] is not None:
            return self.omni_contact_sensors[robot_name].get(dt=self.physics_dt(),
                            contact_link=contact_link,
                            env_indxs=env_indxs,
                            clone=False)
    
    def _set_jnts_to_homing(self, robot_name: str):
        self._robots_art_views[robot_name].set_joints_default_state(positions=self._homing, 
            velocities = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                dtype=self._dtype, device=self._device), 
            efforts = torch.zeros((self._homing.shape[0], self._homing.shape[1]), \
                                dtype=self._dtype, device=self._device))
                
    def _set_root_to_defconfig(self, robot_name: str):
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

        ground_info = f"[Ground info]" + "\n" + \
            "static friction coeff.: " + str(self._terrain_material.GetStaticFrictionAttr().Get()) + "\n" + \
            "dynamics friction coeff.: " + str(self._terrain_material.GetDynamicFrictionAttr().Get()) + "\n" + \
            "restitution coeff.: " + str(self._terrain_material.GetRestitutionAttr().Get()) + "\n" +\
            "friction comb. mode: " + str(self._terrain_physix_material.GetFrictionCombineModeAttr().Get()) + "\n" + \
            "damping comb. mode: " + str(self._terrain_physix_material.GetDampingCombineModeAttr().Get()) + "\n" + \
            "restitution comb. mode: " + str(self._terrain_physix_material.GetRestitutionCombineModeAttr().Get()) + "\n"
        
        Journal.log(self.__class__.__name__,
            "_print_envs_info",
            ground_info,
            LogType.STAT,
            throw_when_excep = True)
        
        for i in range(0, len(self._robot_names)):
            robot_name = self._robot_names[i]
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
        set_camera_view(eye=camera_position, 
                        target=camera_target, 
                        camera_prim_path="/OmniverseKit_Persp")

    def _init_contact_sensors(self, robot_name: str):
        self.omni_contact_sensors[robot_name]=None
        sensor_radii={}
        contact_offsets={}
        self._contact_names[robot_name]=self._env_opts["contact_prims"]
        for contact_prim in self._env_opts["contact_prims"]:
            sensor_radii[contact_prim]=self._env_opts["sensor_radii"]
            contact_offsets[contact_prim]=np.array([0.0, 0.0, 0.0])
        if not (len(self._env_opts["contact_prims"])==0):
            self.omni_contact_sensors[robot_name]=OmniContactSensors(
                name=robot_name, 
                n_envs=self._num_envs, 
                contact_prims=self._env_opts["contact_prims"], 
                contact_offsets=contact_offsets, 
                sensor_radii=sensor_radii, 
                device=self._device, 
                dtype=self._dtype,
                enable_debug=self._debug)
            self.omni_contact_sensors[robot_name].create_contact_sensors(
                self._world,
                envs_namespace=self._env_opts["envs_ns"])            
    
    def _init_robots_state(self):
        
        self._masses = {}
        self._weights = {}

        self._pert_active = {}            # bool mask: (num_envs,)
        self._pert_steps_remaining = {}   # int steps: (num_envs,)
        self._pert_forces_world = {}       # (num_envs,3)
        self._pert_torques_world = {}      # (num_envs,3)
        self._pert_force_local = {}       # (num_envs,3)  (if needed)
        self._pert_torque_local = {}      # (num_envs,3)
        self._pert_lindir = {}
        self._pert_angdir = {}
        self._pert_durations = {}
        self._pert_scratch = {}

        # convert durations in seconds to integer physics steps (min 1 step)
        self._pert_min_steps = max(1, int(math.ceil(self._env_opts["pert_wrenches_min_duration"] / self.physics_dt())))
        self._pert_max_steps = max(self._pert_min_steps, int(math.ceil(self._env_opts["pert_wrenches_max_duration"] / self.physics_dt())))

        pert_wrenches_step_rate=self._env_opts["pert_wrenches_rate"]/self.physics_dt() # 1 pert every n physics steps
        self._pert_wrenches_prob=1.0/pert_wrenches_step_rate # sampling prob to be used

        self._calc_robot_distrib()

        for i in range(0, len(self._robot_names)):

            robot_name = self._robot_names[i]
            pose = self._robots_art_views[robot_name].get_world_poses( 
                clone = True) # tuple: (pos, quat)

            # root p (measured, previous, default)
            self._root_p[robot_name] = pose[0]  
            self._root_p_prev[robot_name] = torch.clone(pose[0])
            # print(self._root_p_default[robot_name].device)
            self._root_p_default[robot_name] = torch.clone(pose[0]) + self._distr_offset[robot_name]
            # root q (measured, previous, default)
            self._root_q[robot_name] = pose[1] # root orientation
            self._root_q_prev[robot_name] = torch.clone(pose[1])
            self._root_q_default[robot_name] = torch.clone(pose[1])
            # jnt q (measured, previous, default)
            self._jnts_q[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) # joint positions 
            self._jnts_q_prev[robot_name] = self._robots_art_views[robot_name].get_joint_positions(
                                            clone = True) 
            self._jnts_q_default[robot_name] = torch.full((self._jnts_q[robot_name].shape[0], 
                                                           self._jnts_q[robot_name].shape[1]), 
                                                            0.0, 
                                                            dtype=self._dtype, 
                                                            device=self._device)
            
            # root v (measured, default)
            self._root_v[robot_name] = self._robots_art_views[robot_name].get_linear_velocities(
                                            clone = True) # root lin. velocityù
            self._root_v_base_loc[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_prev[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            self._root_v_default[robot_name] = torch.full_like(self._root_v[robot_name], fill_value=0.0)
            # root omega (measured, default)
            self._root_omega[robot_name] = self._robots_art_views[robot_name].get_angular_velocities(
                                            clone = True) # root ang. velocity
            self._root_omega_prev[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)

            self._root_omega_base_loc[robot_name] = torch.full_like(self._root_omega[robot_name], fill_value=0.0)
            self._root_omega_default[robot_name] = torch.full((self._root_omega[robot_name].shape[0], self._root_omega[robot_name].shape[1]), 
                                                        0.0, 
                                                        dtype=self._dtype, 
                                                        device=self._device)
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
            self._jnts_v[robot_name] = self._robots_art_views[robot_name].get_joint_velocities( 
                                            clone = True) # joint velocities
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

            self._update_root_offsets(robot_name)

            # boolean active flag per env
            self._pert_active[robot_name] = torch.zeros((self._num_envs,), dtype=torch.bool, device=self._device)
            # remaining steps as integer tensor
            self._pert_steps_remaining[robot_name] = torch.zeros((self._num_envs,), dtype=torch.int32, device=self._device)
            # world force & torque (N and N*m) stored as floats
            self._pert_forces_world[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)
            self._pert_torques_world[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)
            # local frame copies (if you want to store local-frame versions)
            self._pert_force_local[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)
            self._pert_torque_local[robot_name] = torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)
                
            self._pert_lindir[robot_name] =  torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)
            self._pert_angdir[robot_name] =  torch.zeros((self._num_envs, 3), dtype=self._dtype, device=self._device)

            self._pert_durations[robot_name] = torch.zeros((self._num_envs, 1), dtype=torch.int32, device=self._device)

            self._pert_scratch[robot_name] = torch.zeros((self._num_envs, 1), dtype=self._dtype, device=self._device)
            
            self._masses[robot_name] = torch.sum(self._robots_art_views[robot_name].get_body_masses(clone=True), dim=1).to(dtype=self._dtype, device=self._device)

            self._weights[robot_name] = (self._masses[robot_name] * abs(self._env_opts["gravity"][2].item())).reshape((self._num_envs, 1))
            
    def current_tstep(self):
        self._world.current_time_step_index
    
    def current_time(self):
        return self._world.current_time
    
    def physics_dt(self):
        return self._world.get_physics_dt()
    
    def rendering_dt(self):
        return self._env_opts["rendering_dt"]
    
    def set_physics_dt(self, physics_dt:float):
        self._world.set_simulation_dt(physics_dt=physics_dt,rendering_dt=None)
    
    def set_rendering_dt(self, rendering_dt:float):
        self._world.set_simulation_dt(physics_dt=None,rendering_dt=rendering_dt)
    
    def _robot_jnt_names(self, robot_name: str):
        return self._robots_art_views[robot_name].dof_names
