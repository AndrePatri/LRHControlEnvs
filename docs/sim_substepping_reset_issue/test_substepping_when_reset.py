# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import numpy as np

import torch

def get_device(sim_params):

    if "sim_device" in sim_params:
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
    
    return device

def sim_parameters():

    # simulation parameters
    sim_params = {}
    # device settings
    sim_params["use_gpu_pipeline"] = True # disabling gpu pipeline is necessary to be able
    # to retrieve some quantities from the simulator which, otherwise, would have random values
    sim_params["use_gpu"] = True # does this actually do anything?
    if sim_params["use_gpu_pipeline"]:
        sim_params["device"] = "cuda"
    else:
        sim_params["device"] = "cpu"
    device = sim_params["device"]

    # sim_params["dt"] = 1.0/100.0 # physics_dt?
    sim_params["physics_dt"] = 1.0/400.0 # physics_dt?
    sim_params["rendering_dt"] = sim_params["physics_dt"]
    sim_params["substeps"] = 1 # number of physics steps to be taken for for each rendering step
    sim_params["gravity"] = np.array([0.0, 0.0, -9.81])
    sim_params["enable_scene_query_support"] = False
    sim_params["use_fabric"] = True # Enable/disable reading of physics buffers directly. Default is True.
    sim_params["replicate_physics"] = True
    # sim_params["worker_thread_count"] = 4
    sim_params["solver_type"] =  1 # 0: PGS, 1:TGS, defaults to TGS. PGS faster but TGS more stable
    sim_params["enable_stabilization"] = True
    # sim_params["bounce_threshold_velocity"] = 0.2
    # sim_params["friction_offset_threshold"] = 0.04
    # sim_params["friction_correlation_distance"] = 0.025
    # sim_params["enable_sleeping"] = True
    # Per-actor settings ( can override in actor_options )
    sim_params["solver_position_iteration_count"] = 4 # defaults to 4
    sim_params["solver_velocity_iteration_count"] = 1 # defaults to 1
    sim_params["sleep_threshold"] = 0.0 # Mass-normalized kinetic energy threshold below which an actor may go to sleep.
    # Allowed range [0, max_float).
    sim_params["stabilization_threshold"] = 1e-5
    # Per-body settings ( can override in actor_options )
    # sim_params["enable_gyroscopic_forces"] = True
    # sim_params["density"] = 1000 # density to be used for bodies that do not specify mass or density
    # sim_params["max_depenetration_velocity"] = 100.0
    # sim_params["solver_velocity_iteration_count"] = 1

    # GPU buffers settings
    # sim_params["gpu_max_rigid_contact_count"] = 512 * 1024
    # sim_params["gpu_max_rigid_patch_count"] = 80 * 1024
    # sim_params["gpu_found_lost_pairs_capacity"] = 1024
    # sim_params["gpu_found_lost_aggregate_pairs_capacity"] = 1024
    # sim_params["gpu_total_aggregate_pairs_capacity"] = 1024
    # sim_params["gpu_max_soft_body_contacts"] = 1024 * 1024
    # sim_params["gpu_max_particle_contacts"] = 1024 * 1024
    # sim_params["gpu_heap_capacity"] = 64 * 1024 * 1024
    # sim_params["gpu_temp_buffer_capacity"] = 16 * 1024 * 1024
    # sim_params["gpu_max_num_partitions"] = 8

    return sim_params

def reset_state(art_view,
                idxs: torch.Tensor):

    # root q
    art_view.set_world_poses(positions = root_p_default[idxs, :],
                    orientations=root_q_default[idxs, :],
                    indices = idxs)
    # jnts q
    art_view.set_joint_positions(positions = jnts_q_default[idxs, :],
                            indices = idxs)
    
    # root v and omega
    art_view.set_joint_velocities(velocities = jnts_v_default[idxs, :],
                             indices = idxs)
    
    # jnts v
    concatenated_vel = torch.cat((root_v_default[idxs, :], 
                            root_omega_default[idxs, :]), dim=1)

    art_view.set_velocities(velocities = concatenated_vel,
                                indices = idxs)
    
    # jnts eff
    art_view.set_joint_efforts(efforts = jnts_eff_default[idxs, :],
                        indices = idxs)
                
def get_robot_state(
                art_view):

    pose = art_view.get_world_poses( 
                                clone = True) # tuple: (pos, quat)

    # root p (measured, previous, default)
    root_p = pose[0]  
    
    # root q (measured, previous, default)
    root_q = pose[1] # root orientation
    
    # jnt q (measured, previous, default)
    jnts_q = art_view.get_joint_positions(
                                    clone = True) # joint positions 
    
    # root v (measured, default)
    root_v= art_view.get_linear_velocities(
                                    clone = True) # root lin. velocity
    

    # root omega (measured, default)
    root_omega = art_view.get_angular_velocities(
                                    clone = True) # root ang. velocity
   
    # joints v (measured, default)
    jnts_v = art_view.get_joint_velocities( 
                        clone = True) # joint velocities
    
    jnts_eff = art_view.get_measured_joint_efforts(clone = True)

    return root_p, root_q, jnts_q, root_v, root_omega, jnts_v, jnts_eff

from omni.isaac.kit import SimulationApp
import carb

import os

experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.lrhcontrolenvs.headless.kit'

sim_params = sim_parameters()

num_envs = 2
headless = True

simulation_app = SimulationApp({"headless": headless,
                            "physics_gpu": 0}, 
                            experience=experience)

from omni.isaac.core import World
from omni.isaac.core.articulations import ArticulationView

from omni.importer.urdf import _urdf

# urdf import config
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = True 
import_config.import_inertia_tensor = True
import_config.fix_base = False
import_config.self_collision = False

my_world = World(stage_units_in_meters=1.0, 
            physics_dt=sim_params["physics_dt"], 
            rendering_dt=sim_params["rendering_dt"],
            backend="torch",
            device=str(get_device(sim_params=sim_params)),
            physics_prim_path="/physicsScene", 
            set_defaults = False, 
            sim_params=sim_params)

# create initial robot
import omni.isaac.core.utils.prims as prim_utils

# create GridCloner instance
env_ns = "/World/envs"
template_env_ns = env_ns + "/env" # a single env. may contain multiple robots
base_env = template_env_ns + "_0"
base_robot_path = base_env + "/panda"

# get path to resource
from omni.isaac.core.utils.extensions import get_extension_path_from_name
extension_path = get_extension_path_from_name("omni.importer.urdf")

# import URDF at default prim path
import omni.kit
success, robot_prim_path_default = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=extension_path + "/data/urdf/robots/franka_description/robots/panda_arm.urdf",
    import_config=import_config, 
)

# moving default prim to base prim path (for potential cloning)
from omni.isaac.core.utils.prims import move_prim
prim_utils.define_prim(base_env)
move_prim(robot_prim_path_default, # from
        base_robot_path) # to

# cloning
from omni.isaac.cloner import GridCloner

cloner = GridCloner(spacing=6)
_envs_prim_paths = cloner.generate_paths(template_env_ns, num_envs)

position_offsets = np.array([[0.0, 0.0, 0.6]] * num_envs)
cloner.clone(
    source_prim_path=base_env,
    prim_paths=_envs_prim_paths,
    base_env_path=base_env,
    position_offsets=position_offsets,
    replicate_physics=True
)

# Prim paths structure:
# World/envs/env_0/panda/panda_link0/...

# this only in 2023.1.0
art_view = ArticulationView(name = "Panda" + "ArtView",
                        prim_paths_expr = env_ns + "/env_.*"+ "/panda/panda_link0", 
                        reset_xform_properties=False # required as per doc. when cloning
                        )

# moreover, robots are not cloned at different locations
my_world.scene.add(art_view)

ground_plane_prim_path = "/World/terrain"
my_world.scene.add_default_ground_plane(z_position=0, 
                            name="terrain", 
                            prim_path= ground_plane_prim_path, 
                            static_friction=0.5, 
                            dynamic_friction=0.5, 
                            restitution=0.8)

cloner.filter_collisions(physicsscene_path = my_world.get_physics_context().prim_path,
                collision_root_path = "/World/collisions", 
                prim_paths=_envs_prim_paths, 
                global_paths=[ground_plane_prim_path] # can collide with these prims
                )

my_world.reset()

# init default state from measurements
root_p, root_q, jnts_q, root_v, \
    root_omega, jnts_v, jnts_eff = get_robot_state(art_view)

root_p_default = torch.clone(root_p)
root_q_default = torch.clone(root_q)
jnts_q_default = torch.clone(jnts_q)
jnts_v_default = torch.clone(jnts_v)
root_omega_default = torch.clone(root_omega)
root_v_default = torch.clone(root_v)
jnts_eff_default = torch.clone(jnts_eff).zero_()

# default values
root_p_default[:, 0] = 0
root_p_default[:, 1] = 0
root_p_default[:, 2] = 0.5
root_q_default[:, 0] = 0.0
root_q_default[:, 1] = 0.0
root_q_default[:, 2] = 0.0
root_q_default[:, 3] = 1.0
jnts_q_default[:, :] = 1.0
jnts_v_default[:, :] = 0.0
root_omega_default[:, :] = 0.0
root_v_default[:, :] = 0.0

no_gains = torch.zeros((num_envs, jnts_eff_default.shape[1]), device = get_device(sim_params), 
                            dtype=torch.float32)
                                                  
art_view.set_gains(kps = no_gains, 
                    kds = no_gains)

print("Extension path: " + str(extension_path))
print("Prim paths: " + str(art_view.prim_paths))

reset_ever_n_steps = 100

just_reset = False
for i in range(0, 1000):

    if ((i + 1) % reset_ever_n_steps) == 0:

        print("resetting to default")

        reset_state(art_view,
                torch.tensor([0], dtype=torch.int))

        just_reset = True
        
    my_world.step()
    
    # retrieve state
    root_p, root_q, jnts_q, root_v, \
        root_omega, jnts_v, jnts_eff = get_robot_state(art_view)
    
    # if just_reset:

    # check we hace reset correcty

    print("measured")
    print(jnts_q)
    print("default")
    print(jnts_q_default)

simulation_app.close()