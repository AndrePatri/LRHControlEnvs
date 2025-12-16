import os

from typing import Dict

import torch

from EigenIPC.PyEigenIPC import VLevel

from mpc_hive.utilities.math_utils_torch import world2base_frame

from aug_mpc_envs.training_envs.flight_phase_control_env import FlightPhaseControl

class StayingAliveEnv(FlightPhaseControl):
    """Simply env where the agent has to try to stay alive and as still as possible (useful to test disturbance rejection capabilities)."""

    def __init__(self,
            namespace: str,
            actions_dim: int = 10,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):

        # force tracking weights to favour yaw stabilization while keeping linear terms small
        env_opts["task_track_front_weight"]=0.5
        env_opts["task_track_lat_weight"]=0.05
        env_opts["task_track_vert_weight"]=0.05
        env_opts["task_track_omega_x_weight"]=1.0
        env_opts["task_track_omega_y_weight"]=1.0
        env_opts["task_track_omega_z_weight"]=1.0

        env_opts["control_flength"]=True
        env_opts["control_fapex"]=True
        env_opts["control_fend"]=True

        FlightPhaseControl.__init__(self, 
            namespace=namespace,
            actions_dim=actions_dim, # twist + contact flags
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms,
            env_opts=env_opts)

    def get_file_paths(self):
        paths=FlightPhaseControl.get_file_paths(self)
        paths.append(os.path.abspath(__file__))        
        return paths

    def _randomize_task_refs(self,
        env_indxs: torch.Tensor = None):
        # keep twist references null so the agent just maintains stability
        if env_indxs is None:
            self._agent_twist_ref_current_w[:, :]=0.0
        else:
            self._agent_twist_ref_current_w[env_indxs, :]=0.0
        
