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
import torch

from EigenIPC.PyEigenIPC import LogType
from EigenIPC.PyEigenIPC import Journal

from aug_mpc.utils.jnt_imp_control_base import JntImpCntrlBase
from adarl.adapters.GenesisJointImpedanceAdapter import GenesisJointImpedanceAdapter


class GenesisJntImpCntrl(JntImpCntrlBase):
    """Joint-impedance controller for the Genesis backend.

    Mirrors ``XMjJntImpCntrl``: it does not compute torques itself but accumulates a
    per-joint impedance command (pvesd = pos/vel/eff/stiffness/damping) that the world
    interface pushes to the ``GenesisJointImpedanceAdapter`` via
    ``setJointsImpedanceCommand`` (the adapter performs the impedance control internally).
    Unlike the single-env XMj controller, this one is vectorized over ``num_envs``.
    """

    def __init__(self,
        genesis_adapter: GenesisJointImpedanceAdapter,
        num_envs: int,
        default_pgain: float = 300.0,
        default_vgain: float = 30.0,
        device: torch.device = torch.device("cpu"),
        filter_BW=50.0,            # [Hz]
        filter_dt=None,            # should correspond to the dt between samples
        dtype=torch.double,
        enable_safety=True,
        urdf_path: str = None,
        config_path: str = None,
        enable_profiling: bool = False,
        debug_checks: bool = False,
        override_art_controller=False):

        self._genesis_adapter = genesis_adapter  # used to actually apply control to the sim

        controlled_joints = self._genesis_adapter.get_impedance_controlled_joints()
        jnts_names = []
        self._model_name = controlled_joints[0][0]
        for joint in controlled_joints:
            if not self._model_name == joint[0]:
                Journal.log(self.__class__.__name__,
                    "__init__",
                    f"Only one model name is currently supported. Read {joint[0]}, while prev. {self._model_name}",
                    LogType.EXCEP,
                    throw_when_excep=True)
            jnts_names.append(joint[1])
        n_jnts = len(controlled_joints)

        # (num_envs, n_jnts, 5) -> [pos_ref, vel_ref, eff_ref, stiffness, damping]
        # kept on the controller device so it can be forwarded straight to the adapter
        self._pvesd_adapter = torch.zeros((num_envs, n_jnts, 5),
            device=device,
            dtype=dtype)
                
        super().__init__(num_envs=num_envs,
            n_jnts=n_jnts,
            jnt_names=jnts_names,
            default_pgain=default_pgain,
            default_vgain=default_vgain,
            device=device,
            filter_BW=filter_BW,
            filter_dt=filter_dt,
            dtype=dtype,
            enable_safety=enable_safety,
            urdf_path=urdf_path,
            config_path=config_path,
            enable_profiling=enable_profiling,
            debug_checks=debug_checks,
            override_low_lev_controller=override_art_controller)

    def get_pvesd(self):
        return self._pvesd_adapter

    def _set_gains(self,
        kps: torch.Tensor = None,
        kds: torch.Tensor = None):
        if kps is not None:
            self._pvesd_adapter[:, :, 3] = kps.to(self._pvesd_adapter.device)
        if kds is not None:
            self._pvesd_adapter[:, :, 4] = kds.to(self._pvesd_adapter.device)

    def _set_pos_ref(self, pos: torch.Tensor):
        self._pvesd_adapter[:, :, 0] = pos.to(self._pvesd_adapter.device)

    def _set_vel_ref(self, vel: torch.Tensor):
        self._pvesd_adapter[:, :, 1] = vel.to(self._pvesd_adapter.device)

    def _set_joint_efforts(self, effort: torch.Tensor):
        self._pvesd_adapter[:, :, 2] = effort.to(self._pvesd_adapter.device)
