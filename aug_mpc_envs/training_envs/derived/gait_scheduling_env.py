import torch
from typing import Dict

from EigenIPC.PyEigenIPC import VLevel

from aug_mpc_envs.training_envs.derived.fake_pos_tracking_env_with_demo import FakePosTrackingEnvWithDemo


class GaitSchedulingEnv(FakePosTrackingEnvWithDemo):
    """
    Same as FakePosTrackingEnvWithDemo but does not write MPC twist references
    to shared memory. Contact flags are still written so gait scheduling
    can be exercised without overriding twist refs.
    """

    def __init__(self,
            namespace: str,
            verbose: bool = False,
            vlevel: VLevel = VLevel.V1,
            use_gpu: bool = True,
            dtype: torch.dtype = torch.float32,
            debug: bool = True,
            override_agent_refs: bool = False,
            timeout_ms: int = 60000,
            env_opts: Dict = {}):

        env_opts["add_heightmap_obs"] = False
        env_opts.setdefault("walk_to_trot_delay_s", 0.8)
        env_opts.setdefault("trot_to_walk_delay_s", 0.5)
        env_opts.setdefault("enable_gait_transition", False)
        env_opts.setdefault("default_gait", "trot")
        env_opts.setdefault("airborne_safety", True)
        
        env_opts["full_demo"] = False
        env_opts["smooth_twist_cmd"] = False
        
        super().__init__(namespace=namespace,
            verbose=verbose,
            vlevel=vlevel,
            use_gpu=use_gpu,
            dtype=dtype,
            debug=debug,
            override_agent_refs=override_agent_refs,
            timeout_ms=timeout_ms,
            env_opts=env_opts)

        # Override gait transition defaults for this env
        self._env_opts["stopping_thresh"] = self._env_opts.get("stopping_thresh", 0.08)
        self._env_opts["walk_to_trot_thresh_linvel"] = 0.7
        self._env_opts["walk_to_trot_thresh_omega"] = 0.6
        # Adaptive gait cadence based on refs: (min,max) event intervals between successive lift-offs
        self.phase_period_walk_min=0.7
        self.phase_period_walk_max=1.0
        self.phase_period_trot_min=0.65
        self.phase_period_trot_max=1.5
        # phase accumulators and buffers
        self._walk_phase = torch.zeros((self._n_demo_envs, 1), device=self._device, dtype=self._dtype)
        self._trot_phase = torch.zeros_like(self._walk_phase)
        # Walk sequence: FL -> FR -> BR -> BL (clockwise)
        # Contact liftoff occurs when sin crosses 0; offset = pi - desired_liftoff_phase
        self._walk_offsets = torch.tensor([[torch.pi, torch.pi/2, 0.0, 3*torch.pi/2]],
                                          device=self._device, dtype=self._dtype)
        self._trot_offsets = torch.tensor([[0.0, torch.pi, torch.pi, 0.0]],
                                          device=self._device, dtype=self._dtype)
        self._prev_contact_walk = torch.ones((self._n_demo_envs, 4), device=self._device, dtype=torch.bool)
        self._prev_contact_trot = torch.ones_like(self._prev_contact_walk)
        self._walk_init_mask = torch.zeros((self._n_demo_envs,), device=self._device, dtype=torch.bool)
        self._trot_init_mask = torch.zeros_like(self._walk_init_mask)
        self._gait_mode = torch.zeros((self._n_demo_envs,), device=self._device, dtype=torch.bool)  # False -> walk, True -> trot
        self._trot_delay_timer = torch.zeros((self._n_demo_envs, 1), device=self._device, dtype=self._dtype)
        self._walk_delay_timer = torch.zeros_like(self._trot_delay_timer)

        self._dt = self._substep_dt * self._action_repeat
        
    def _write_rhc_refs(self):
        """Do not touch MPC twist references; only push contact flags if needed."""
        if self._use_gpu:
            self._rhc_refs.contact_flags.synch_mirror(from_gpu=True, non_blocking=False)
            self._rhc_refs.rob_refs.contact_pos.synch_mirror(from_gpu=True,non_blocking=False)

        self._rhc_refs.contact_flags.synch_all(read=False, retry=True)
        self._rhc_refs.rob_refs.contact_pos.synch_all(read=False, retry=True)

    def get_file_paths(self):
        paths = super().get_file_paths()
        import os
        paths.append(os.path.abspath(__file__))
        return paths

    def _map_period(self, speed_val, pmin, pmax, smin: float = 0.0, smax: float = 1.5):
        """Map speed to gait event period within [pmin, pmax]."""
        speed_clamped = torch.clamp(speed_val, smin, smax)
        alpha = (speed_clamped - smin) / (smax - smin + 1e-6)
        return pmax - alpha * (pmax - pmin)

    # --- helpers to keep _override_actions_with_demo tidy ---
    def _compute_speed_terms(self, rhc_twist_refs):
        speed_lin = rhc_twist_refs[:, 0:2].norm(dim=1, keepdim=True)
        speed_yaw = rhc_twist_refs[:, 5:6].abs()
        speed = torch.maximum(speed_lin, speed_yaw)
        walk_event = self._map_period(speed, self.phase_period_walk_min, self.phase_period_walk_max)
        trot_event = self._map_period(speed, self.phase_period_trot_min, self.phase_period_trot_max)
        walk_period_full = walk_event * 4
        trot_period_full = trot_event * 2
        return speed_lin, speed_yaw, walk_period_full, trot_period_full

    def _update_phases(self, walk_period_full, trot_period_full, flight_lengths_now):
        demo_mask = self._demo_envs_idxs_bool
        walk_period_demo = walk_period_full[demo_mask, :]
        trot_period_demo = trot_period_full[demo_mask, :]

        if self._env_opts["airborne_safety"]:
            flight_lengths_demo = flight_lengths_now[demo_mask, :]
            valid_lengths = torch.isfinite(flight_lengths_demo) & (flight_lengths_demo > 0.0)
            safe_lengths = torch.where(valid_lengths, flight_lengths_demo, torch.zeros_like(flight_lengths_demo))
            flight_len_max = safe_lengths.max(dim=1, keepdim=True).values
            has_valid_lengths = valid_lengths.any(dim=1, keepdim=True)

            # Safety bounds on full-cycle period:
            # walk -> T_event = T_full / 4 >= flight_len_max (never 2 feet airborne together)
            # trot -> T_event = T_full / 2 >= flight_len_max (never all feet airborne)
            walk_period_min = 4.0 * flight_len_max
            trot_period_min = 2.0 * flight_len_max
            walk_period_demo = torch.where(has_valid_lengths,
                                           torch.maximum(walk_period_demo, walk_period_min),
                                           walk_period_demo)
            trot_period_demo = torch.where(has_valid_lengths,
                                           torch.maximum(trot_period_demo, trot_period_min),
                                           trot_period_demo)

        omega_walk = 2 * torch.pi / torch.clamp(walk_period_demo, min=1e-3)
        omega_trot = 2 * torch.pi / torch.clamp(trot_period_demo, min=1e-3)
        self._walk_phase = (self._walk_phase + omega_walk * self._dt) % (2 * torch.pi)
        self._trot_phase = (self._trot_phase + omega_trot * self._dt) % (2 * torch.pi)

    def _build_masks(self, speed_lin, speed_yaw, demo_mask):
        fast = (speed_lin > self._env_opts["walk_to_trot_thresh_linvel"]) | \
               (speed_yaw > self._env_opts["walk_to_trot_thresh_omega"])
        fast_and_demo = fast.flatten() & demo_mask
        slow_and_demo = (~fast.flatten()) & demo_mask

        stop = (speed_lin < self._env_opts["stopping_thresh"]) & \
               (speed_yaw < self._env_opts["stopping_thresh"])
        stop_and_demo = stop.flatten() & demo_mask
        return fast_and_demo, slow_and_demo, stop_and_demo

    def _compute_dynamic_transition_delays(self, flight_lengths_now):
        demo_mask = self._demo_envs_idxs_bool
        flight_lengths_demo = flight_lengths_now[demo_mask, :]
        valid_lengths = torch.isfinite(flight_lengths_demo) & (flight_lengths_demo > 0.0)
        safe_lengths = torch.where(valid_lengths, flight_lengths_demo, torch.zeros_like(flight_lengths_demo))
        flight_len_max = safe_lengths.max(dim=1).values
        has_valid_lengths = valid_lengths.any(dim=1)

        walk_to_trot_delay_now = torch.full_like(
            flight_len_max, float(self._env_opts["walk_to_trot_delay_s"])
        )
        trot_to_walk_delay_now = torch.full_like(
            flight_len_max, float(self._env_opts["trot_to_walk_delay_s"])
        )
        walk_to_trot_delay_now = torch.where(
            has_valid_lengths,
            torch.maximum(walk_to_trot_delay_now, flight_len_max),
            walk_to_trot_delay_now,
        )
        trot_to_walk_delay_now = torch.where(
            has_valid_lengths,
            torch.maximum(trot_to_walk_delay_now, flight_len_max),
            trot_to_walk_delay_now,
        )
        return walk_to_trot_delay_now, trot_to_walk_delay_now

    def _update_gait_mode(self, fast_and_demo, slow_and_demo,
                          walk_to_trot_delay_now=None, trot_to_walk_delay_now=None):
        if self._env_opts["enable_gait_transition"]:
            fast_demo_mask = torch.zeros_like(self._gait_mode)
            slow_demo_mask = torch.zeros_like(self._gait_mode)
            if fast_and_demo.any():
                fast_demo_mask[self._env_to_gait_sched_mapping[fast_and_demo]] = True
            if slow_and_demo.any():
                slow_demo_mask[self._env_to_gait_sched_mapping[slow_and_demo]] = True

            self._trot_delay_timer[fast_demo_mask, 0] += self._dt
            self._trot_delay_timer[~fast_demo_mask, 0] = 0.0
            self._walk_delay_timer[slow_demo_mask, 0] += self._dt
            self._walk_delay_timer[~slow_demo_mask, 0] = 0.0

            if walk_to_trot_delay_now is None:
                walk_to_trot_delay_now = torch.full(
                    (self._n_demo_envs,),
                    float(self._env_opts["walk_to_trot_delay_s"]),
                    device=self._device,
                    dtype=self._dtype,
                )
            if trot_to_walk_delay_now is None:
                trot_to_walk_delay_now = torch.full(
                    (self._n_demo_envs,),
                    float(self._env_opts["trot_to_walk_delay_s"]),
                    device=self._device,
                    dtype=self._dtype,
                )

            switch_to_trot = (~self._gait_mode) & (self._trot_delay_timer[:, 0] >= walk_to_trot_delay_now)
            switch_to_walk = self._gait_mode & (self._walk_delay_timer[:, 0] >= trot_to_walk_delay_now)

            if switch_to_trot.any():
                self._walk_init_mask[switch_to_trot] = False
            if switch_to_walk.any():
                self._trot_init_mask[switch_to_walk] = False

            self._gait_mode[switch_to_trot] = True
            self._gait_mode[switch_to_walk] = False
        else:
            use_trot = self._env_opts["default_gait"].lower() == "trot"
            self._gait_mode.fill_(use_trot)
            self._trot_delay_timer.zero_()
            self._walk_delay_timer.zero_()

    def _apply_contact_pattern(self, active_env_mask, phase_tensor, offsets, prev_contact, init_mask, agent_action):
        if not active_env_mask.any():
            return
        env_idxs = self._env_to_gait_sched_mapping[active_env_mask]
        sin_val = torch.sin(phase_tensor[env_idxs, :] + offsets)
        is_contact = sin_val > 0.0
        liftoff = torch.logical_and(prev_contact[env_idxs, :], torch.logical_not(is_contact))
        liftoff = torch.logical_and(liftoff, init_mask[env_idxs].unsqueeze(1))
        contact_flag = torch.ones_like(is_contact, dtype=self._dtype)
        contact_flag[liftoff] = -1.0
        pulses = (contact_flag == -1).sum(dim=1, keepdim=True)
        if (pulses > 2).any():
            contact_flag[pulses > 2] = 1.0
        agent_action[active_env_mask, 6:10] = contact_flag
        prev_contact[env_idxs, :] = is_contact.detach().clone()
        init_mask[env_idxs] = True

    def _override_actions_with_demo(self):
        """Use measured twist to drive gait scheduling while keeping twist refs untouched."""
        if self.demo_active():

            agent_action = self.get_actions()
            demo_mask = self._demo_envs_idxs_bool
            demo_idxs = self._env_to_gait_sched_mapping[demo_mask]

            # agent_twist_ref_current = self._agent_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
            # use current MPC refs to decide gait mode
            rhc_twist_refs = self._rhc_refs.rob_refs.root_state.get(data_type="twist", gpu=self._use_gpu)
            flight_lengths_now = self._rhc_refs.flight_info.get(data_type="len",gpu=self._use_gpu)*self._rhc_dts # read flight length [s]
            
            speed_lin, speed_yaw, walk_period_full, trot_period_full = self._compute_speed_terms(rhc_twist_refs)
            self._update_phases(walk_period_full, trot_period_full, flight_lengths_now)

            fast_and_demo, slow_and_demo, stop_and_demo = self._build_masks(speed_lin, speed_yaw, demo_mask)
            walk_to_trot_delay_now, trot_to_walk_delay_now = \
                self._compute_dynamic_transition_delays(flight_lengths_now)
            self._update_gait_mode(
                fast_and_demo,
                slow_and_demo,
                walk_to_trot_delay_now=walk_to_trot_delay_now,
                trot_to_walk_delay_now=trot_to_walk_delay_now,
            )

            active_walk_env = torch.zeros_like(demo_mask)
            active_trot_env = torch.zeros_like(demo_mask)
            active_walk_env[self._demo_envs_idxs_bool] = (~self._gait_mode)[demo_idxs]
            active_trot_env[self._demo_envs_idxs_bool] = self._gait_mode[demo_idxs]

            # Walk contact pattern from phase accumulator (50% duty via sin > 0)
            self._apply_contact_pattern(active_walk_env, self._walk_phase, self._walk_offsets,
                                        self._prev_contact_walk, self._walk_init_mask, agent_action)

            self._apply_contact_pattern(active_trot_env, self._trot_phase, self._trot_offsets,
                                        self._prev_contact_trot, self._trot_init_mask, agent_action)

            if stop_and_demo.any():
                agent_action[stop_and_demo, 6:10] = 1.0
