# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from kyonrlstepping.gym.omni_vect_env.vec_envs import RobotVecEnv

import abc
import queue

class TaskStopException(Exception):
    """ Exception class for signalling task termination. """

    pass


class TrainerMT(abc.ABC):
    """ A base abstract trainer class for controlling starting and stopping of RL policy. """

    @abc.abstractmethod
    def run(self):
        """ Runs RL loop in a new thread """
        pass

    @abc.abstractmethod
    def stop(self):
        """ Stop RL thread """
        pass


class VecEnvMT(RobotVecEnv):
    """ This class provides a base interface for connecting RL policies with task implementations
        in a multi-threaded fashion. RL policies using this class will run on a different thread
        than the thread simulation runs on. This can be useful for interacting with the UI before,
        during, and after running RL policies. Data sharing between threads happen through message
        passing on multi-threaded queues.
    """

    def initialize(self, action_queue, data_queue, timeout=30):
        """ Initializes queues for sharing data across threads.

        Args: 
            action_queue (queue.Queue): Queue for passing actions from policy to task.
            data_queue (queue.Queue): Queue for passing data from task to policy.
            timeout (Optional[int]): Seconds to wait for data when queue is empty. An exception will
                                     be thrown when the timeout limit is reached. Defaults to 30 seconds.
        """

        self._action_queue = action_queue
        self._data_queue = data_queue
        self._stop = False
        self._first_frame = True
        self._timeout = timeout

    def get_actions(self, block=True):
        """ Retrieves actions from policy by waiting for actions to be sent to the queue from the RL thread.
        
        Args:
            block (Optional[bool]): Whether to block thread when waiting for data.

        Returns:
            actions (Union[np.ndarray, torch.Tensor, None]): actions buffer retrieved from queue.
        """
        if not self._stop:
            try:
                actions = self._action_queue.get(block, self._timeout)
                if actions is None:
                    self._stop = True
                    self._action_queue.task_done()
                    raise TaskStopException()
                else:
                    actions = actions.clone()
                self._action_queue.task_done()
            except (queue.Full, queue.Empty) as e:
                print("Getting actions: timeout occurred.")
                actions = None
                self._stop = True
        else:
            actions = None

        return actions

    def send_actions(self, actions, block=True):
        """ Sends actions from RL thread to simulation thread by adding actions to queue.

        Args:
            actions (Union[np.ndarray, torch.Tensor]): actions buffer to be added to queue.
            block (Optional[bool]): Whether to block thread when writing to queue.
        """

        if not self._stop:
            try:
                self._action_queue.put(actions, block, self._timeout)
            except (queue.Full, queue.Empty) as e:
                print("Sending actions: timeout occurred.")
                self._stop = True

    def get_data(self, block=True):
        """ Retrieves data from task by waiting for data dictionary to be sent to the queue from the simulation thread.
        
        Args:
            block (Optional[bool]): Whether to block thread when waiting for data.

        Returns:
            actions (Union[np.ndarray, torch.Tensor, None]): data dictionary retrieved from queue.
        """
        if not self._stop:
            try:
                if self._first_frame:
                    data = self._data_queue.get(block)
                    self._first_frame = False
                else:
                    data = self._data_queue.get(block, self._timeout)
                if data is None:
                    self._stop = True
                    raise TaskStopException()
                else:
                    self._parse_data(data)
                self._data_queue.task_done()
            except (queue.Full, queue.Empty) as e:
                print("Getting states: timeout occurred.")
                self._stop = True
                data = None
        else:
            data = None

        return data

    def send_data(self, data, block=True):
        """ Sends data from task thread to RL thread by adding data to queue.

        Args:
            data (dict): Dictionary containing task data.
            block (Optional[bool]): Whether to block thread when writing to queue.
        """

        if not self._stop:
            try:
                self._data_queue.put(data, block, self._timeout)
            except (queue.Full, queue.Empty) as e:
                print("Sending states: timeout occurred.")
                self._stop = True

    def clear_queues(self):
        """ Clears all queues. """

        while not self._action_queue.empty():
            self._action_queue.get_nowait()
            self._action_queue.task_done()
        while not self._data_queue.empty():
            self._data_queue.get_nowait()
            self._data_queue.task_done()

    def _collect_data(self, obs, rew, reset, extras, states):
        """ Helper function to combine buffers into a single dictionary.

        Args:
            obs (Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rew (Union[numpy.ndarray, torch.Tensor]: Buffer of rewards data.
            reset (Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            extras (dict): Dictionary of extras data.
            states (Union[numpy.ndarray, torch.Tensor]): Buffer of states data.

        Returns:
            data (dict): Dictionary containing all task buffers.
        """

        data = dict()
        data["obs"] = obs.clone()
        data["rew"] = rew.clone()
        data["reset"] = reset.clone()
        data["extras"] = extras.copy()
        data["states"] = states.clone()

        return data

    def run(self, trainer):
        """ Main loop for controlling simulation and task stepping.
            This method is responsible for starting simulation, stepping task and simulation, 
            collecting buffers from task, sending data to policy, and retrieving actions from policy.
            It also deals with the case when the policy terminates on completion and continues
            the simulation thread so that UI does not get affected.

            Args:
                trainer (TrainerMT): A Trainer object that implements APIs for starting and stopping RL thread.

        """

        from omni.isaac.core.simulation_context.simulation_context import SimulationContext

        trainer_initialized = False
        self._world.reset()

        frames_stopped = 0
        while self._simulation_app.is_running():
            try:
                if self._world.is_playing():
                    frames_stopped = 0
                    # initialize sim on first step
                    if self._world.get_physics_context()._use_flatcache:
                        self._world.get_physics_context().enable_flatcache(True)
                    if self._world.current_time_step_index == 0:
                        self._world.reset(soft=False)
                    actions = self.get_actions()
                    self._task.pre_physics_step(actions)
                    for _ in range(self._task.control_frequency_inv):
                        self._world.step(render=self._render)
                        self.sim_frame_count += 1
                    obs, rew, reset, extras = self._task.post_physics_step()
                    states = self._task.get_states()
                    data = self._collect_data(obs, rew, reset, extras, states)
                    self.send_data(data)
                elif self._world.is_stopped():
                    # terminate process in headless mode
                    if not self._render:
                        self._simulation_app.close()
                        return
                    if self._world.get_physics_context()._use_flatcache:
                        self._world.get_physics_context().enable_flatcache(False)
                    if trainer:
                        # this means simulation was stopped from UI - send stop signal to RL thread
                        if not self._stop and frames_stopped == 0:
                            self.send_data(None, block=False)
                            self.get_actions(block=False)
                            trainer_initialized = False
                        # RL thread already stopped, but trainer not initialized yet
                        elif self._stop and not trainer_initialized:
                            # start trainer - trainer must start before simulation to prevent deadlock. This prepares for simulation restart.
                            trainer.run()
                            trainer_initialized = True
                    # do not trigger task functions when simulation not running
                    SimulationContext.step(self._world, render=self._render)
                    frames_stopped += 1
                elif self._render:
                    SimulationContext.render(self._world)
            # signals task stopped
            except TaskStopException:
                if trainer:
                    trainer.stop()
                    trainer_initialized = False
                self._world.stop()
                continue
        self._simulation_app.close()
