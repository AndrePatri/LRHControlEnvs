from aug_mpc.utils.shared_data.agent_refs import AgentRefs
from aug_mpc.utils.shared_data.training_env import Actions

from mpc_hive.utilities.shared_data.rhc_data import RobotState
from mpc_hive.utilities.math_utils import world2base_frame_twist

from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import dtype

import math

import numpy as np

class AgentRefsFromZMQ:

    def __init__(self, 
                namespace: str, 
                verbose = False,
                agent_refs_world: bool = True,
                env_idx: int = None):

        self._env_idx=env_idx

        self._verbose = verbose

        self._agent_refs_world=agent_refs_world
        
        self.namespace = namespace

        self._closed = False
        
        self.enable_linvel = False
        self.enable_omega = False
        self.enable_omega_roll = False
        self.enable_omega_pitch = False
        self.enable_omega_yaw = False

        self.enable_pos = False

        self.dpos = 0.1 # [m]
        self.dxy = 0.05 # [m/s]
        self.dvxyz = 0.05 # [m/s]
        self.dheading=0.05
        self._dtwist = 1.0 * math.pi / 180.0 # [rad]

        self._v_magnitude=0.0
        self._heading_lat=0.0
        self._heading_frontal=0.0
        self._heading=0.0
        self.agent_refs = None

        self._max_vxy_magn=1.5 # [m/s]
        self._max_vz_magn=0.0
        self._max_pitch_rate=0.0 # [rad/s]
        self._max_roll_rate=0.0 # [rad/s]
        self._max_yaw_rate=0.8 # [rad/s]

        self.cluster_idx = -1
        self.cluster_idx_np = np.array(self.cluster_idx)

        self._twist_null = None

        self._init_shared_data()

    def _init_shared_data(self):
        
        self.env_index=None
        if self._env_idx is None:
            self.env_index = SharedTWrapper(namespace = self.namespace,
                    basename = "EnvSelector",
                    is_server = False, 
                    verbose = True, 
                    vlevel = VLevel.V2,
                    safe = False,
                    dtype=dtype.Int)
            
            self.env_index.run()
        
        self._init_rhc_ref_subscriber()
        
        self._current_twist_ref_world = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        self._current_twist_ref_base=np.full_like(self._current_twist_ref_world, fill_value=0.0).reshape(1, -1)

        self._current_pos_ref = np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="p", robot_idxs=self.cluster_idx_np), 
                fill_value=0.0).reshape(-1)
        
        self._robot_state = RobotState(namespace=self.namespace,
                            is_server=False, 
                            safe=False,
                            verbose=True,
                            vlevel=VLevel.V2)
        self._robot_state.run()            

    def _init_rhc_ref_subscriber(self):

        self.agent_refs = AgentRefs(namespace=self.namespace,
                                is_server=False, 
                                safe=True, 
                                verbose=self._verbose,
                                vlevel=VLevel.V2,
                                with_gpu_mirror=False,
                                with_torch_view=False)

        self.agent_refs.run()

        self._twist_null = self.agent_refs.rob_refs.root_state.get(data_type="twist", robot_idxs=self.cluster_idx_np)
        self._twist_null[:]=0.0

        q0=np.full_like(self.agent_refs.rob_refs.root_state.get(data_type="q", robot_idxs=self.cluster_idx_np),fill_value=0.0)
        q0[0]=1.0
        self.agent_refs.rob_refs.root_state.set(data_type="q",data=q0,
                                        robot_idxs=self.cluster_idx_np)
        
    def __del__(self):

        if not self._closed:
            self._close()
    
    def _close(self):
        
        if self.agent_refs is not None:
            self.agent_refs.close()
        if self._robot_state is not None:
            self._robot_state.close()

        self._closed = True
    
    def _synch(self):
        
        if self.env_index is not None:
            self.env_index.synch_all(read=True, retry=True)
            env_index = self.env_index.get_numpy_mirror()
            self._env_idx=env_index[0, 0].item()
        self.cluster_idx = self._env_idx
        self.cluster_idx_np = self.cluster_idx            
    
    def _update_navigation(self, 
                    nav_type: str = "",
                    increment = True,
                    reset: bool = False,
                    refs_in_wframe: bool = False):
        
        current_twist_ref=self._current_twist_ref_world

        # randomizng in base frame if not refs_in_wframe, otherwise world
        if not reset:

            # xy vel (polar coordinates)
            if nav_type=="frontal" and not increment:
                if self._heading_lat>0:
                    self._heading_lat=self._heading_lat + self.dheading
                else:
                    self._heading_lat=self._heading_lat - self.dheading
                self._heading_frontal=self._heading_lat-np.pi/2
            if nav_type=="frontal" and increment:
                if self._heading_lat>0:
                    self._heading_lat=self._heading_lat - self.dheading
                else:
                    self._heading_lat=self._heading_lat + self.dheading
                self._heading_frontal=self._heading_lat-np.pi/2
            if nav_type=="lateral" and not increment:
                if self._heading_frontal>0:
                    self._heading_frontal=self._heading_frontal + self.dheading
                else:
                    self._heading_frontal=self._heading_frontal - self.dheading
                self._heading_lat=self._heading_frontal+np.pi/2
            if nav_type=="lateral" and increment:
                if self._heading_frontal>0:
                    self._heading_frontal=self._heading_frontal - self.dheading
                else:
                    self._heading_frontal=self._heading_frontal + self.dheading
                self._heading_lat=self._heading_frontal+np.pi/2
            
            if nav_type=="magnitude" and increment:
                self._v_magnitude=self._v_magnitude+self.dxy
            if nav_type=="magnitude" and not increment:
                self._v_magnitude=self._v_magnitude-self.dxy

            # vertical vel
            if nav_type=="vertical" and not increment:
                # frontal motion
                current_twist_ref[2] = current_twist_ref[2] - self.dvxyz
            if nav_type=="vertical" and increment:
                # frontal motion
                current_twist_ref[2] = current_twist_ref[2] + self.dvxyz

            # omega
            if nav_type=="twist_roll" and increment:
                # rotate counter-clockwise
                current_twist_ref[3] = current_twist_ref[3] + self._dtwist 
            if nav_type=="twist_roll" and not increment:
                current_twist_ref[3] = current_twist_ref[3] - self._dtwist 
            if nav_type=="twist_pitch" and increment:
                # rotate counter-clockwise
                current_twist_ref[4] = current_twist_ref[4] + self._dtwist 
            if nav_type=="twist_pitch" and not increment:
                current_twist_ref[4] = current_twist_ref[4] - self._dtwist 
            if nav_type=="twist_yaw" and increment:
                # rotate counter-omega_cmd
                current_twist_ref[5] = current_twist_ref[5] + self._dtwist
            if nav_type=="twist_yaw" and not increment:
                current_twist_ref[5] = current_twist_ref[5] - self._dtwist 

        else:
            
            if "lin" in nav_type:
                self._v_magnitude=0.0
                self._heading=0.0
                self._heading_frontal=0.0
                self._heading_lat=self._heading_frontal+np.pi/2
                
                current_twist_ref[0:3] = 0
                if self._agent_refs_world:
                    self._current_twist_ref_world[0:3]=0
            
            if "omega" in nav_type:
                current_twist_ref[3:] = 0
                if self._agent_refs_world:
                    self._current_twist_ref_world[3:]=0

        if self._heading_frontal>math.pi:
            self._heading_frontal=math.pi
        if self._heading_frontal<-math.pi:
            self._heading_frontal=-math.pi
        if self._heading_lat>math.pi:
            self._heading_lat=math.pi
        if self._heading_lat<-math.pi:
            self._heading_lat=-math.pi
                    
        self._heading=self._heading_frontal

        self._v_magnitude=np.clip(self._v_magnitude, a_min=0.0, a_max=self._max_vxy_magn)
        current_twist_ref[0] = self._v_magnitude*np.cos(self._heading)
        current_twist_ref[1] = self._v_magnitude*np.sin(self._heading)

        current_twist_ref[2]=np.clip(current_twist_ref[2], a_min=0.0, a_max=self._max_vz_magn)
        current_twist_ref[3]=np.clip(current_twist_ref[3], a_min=0.0, a_max=self._max_roll_rate)
        current_twist_ref[4]=np.clip(current_twist_ref[4], a_min=0.0, a_max=self._max_pitch_rate)
        current_twist_ref[5]=np.clip(current_twist_ref[5], a_min=0.0, a_max=self._max_yaw_rate)

    def _update_pos(self, 
        nav_type: str = "",
        increment = True,
        reset: bool = False):
        
        current_pos_ref=self._current_pos_ref

        if not reset:
            # xy vel
            if nav_type=="lateral" and not increment:
                current_pos_ref[1]-=self.dpos
            if nav_type=="lateral" and increment:
                current_pos_ref[1]+=self.dpos
            if nav_type=="frontal" and not increment:
                current_pos_ref[0]-=self.dpos
            if nav_type=="frontal" and increment:
                current_pos_ref[0]+=self.dpos
            if nav_type=="vertical" and not increment:
                current_pos_ref[2]-=self.dpos
            if nav_type=="vertical" and increment:
                current_pos_ref[2]+=self.dpos
        else:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            current_pos_ref[:]=robot_p
             
    def _set_omega(self, 
                key):
        
        if key == "T":
            self.enable_omega = not self.enable_omega
            info = f"Twist change enabled: {self.enable_omega}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if not self.enable_omega:
            self._update_navigation(nav_type="omega",reset=True)

        if self.enable_omega and key == "x":
            self.enable_omega_roll = not self.enable_omega_roll
            info = f"Twist roll change enabled: {self.enable_omega_roll}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "y":
            self.enable_omega_pitch = not self.enable_omega_pitch
            info = f"Twist pitch change enabled: {self.enable_omega_pitch}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        if self.enable_omega and key == "z":
            self.enable_omega_yaw = not self.enable_omega_yaw
            info = f"Twist yaw change enabled: {self.enable_omega_yaw}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)

        if key == "9" and self.enable_omega:
            if self.enable_omega_roll:
                self._update_navigation(nav_type="twist_roll",
                                increment = True)
            if self.enable_omega_pitch:
                self._update_navigation(nav_type="twist_pitch",
                                    increment = True)
            if self.enable_omega_yaw:
                self._update_navigation(nav_type="twist_yaw",
                                    increment = True)
        if key == "3" and self.enable_omega:
            if self.enable_omega_roll:
                self._update_navigation(nav_type="twist_roll",
                                    increment = False)
            if self.enable_omega_pitch:
                self._update_navigation(nav_type="twist_pitch",
                                    increment = False)
            if self.enable_omega_yaw:
                self._update_navigation(nav_type="twist_yaw",
                                    increment = False)
               
    def _set_linvel(self,
                key):
        if key == "n":
            self.enable_linvel = not self.enable_linvel
            info = f"High level navigation enabled: {self.enable_linvel}"
            Journal.log(self.__class__.__name__,
                "_set_linvel",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_linvel:
            self._update_navigation(nav_type="lin", reset = True)
            
        if key == "6" and self.enable_linvel:
            self._update_navigation(nav_type="lateral", 
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "4" and self.enable_linvel:
            self._update_navigation(nav_type="lateral",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "8" and self.enable_linvel:
            self._update_navigation(nav_type="frontal",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "2" and self.enable_linvel:
            self._update_navigation(nav_type="frontal",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "+" and self.enable_linvel:
            self._update_navigation(nav_type="magnitude",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "-" and self.enable_linvel:
            self._update_navigation(nav_type="magnitude",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        if key == "7" and self.enable_linvel:
            self._update_navigation(nav_type="vertical",
                            increment = True,
                            refs_in_wframe=self._agent_refs_world)
        if key == "1" and self.enable_linvel:
            self._update_navigation(nav_type="vertical",
                            increment = False,
                            refs_in_wframe=self._agent_refs_world)
        
    def _set_position(self,key):

        if key == "P":
            self.enable_pos = not self.enable_pos
            info = f"High level pos reference change: {self.enable_pos}"
            Journal.log(self.__class__.__name__,
                "set_position",
                info,
                LogType.INFO,
                throw_when_excep = True)
        
        if not self.enable_pos:
            self._update_pos(reset = True)
            
        if key == "6" and self.enable_pos:
            self._update_pos(nav_type="lateral", 
                            increment = False)
        if key == "4" and self.enable_pos:
            self._update_pos(nav_type="lateral", 
                            increment = True)
        if key == "8" and self.enable_pos:
            self._update_pos(nav_type="frontal", 
                            increment = True)
        if key == "2" and self.enable_pos:
            self._update_pos(nav_type="frontal", 
                            increment = False)
        if key == "+" and self.enable_pos:
            self._update_pos(nav_type="vertical", 
                            increment = True)
        if key == "-" and self.enable_pos:
            self._update_pos(nav_type="vertical", 
                            increment = False)
            
    def _on_press(self, key):

        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char

        self._set_linvel(key)
        self._set_omega(key)
        self._set_position(key)

    def _on_release(self, key):
        
        if not self._read_from_stdin:
            if hasattr(key, 'char'):
                key=key.char
                
        # self._current_twist_ref_base[:, :]=0.0
        # nullify vel ref
        # self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._twist_null,
        #                     robot_idxs=self.cluster_idx_np)

    def _write_to_shared_mem(self):

        self.agent_refs.rob_refs.root_state.synch_all(read=True)
        self._robot_state.root_state.synch_all(read = True, retry = True) # read robot state        
        
        if self.enable_pos:
            robot_p = self._robot_state.root_state.get(data_type="p")[self.cluster_idx_np, :].reshape(-1)
            robot_p[2]=0.0
            # self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref-robot_p,
            #                                 robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.set(data_type="p",data=self._current_pos_ref,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=0, 
                                        n_rows=1, n_cols=3,
                                        read=False)
            
        if self.enable_omega or self.enable_linvel: # angular velocity or linear
            if self._agent_refs_world:
                # ref was set in world frame -> we need to move it in base frame before setting it to the agent
                robot_q = self._robot_state.root_state.get(data_type="q")[self.cluster_idx_np, :].reshape(1, -1)
                world2base_frame_twist(t_w=self._current_twist_ref_world.reshape(1, -1), 
                    q_b=robot_q, 
                    t_out=self._current_twist_ref_base)
                    
                self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                                robot_idxs=self.cluster_idx_np)
            else:
                self._current_twist_ref_base[:, :]=self._current_twist_ref_world.reshape(1, -1)
            self.agent_refs.rob_refs.root_state.set(data_type="twist",data=self._current_twist_ref_base,
                                            robot_idxs=self.cluster_idx_np)
            self.agent_refs.rob_refs.root_state.synch_retry(row_index=self.cluster_idx, col_index=7, 
                                        n_rows=1, n_cols=6,
                                        read=False)
              
    def run(self,
        release_timeout: float = 0.1):

        info = f"Ready. Starting to listen for commands..."

        Journal.log(self.__class__.__name__,
            "run",
            info,
            LogType.INFO,
            throw_when_excep = True)
        
        self._update_navigation(reset=True)
        
        
        import time

        self.agent_refs.run()
        
        from AugMPCEnvs.aug_mpc_envs.utils.listener_xbot_zmq import KeyListenerXBot2ZMQ

        with KeyListenerXBot2ZMQ(on_press=self._on_press, 
            on_release=self._on_release, 
            release_timeout=release_timeout) as listener:
            
            while not listener.done:
                self._synch() 
                self._write_to_shared_mem()
                time.sleep(0.01)  # Keep the main thread alive
            listener.stop()