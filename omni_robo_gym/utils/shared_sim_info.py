from control_cluster_bridge.utilities.shared_mem import SharedMemSrvr, SharedMemClient, SharedStringArray

import torch

class SharedSimInfo:
      
    def __init__(self, 
                is_client = False):
        
        self._terminate = False
        
        self.is_client = is_client

        # this creates a shared memory block of the right size for the state
        # and a corresponding view of it

        self.init = None                                                  

        self.shared_sim_datanames = None    

        if self.is_client:
            
            self.shared_sim_data = SharedMemClient(name="SharedSimInfo",
                                        namespace="", 
                                        dtype=torch.float32, 
                                        verbose=True)
        else: 
            
            self.init = ["gpu_pipeline_active [>0 True]", \
                    "integration_dt [s]", \
                    "rendering_dt [s]", 
                    "cluster_dt [s]", 
                    "sim_rt_factor", 
                    "cumulative_rt_factor", 
                    "time_for_sim_stepping [s]"]
                    
            self.shared_sim_datanames = SharedStringArray(len(self.init), 
                                        "SharedSimInfoNames", 
                                        is_server=not self.is_client, 
                                        namespace="", 
                                        verbose=True)
                                                 
            self.shared_sim_data = SharedMemSrvr(n_rows=1, 
                                        n_cols=len(self.init), 
                                        name="SharedSimInfo",
                                        namespace="", 
                                        dtype=torch.float32)
    
    def start(self, 
            gpu_pipeline_active: bool = None, 
            integration_dt: float = None, 
            rendering_dt: float = None, 
            cluster_dt: float = None):

        if self.is_client:
            
            self.shared_sim_data.attach() 
            
            self.shared_sim_datanames = SharedStringArray(len(self.shared_sim_data.tensor_view[0, :]), 
                                        "SharedSimInfoNames", 
                                        is_server=not self.is_client, 
                                        namespace="", 
                                        verbose=True)
            

            self.shared_sim_datanames.start()

        else:

            self.shared_sim_datanames.start(self.init) 

            self.shared_sim_data.start() 
        
            # write static information
            self.shared_sim_data.tensor_view[0, 0] = gpu_pipeline_active
            self.shared_sim_data.tensor_view[0, 1] = integration_dt
            self.shared_sim_data.tensor_view[0, 2] = rendering_dt
            self.shared_sim_data.tensor_view[0, 3] = cluster_dt

    def update(self, 
               sim_rt_factor, 
               cumulative_rt_factor,
               time_for_sim_stepping):

        # write runtime information

        self.shared_sim_data.tensor_view[0, 4] = sim_rt_factor
        self.shared_sim_data.tensor_view[0, 5] = cumulative_rt_factor
        self.shared_sim_data.tensor_view[0, 6] = time_for_sim_stepping        

    def get_names(self):
    
        return self.shared_sim_datanames.read()

    def is_gpu_pipeline(self):

        return self.shared_sim_data.tensor_view[0, 0]
            
    def get_integration_dt(self):
            
        return self.shared_sim_data.tensor_view[0, 1]
    
    def get_rendering_dt(self):

        return self.shared_sim_data.tensor_view[0, 2]
    
    def get_cluster_dt(self):

        return self.shared_sim_data.tensor_view[0, 3]

    def get_sim_rt_factor(self):

        return self.shared_sim_data.tensor_view[0, 4]

    def get_cumulative_rt_factor(self):

        return self.shared_sim_data.tensor_view[0, 5]

    def get_time_for_stepping(self):

        return self.shared_sim_data.tensor_view[0, 6]
            
    def terminate(self):
        
        if not self._terminate:

            self._terminate = True

            self.shared_sim_datanames.terminate()

            self.shared_sim_data.terminate()

    def __del__(self):
        
        self.terminate()
