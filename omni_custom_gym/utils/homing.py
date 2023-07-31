from omni.isaac.core.articulations.articulation_view import ArticulationView

import torch

import xml.etree.ElementTree as ET

class OmniRobotHomer:

    def __init__(self, 
            articulation: ArticulationView, 
            srdf_path: str, 
            backend = "torch", 
            device: torch.device = torch.device("cpu")):

        self._info = "info"
        self._status = "status"
        self._warning = "warning" 
        self._exception = "exception"

        if not articulation.initialized:

            raise Exception(f"[{self.__class__.__name__}]" + f"[{self.exception}]" + ": the provided articulation is not initialized properly!!")
        
        self._articulation = articulation
        self.srdf_path = srdf_path

        self._device = device

        self.num_robots = self._articulation.count
        self.n_dofs = self._articulation.num_dof
        self.jnts_names = self._articulation.dof_names

        self.joint_idx_map = {}
        for joint in range(0, self.n_dofs):

            self.joint_idx_map[self.jnts_names[joint]] = joint 

        if (backend != "torch"):

            print(f"[{self.__class__.__name__}]"  + f"[{self.info}]" + ": forcing torch backend. Other backends are not yet supported.")
        
        self._backend = "torch"

        self._homing = torch.full((self.num_robots, self.n_dofs), 
                        0.0, 
                        device = self._device, 
                        dtype=torch.float32) # homing configuration
        
        # open srdf and parse the homing field
        
        with open(srdf_path, 'r') as file:
            
            self._srdf_content = file.read()

        try:
            self._srdf_root = ET.fromstring(self._srdf_content)
            # Now 'root' holds the root element of the XML tree.
            # You can navigate through the XML tree to extract the tags and their values.
            # Example: To find all elements with a specific tag, you can use:
            # elements = root.findall('.//your_tag_name')

            # Example: If you know the specific structure of your .SRDF file, you can extract
            # the data accordingly, for instance:
            # for child in root:
            #     if child.tag == 'some_tag_name':
            #         tag_value = child.text
            #         # Do something with the tag value.
            #     elif child.tag == 'another_tag_name':
            #         # Handle another tag.

        except ET.ParseError as e:
        
            print(f"[{self.__class__.__name__}]" + f"[{self._exception}]" + ": could not read SRDF properly!!")

        # Find all the 'joint' elements within 'group_state' with the name attribute and their values
        joints = self._srdf_root.findall(".//group_state[@name='home']/joint")

        self._homing_map = {}

        for joint in joints:
            joint_name = joint.attrib['name']
            joint_value = joint.attrib['value']
            self._homing_map[joint_name] =  float(joint_value)
        
        self._assign2homing()

    def _assign2homing(self):
        
        for joint in list(self._homing_map.keys()):
            
            if joint in self.joint_idx_map:
                
                self._homing[:, self.joint_idx_map[joint]] = torch.full((self.num_robots, 1), 
                                                                self._homing_map[joint],
                                                                device = self._device, 
                                                                dtype=torch.float32).flatten()
            else:

                print(f"[{self.__class__.__name__}]" + f"[{self._warning}]" + f"[{self._assign2homing.__name__}]" \
                      + ": joint " + f"{joint}" + " is not present in the articulation. It will be ignored.")
                
    def get_homing(self):

        return self._homing