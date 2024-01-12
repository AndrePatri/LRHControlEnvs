import xml.etree.ElementTree as ET

class UrdfLimitsParser:
    def __init__(self, urdf_path, joint_names,
                backend = "numpy"):
        self.urdf_path = urdf_path
        self.joint_names = joint_names
        self.limits_matrix = None

        self.backend = backend

        self.parse_urdf()

    def parse_urdf(self):
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        num_joints = len(self.joint_names)

        self.limits_matrix = None
        self.nan = None

        if self.backend == "numpy":

            import numpy as np

            self.limits_matrix = np.full((num_joints, 6), np.nan)
            
            self.nan = np.nan

        elif self.backend == "torch":
            
            import torch

            self.limits_matrix = torch.full((num_joints, 6), torch.nan)

            self.nan = torch.nan

        else:

            raise Exception("Backend not supported")

        for joint_name in self.joint_names:

            joint_element = root.find(".//joint[@name='{}']".format(joint_name))

            if joint_element is not None:

                limit_element = joint_element.find('limit')

                jnt_index = self.joint_names.index(joint_name)

                # position limits
                
                q_lower = float(limit_element.get('lower', self.nan))
                q_upper = float(limit_element.get('upper', self.nan))

                # effort limits
                effort_limit = float(limit_element.get('effort', self.nan))
                
                # vel limits
                velocity_limit = float(limit_element.get('velocity', self.nan))

                self.limits_matrix[jnt_index, 0] = q_lower
                self.limits_matrix[jnt_index, 3] = q_upper
                self.limits_matrix[jnt_index, 1] = - abs(effort_limit)
                self.limits_matrix[jnt_index, 4] = abs(effort_limit)
                self.limits_matrix[jnt_index, 2] = - abs(velocity_limit)
                self.limits_matrix[jnt_index, 5] = abs(velocity_limit)

    def get_limits_matrix(self):
        return self.limits_matrix

# Example usage
urdf_parser = UrdfLimitsParser("/home/apatrizi/Desktop/kyon.urdf", ["knee_pitch_4", "hip_roll_4", "wheel_joint_2"])
limits_matrix = urdf_parser.get_limits_matrix()

print("Joint Limits Matrix:")
print(limits_matrix)

