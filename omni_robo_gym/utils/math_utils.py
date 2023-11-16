import torch
import time

def normalize_quaternion(q):
    # Normalizes the quaternion
    return q / torch.norm(q, dim=-1, keepdim=True)

def quaternion_difference(q1, q2):
    """ Compute the quaternion difference needed to rotate from q1 to q2 """
    def quat_conjugate(q):
        # Computes the conjugate of a quaternion
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    q1_conj = quat_conjugate(q1)

    return quaternion_multiply(q2, q1_conj)

def quaternion_multiply(q1, q2):
    """ Multiply two quaternions. """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quaternion_to_angular_velocity(q_diff, dt):
    """ Convert a quaternion difference to an angular velocity vector. """
    angle = 2 * torch.arccos(q_diff[..., 0].clamp(-1.0, 1.0))  # Clamping for numerical stability
    axis = q_diff[..., 1:]
    norm = axis.norm(dim=-1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    axis = axis / norm
    angle = angle.unsqueeze(-1)  # Add an extra dimension for broadcasting
    return (angle / dt) * axis

def quat_to_omega(q0, q1, dt):
    """ Convert quaternion pairs to angular velocities """
    if q0.shape != q1.shape:
        raise ValueError("Tensor shapes do not match in quat_to_omega.")

    # Normalize quaternions and compute differences
    q0_normalized = normalize_quaternion(q0)
    q1_normalized = normalize_quaternion(q1)
    q_diff = quaternion_difference(q0_normalized, q1_normalized)

    return quaternion_to_angular_velocity(q_diff, dt)

# Example usage
n_envs = 100  # Number of environments
dt = 0.1    # Time step

# Random example tensors for initial and final orientations
q_initial = torch.randn(n_envs, 4)
q_final = torch.randn(n_envs, 4)

start_time = time.perf_counter()
# Convert to angular velocities
omega = quat_to_omega(q_initial, q_final, dt)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Time taken to compute angular velocities: {elapsed_time:.6f} seconds")
