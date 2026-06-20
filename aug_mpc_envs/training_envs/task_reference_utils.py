import torch


def position_target_to_velocity(target_xy: torch.Tensor,
        robot_xy: torch.Tensor,
        max_dp: float,
        max_dt: float,
        eps: float = 1e-6):
    delta = target_xy - robot_xy
    distance = delta.norm(dim=1, keepdim=True)
    direction = delta / distance.clamp_min(eps)
    limited_distance = distance.clamp_max(max_dp)
    velocity = limited_distance * direction / max_dt
    return delta, limited_distance, direction, velocity


def resample_bernoulli_coefficients(coefficients: torch.Tensor,
        probabilities: torch.Tensor,
        selector=None):
    if selector is None:
        torch.bernoulli(input=probabilities, out=coefficients)
    else:
        coefficients[selector, :] = torch.bernoulli(probabilities[selector, :])
