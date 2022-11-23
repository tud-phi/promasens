from functools import partial
from pytorch3d.transforms import euler_angles_to_matrix, Rotate, Translate
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
import torch
from torchquad import Simpson, set_up_backend
from typing import *

from .base_kinematics import BaseKinematics


# prepare torchquad backend
set_up_backend("torch", data_type="float32")
# Simpson integrator
simp = Simpson()
# prepare jit compilation of the integration
# jit_int_fun = simp.get_jit_compiled_integrate(
#     dim=1, N=101, backend="torch"
# )


class AcKinematics(BaseKinematics):
    """
    Affine curvature kinematics for one segment in the quasi-2D case
    The configuration consists of 4 parameters:
        q = [theta0, theta1, phi, delta_L]
    where theta0 and theta1 are the parameters of the affine curvature: kappa = theta0 + theta1 * s,
    phi is the azimuth angle of bending and delta_L is the change in length from the reference length L0
    """

    def __init__(self, L0: float, num_int_points: int = 101):
        super(AcKinematics, self).__init__()

        self.L0 = L0
        self.num_int_points = num_int_points

    def forward_kinematics_batched(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward kinematics with affine curvature for one segment in a quasi-2D case
        :param q: torch tensor of size (N, 4, ) containing the configuration variables
        :param v: torch tensor of size (N, ) containing the positions of the points in the interval [0, 1]
        :return: torch tensor of size (N, 4, 4) containing the transformation matrix
        """

        # curvature at each integration point
        # kappa = q[0] + q[1] * s
        # the orientation is the integral of the curvature
        alpha = q[..., 0] * v + q[..., 1] * v ** 2 / 2

        # integrated bending coordinates
        xy_norm_list = []
        z_norm_list = []
        for idx in range(v.size(0)):
            v_ = v[idx]
            q_ = q[idx]

            # define partial functions to integrate
            sin_alpha_fun = partial(_compute_sin_alpha, q_)
            cos_alpha_fun = partial(_compute_cos_alpha, q_)

            # implementation with jitted function has some issues as it gets collected by garbage collector in some cases
            # xy_norm_list.append(jit_int_fun(sin_alpha_fun, integration_domain=q.new_tensor([[0, v_]])))
            # z_norm_list.append(jit_int_fun(cos_alpha_fun, integration_domain=q.new_tensor([[0, v_]])))

            xy_norm_v = simp.integrate(
                sin_alpha_fun,
                integration_domain=q.new_tensor([[0, v_]]),
                dim=1,
                N=self.num_int_points,
                backend="torch",
            )
            z_norm_v = simp.integrate(
                cos_alpha_fun,
                integration_domain=q.new_tensor([[0, v_]]),
                dim=1,
                N=self.num_int_points,
                backend="torch",
            )
            xy_norm_list.append(xy_norm_v)
            z_norm_list.append(z_norm_v)

        xy_norm = torch.stack(xy_norm_list, dim=0).to(device=q.device)
        z_norm = torch.stack(z_norm_list, dim=0).to(device=q.device)

        t = torch.einsum("i,ij -> ij", self.L0 + q[..., -1], torch.stack([
            xy_norm * torch.sin(q[..., 2]),
            - xy_norm * torch.cos(q[..., 2]),
            z_norm
        ], dim=-1))

        # we first rotate around the z-axis by the azimuth angle
        # then we apply the rotation alpha by rotating around the y-axis
        # we finally rotate back around the z-axis by the negative azimuth angle
        R = euler_angles_to_matrix(torch.stack([q[..., 2], alpha, -q[..., 2]], dim=-1), convention="ZXZ")

        T = torch.cat([
            torch.cat([R, t.unsqueeze(dim=-1)], dim=-1),
            q.new_tensor([[0.0, 0.0, 0.0, 1.0]]).unsqueeze(0).expand(v.size(0), -1, -1),
        ], dim=-2)

        return T


def _compute_alpha(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Compute the orientation alpha for the point s
    :param q: configuration of size (4, )
    :param s: position of the point in the interval [0, 1]. Must be a scalar
    """
    # the orientation is the integral of the curvature
    alpha = q[..., 0] * s + q[..., 1] * s ** 2 / 2
    return alpha


def _compute_cos_alpha(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.cos(_compute_alpha(q, s))


def _compute_sin_alpha(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.sin(_compute_alpha(q, s))
