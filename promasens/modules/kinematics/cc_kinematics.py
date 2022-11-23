import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix
# https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
from scipy.spatial.transform import Rotation
import torch
from torch import linalg, nn
from torch.autograd.functional import jacobian
from typing import *


from .base_kinematics import BaseKinematics


class CcKinematics(BaseKinematics):
    """
    Constant curvature kinematics for one segment
    """

    def __init__(self, L0: float, d: float):
        super(CcKinematics, self).__init__()

        self.L0 = L0
        self.d = d

    def forward_kinematics_batched(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward kinematics for one segment according to constant curvature assumption
        :param q: torch tensor of size (n, 4, ) containing the configuration variables
        :param v: torch tensor of size (n, ) containing the positions of the points in the interval [0, 1]
        :return: torch tensor of size (n, 4, 4) containing the transformation matrix
        """
        n = q.size(0)  # batch size

        Delta_x = q[:, 0]
        Delta_y = q[:, 1]
        delta_L = q[:, 2]

        Delta_x_a = v * Delta_x + self.eps
        Delta_y_a = v * Delta_y + self.eps
        Delta_norm_a = torch.sqrt(Delta_x_a ** 2 + Delta_y_a ** 2)

        S_v = torch.sin(Delta_norm_a / self.d)
        C_v = torch.cos(Delta_norm_a / self.d)

        R1 = torch.stack(
            [1 + Delta_x_a ** 2 / Delta_norm_a ** 2 * (C_v - 1), Delta_x_a * Delta_y_a / Delta_norm_a ** 2 * (C_v - 1),
             Delta_x_a / Delta_norm_a * S_v], dim=1)
        R2 = torch.stack(
            [Delta_x_a * Delta_y_a / Delta_norm_a ** 2 * (C_v - 1), 1 + Delta_y_a ** 2 / Delta_norm_a ** 2 * (C_v - 1),
             Delta_y_a / Delta_norm_a * S_v], dim=1)
        R3 = torch.stack([-Delta_x_a / Delta_norm_a * S_v, -Delta_y_a / Delta_norm_a * S_v, C_v], dim=1)
        R = torch.stack([R1, R2, R3], dim=1)

        f_t = v * self.d * (self.L0 + delta_L) / (Delta_norm_a ** 2)
        t = torch.stack([f_t * Delta_x_a * (1 - C_v), f_t * Delta_y_a * (1 - C_v), f_t * Delta_norm_a * S_v], dim=1)
        t = t.unsqueeze(dim=2)

        T = q.new_zeros((n, 4, 4))
        T[:, 3, 3] = q.new_ones(n)
        T[:, :3, :3] = R
        T[:, :3, 3:4] = t

        return T

    def inverse_kinematics(self, t_r: torch.Tensor, R_r: torch.Tensor, l_r: float) -> torch.Tensor:
        if abs(R_r[2, 2]) >= 1 and R_r[2, 2] > 0:
            R_r[2, 2] = R_r[2, 2] - pow(10, 3) * self.eps
        elif abs(R_r[2, 2]) >= 1 and R_r[2, 2] < 0:
            R_r[2, 2] = R_r[2, 2] + pow(10, 3) * self.eps

        delta_L_r = t_r[2] * (torch.acos(R_r[2, 2]) / torch.sin(torch.acos(R_r[2, 2]))) - l_r

        delta_c_r = (self.d / (l_r + delta_L_r)) * ((torch.acos(R_r[2, 2])) ** 2 / (1 - R_r[2, 2]))

        delta_x_r = t_r[0] * delta_c_r
        delta_y_r = t_r[1] * delta_c_r

        q_r = torch.stack([delta_x_r, delta_y_r, delta_L_r])
        q = (self.L0 / l_r) * q_r

        return q
