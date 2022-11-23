import numpy as np
import torch
from torch import linalg, nn
from typing import *

from .base_kinematics import BaseKinematics
from .robot_kinematics import RobotKinematics


class SensorMagnetKinematics(BaseKinematics):
    def __init__(self, robot_kinematics: RobotKinematics):
        super(SensorMagnetKinematics, self).__init__()

        self.robot_kinematics = robot_kinematics
        self.phi_e, self.n_e = robot_kinematics.phi_e, robot_kinematics.n_e
        if isinstance(self.n_e, np.ndarray):
            self.n_e = torch.tensor(self.n_e)

        self.dim_xi_jk = 4

    def forward(self, q: torch.Tensor = None) -> torch.Tensor:
        if q is not None:
            T_base, T_tip = self.robot_kinematics.apply_configuration(q)
            T_sensors = self.robot_kinematics.forward_kinematics_sensors()
            T_magnets = self.robot_kinematics.forward_kinematics_magnets()
        else:
            q = self.robot_kinematics.q
            T_base, T_tip = self.robot_kinematics.T_base, self.robot_kinematics.T_tip
            T_sensors, T_magnets = self.robot_kinematics.T_sensors, self.robot_kinematics.T_magnets

        num_sensors = T_sensors.size(0)
        num_magnets = T_magnets.size(0)

        xi = torch.zeros((num_sensors, 1+num_magnets*self.dim_xi_jk), device=q.device, dtype=q.dtype)
        for j in range(num_sensors):
            lambda_j = self.compute_lambda_j(T_sensors[j])
            # lambda_j = torch.tensor(0.)

            xi_j_list = [lambda_j.unsqueeze(0)]
            for k in range(num_magnets):
                xi_jk = self.compute_xi_jk(T_sensors[j], T_magnets[k])
                xi_j_list.append(xi_jk)
            xi_j = torch.cat(xi_j_list, dim=0)
            xi[j, :] = xi_j

        return xi

    def compute_xi_jk(self, T_sensor: torch.Tensor, T_magnet: torch.Tensor) -> torch.Tensor:
        """
        Computes the kinematic parametrization of the spatial relationship between magnet and sensor
        """
        o_s_j = T_sensor[:3, 2:3]
        o_m_k = T_magnet[:3, 2:3]

        # angle between the magnet direction and the sensor measurement direction:
        alpha_jk = self.compute_angle_between_vectors_with_eps(o_m_k, o_s_j)

        # translation between sensor j and magnet k in the inertial frame
        t_jk = T_sensor[:3, 3:4] - T_magnet[:3, 3:4]

        # distance between sensor_j and magnet_k:
        d_jk = linalg.vector_norm(t_jk, ord=2)

        # angle between cylindrical axis of the magnet and sensor
        theta_jk = self.compute_angle_between_vectors_with_eps(t_jk / d_jk, o_m_k)

        # angle between the normal measurement direction of the sensor and the vector from the magnet to the sensor:
        beta_jk = self.compute_angle_between_vectors_with_eps(t_jk / d_jk, o_s_j)

        xi_jk = torch.stack([d_jk, alpha_jk, beta_jk, theta_jk], dim=0)
        return xi_jk

    def compute_lambda_j(self, T_sensor: torch.Tensor) -> torch.Tensor:
        # measurement direction of sensor in the inertial frame
        o_s_j = T_sensor[:3, 2:3]

        # unit vector of earth magnetic field in the base frame
        if self.n_e is not None:
            n_e = self.n_e.to(device=T_sensor.device).to(dtype=T_sensor.dtype)
        else:
            # assuming that the earth magnetic field does not contain any vertical components
            if self.phi_e is None:
                phi_e = torch.tensor([0.], dtype=T_sensor.dtype, device=T_sensor.device)
            else:
                phi_e = torch.tensor([self.phi_e], dtype=T_sensor.dtype, device=T_sensor.device)

            n_e = torch.cat([torch.cos(phi_e), torch.sin(phi_e),
                             torch.tensor([0], dtype=T_sensor.dtype, device=T_sensor.device)], dim=0).unsqueeze(dim=1)

        # angle between measurement direction and direction of earths magnetic field (e):
        lambda_j = self.compute_angle_between_vectors_with_eps(o_s_j, n_e)

        return lambda_j

    def compute_angle_between_vectors_with_eps(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Computes the angle between two vectors using the dot product with eps added to avoid out-of-range errors
        """
        # prevent out-of-range errors because of eps added to cc kinematics with clamp
        dot_prod = torch.clamp(torch.dot(v1.squeeze(), v2.squeeze()), -1.0, 1.0)
        angle = torch.acos(dot_prod - torch.sign(dot_prod) * self.eps)
        return angle
