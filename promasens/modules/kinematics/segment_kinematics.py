import numpy as np
import torch
from torch import nn
from typing import *


class SegmentKinematics(nn.Module):
    def __init__(self, L0: float, d: float, num_sensors: int, kinematic_parametrization: str = "cc",
                 **kwargs):
        super(SegmentKinematics, self).__init__()

        self.L0, self.d = L0, d
        if kinematic_parametrization == "cc":
            from .cc_kinematics import CcKinematics
            self.kinematic_class = CcKinematics(L0, d)
        elif kinematic_parametrization == "ac":
            from .ac_kinematics import AcKinematics
            self.kinematic_class = AcKinematics(L0)
        else:
            raise ValueError(f"The stated kinematic parametrization {kinematic_parametrization} is not implemented.")

        # initialize magnet parameters, but they will be overwritten later
        self.num_magnets = 1
        self.d_m = None

        self.num_sensors = num_sensors
        self.sensor_polarization = -1

        self.phi_off = 0.
        self.psi_s = 0.  # polar tilt angle of the sensor with respect to backbone plane
        self.set_segment_params(num_sensors=num_sensors, **kwargs)

        self.q, self.T = None, None
        self.apply_configuration(torch.zeros(3))

    def set_segment_params(self, d_m: Union[float, List, np.ndarray],
                           d_s_r: Union[float, List, np.ndarray], d_s_a: Union[float, List, np.ndarray],
                           num_sensors: int = None, phi_off: float = 0.,
                           psi_s: Union[float, List, np.ndarray] = 0.0, **kwargs):
        if isinstance(d_m, float):
            self.num_magnets = 1
            self.d_m = torch.ones((self.num_magnets, )) * d_m  # distance from base to magnet along backbone
        else:
            self.num_magnets = len(d_m)
            self.d_m = torch.tensor(d_m)

        if num_sensors is not None:
            self.num_sensors = num_sensors

        if isinstance(d_s_r, float):
            self.d_s_r = torch.ones(self.num_sensors) * d_s_r
        else:
            self.d_s_r = torch.tensor(d_s_r)

        if isinstance(d_s_a, float):
            self.d_s_a = torch.ones(self.num_sensors) * d_s_a
        else:
            self.d_s_a = torch.tensor(d_s_a)

        if isinstance(psi_s, float):
            self.psi_s = torch.ones(self.num_sensors) * psi_s
        else:
            self.psi_s = torch.tensor(psi_s)

        self.phi_off = phi_off

        self.T_sensors_static = self.kinematic_class.compute_fixed_in_plane_transformations(
            n=num_sensors,
            d_r=torch.ones((num_sensors, )) * self.d_s_r,
            phi=self.phi_off + torch.arange(0, self.num_sensors) * 2 * torch.pi / self.num_sensors,
            psi=torch.ones((num_sensors, )) * self.psi_s,
            polarization=torch.ones((num_sensors, )) * self.sensor_polarization,
        )

        print("kwargs in set_segment_params (should be empty except for constant params):", kwargs)

    def apply_configuration(self, q: torch.Tensor) -> torch.Tensor:
        self.q = q
        self.T = self.kinematic_class.forward_kinematics(q, q.new_tensor(1.))
        return self.T

    def forward_kinematics_magnets(self) -> torch.Tensor:
        q_batched = self.q.unsqueeze(dim=0).repeat((self.num_magnets, 1))
        d_m = self.d_m.to(dtype=self.q.dtype, device=self.q.device)

        # perform forward kinematics for all magnets
        T = self.kinematic_class.forward_kinematics_batched(q=q_batched, v=(d_m / self.L0))

        return T

    def forward_kinematics_sensors(self) -> torch.Tensor:
        q_batched = self.q.unsqueeze(dim=0).repeat((self.num_sensors, 1))
        d_s_a = self.d_s_a.to(dtype=self.q.dtype, device=self.q.device)

        # perform forward kinematics for all sensors
        T = self.kinematic_class.forward_kinematics_batched(q=q_batched, v=(d_s_a / self.L0))

        # apply static transformations
        self.T_sensors_static = self.T_sensors_static.to(device=self.q.device, dtype=self.q.dtype)
        T = torch.matmul(T, self.T_sensors_static)

        return T

    def get_translations(self, v: torch.Tensor):
        # s is a torch tensor of dimension n with normalized positions along backbone in interval [0, 1]
        q_batched = self.q.unsqueeze(dim=0).repeat((v.size(0), 1))
        T = self.kinematic_class.forward_kinematics_batched(q_batched, v)
        return T[:, :3, 3]
