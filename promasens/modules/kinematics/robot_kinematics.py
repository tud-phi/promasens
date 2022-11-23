import torch
from torch import nn
from typing import *

from .segment_kinematics import SegmentKinematics


class RobotKinematics(nn.Module):
    def __init__(self, num_segments: int, L0: float, d: float, d_m: float,
                 num_sensors_per_segment: int, d_s_r: float, d_s_a: float,
                 phi_e: float = None, n_e: torch.Tensor = None, **kwargs):
        super(RobotKinematics, self).__init__()
        self.num_segments = num_segments
        self.num_sensors_per_segment = num_sensors_per_segment
        self.num_sensors = num_segments * num_sensors_per_segment

        self.segment_kinematics_list = []
        self.L0 = 0.
        self.num_magnets = 0
        for i in range(num_segments):
            segment_kinematics = SegmentKinematics(L0=L0, d=d, d_m=d_m, num_sensors=num_sensors_per_segment,
                                                   d_s_r=d_s_r, d_s_a=d_s_a, **kwargs)
            self.segment_kinematics_list.append(segment_kinematics)
            self.L0 += L0
            self.num_magnets += segment_kinematics.num_magnets

        # not used immediately for robot kinematics, but rather for magnet sensor kinematics (e.g. lambda)
        self.phi_e, self.n_e = phi_e, n_e

        self.T_base, self.T_tip, self.T_sensors, self.T_magnets = None, None, None, None
        self.segment_id_per_sensor = None  # segment id for each sensor
        q_straight = torch.zeros((num_segments, 3))
        self.forward(q_straight)

    def set_robot_params(self, *args, **kwargs):
        self.num_magnets = 0
        for segment_kinematics in self.segment_kinematics_list:
            if "num_sensors_per_segment" in kwargs:
                kwargs["num_sensors"] = kwargs["num_sensors_per_segment"]
            segment_kinematics.set_segment_params(*args, **kwargs)
            self.num_magnets += segment_kinematics.num_magnets

    def forward(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.apply_configuration(q)
        self.forward_kinematics_sensors()
        self.forward_kinematics_magnets()
        return self.T_base, self.T_tip, self.T_magnets, self.T_sensors

    def apply_configuration(self, q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param q: configuration of the robot with the shape n_b x n_q
        :return: the transformation from the inertial frame to the base frame of each segment with the shape n_b x 4 x 4
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)
        assert q.size(0) == self.num_segments
        self.q = q

        T_base = torch.zeros((self.num_segments, 4, 4), device=q.device, dtype=q.dtype)
        T_tip = T_base.clone()

        T_rolling = torch.eye(4, device=q.device, dtype=q.dtype)
        for i in range(self.num_segments):
            T_i = self.segment_kinematics_list[i].apply_configuration(q[i, ...])

            # T_i is still in the local frame of the base of segment
            T_base[i, ...] = T_rolling
            T_rolling = torch.matmul(T_rolling, T_i)
            T_tip[i, ...] = T_rolling

        self.T_base, self.T_tip = T_base, T_tip

        return T_base, T_tip

    def forward_kinematics_sensors(self) -> torch.Tensor:
        num_sensors = self.num_segments*self.num_sensors_per_segment
        T_sensors = torch.zeros((num_sensors, 4, 4), device=self.q.device, dtype=self.q.dtype)
        self.segment_id_per_sensor = torch.ones((num_sensors,), device=self.q.device, dtype=torch.int64)
        for i in range(1, self.num_segments + 1):
            T_sensors_segment = self.segment_kinematics_list[i-1].forward_kinematics_sensors()
            T_sensors_segment = torch.matmul(self.T_base[i-1, ...], T_sensors_segment)

            selector_low, selector_high = (i-1)*self.num_sensors_per_segment, i*self.num_sensors_per_segment
            T_sensors[selector_low:selector_high, ...] = T_sensors_segment
            self.segment_id_per_sensor[selector_low:selector_high] \
                = i*self.segment_id_per_sensor[selector_low:selector_high]

        self.T_sensors = T_sensors

        return T_sensors

    def forward_kinematics_magnets(self) -> torch.Tensor:
        T_magnets = torch.zeros((self.num_magnets, 4, 4), device=self.q.device, dtype=self.q.dtype)
        k_start = 0
        for i in range(self.num_segments):
            T_magnets_i = self.segment_kinematics_list[i].forward_kinematics_magnets()
            k_stop = k_start + T_magnets_i.size(0)

            T_magnets[k_start:, ...] = torch.matmul(self.T_base[i, ...], T_magnets_i)
            k_start = k_stop

        self.T_magnets = T_magnets

        return T_magnets

    def get_translations(self, s: torch.Tensor):
        # s is a torch tensor of dimension n with normalized positions along backbone in interval [0, n_b]
        t = s.new_zeros((s.size(0), 3, 1))
        for i in range(1, self.num_segments + 1):
            selector = (s.new_tensor(i-1) < s) & (s <= s.new_tensor(i))
            if i == 0:
                selector = selector | (s == 0.)
            if selector.sum() == 0:
                # nothing to do for this segment
                continue

            s_i = s[selector] - (i-1)
            t_i = self.segment_kinematics_list[i-1].get_translations(s_i)

            p_i = torch.cat([t_i, self.q.new_ones((t_i.size(0), 1))], dim=1).unsqueeze(dim=2)
            t[selector, ...] = torch.matmul(self.T_base[i-1, ...], p_i)[:, :3, :]

        return t
