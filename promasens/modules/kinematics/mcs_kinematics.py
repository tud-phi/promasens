from multiprocessing.sharedctypes import Value
import pandas as pd
import torch
from torch import nn
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRot
from typing import *

from .base_kinematics import BaseKinematics
from .cc_kinematics import CcKinematics

dtype = torch.float32


class McsKinematics(BaseKinematics):
    """
    Used to derive the sensor magnet kinematics between magnet k and sensor j located within segment i
    """
    def __init__(self, L0: float, d: float, l_r: float, **kwargs):
        super(McsKinematics, self).__init__()

        self.L0 = L0
        self.d = d
        self.l_r = l_r

        self.x_offset = 0
        self.y_offset = 0
        self.z_offset = 2.7E-3

        self.correct_static_tip_offset = True
            
        self.cc_kinematics = CcKinematics(L0=self.L0, d=self.d)

    def forward(self, df: pd.DataFrame):
        # we need to manually switch coordinate axis between the MoCap frame and the base frame
        # y-axis in MoCap frame becomes z-axis in base frame
        # z-axis in MoCap frame becomes negative y-axis in base frame

        quat_r_0 = torch.tensor(df[['W_r']].values, dtype=dtype)
        quat_r_1 = torch.tensor(df[['X_r']].values, dtype=dtype)
        quat_r_2 = -torch.tensor(df[['Z_r']].values, dtype=dtype)
        quat_r_3 = torch.tensor(df[['Y_r']].values, dtype=dtype)

        quat_b_0 = torch.tensor(df[['W_b']].values, dtype=dtype)
        quat_b_1 = torch.tensor(df[['X_b']].values, dtype=dtype)
        quat_b_2 = -torch.tensor(df[['Z_b']].values, dtype=dtype)
        quat_b_3 = torch.tensor(df[['Y_b']].values, dtype=dtype)

        x_r = torch.tensor(df[['TX_r']].values, dtype=dtype)
        y_r = -torch.tensor(df[['TZ_r']].values, dtype=dtype)
        z_r = torch.tensor(df[['TY_r']].values, dtype=dtype)

        x_b = torch.tensor(df[['TX_b']].values, dtype=dtype)
        y_b = -torch.tensor(df[['TZ_b']].values, dtype=dtype)
        z_b = torch.tensor(df[['TY_b']].values, dtype=dtype)

        q_list = []  # Redefining MCS base coordinate system

        q_tensor = None
        static_tip_offset = None
        for i in range(len(x_r)):
            t_b_0_hat = torch.tensor([[x_b[0]], [y_b[0]], [z_b[0]]], dtype=dtype)
            t_b_0 = t_b_0_hat - torch.tensor([[(self.x_offset)], [self.y_offset], [self.z_offset]],
                                             dtype=dtype)  # Translation of MCS frame to robot base frame

            quat_xyzw = np.array([quat_r_1[i].item(), quat_r_2[i].item(), quat_r_3[i].item(), quat_r_0[i].item()])
            R_0_r = torch.tensor(ScipyRot.from_quat(quat_xyzw).as_matrix(), dtype=dtype)

            # somehow the MoCap records the inverse orientation of each rigid-body
            R_0_r = R_0_r.transpose(0, 1)

            T_b_0 = torch.eye(4, dtype=dtype)
            T_b_0[0:3, 3] = t_b_0.squeeze()

            x_b_r = torch.tensor([[x_r[i]], [y_r[i]], [z_r[i]], [1]], dtype=dtype)
            x_0_r = torch.matmul(torch.linalg.inv(T_b_0), x_b_r)

            # as the MoCap position is sometimes slightly off, we need to correct it with the initial
            # (we assume straight) tip position in the base frame
            if self.correct_static_tip_offset is True and static_tip_offset is None:
                static_tip_offset = torch.clone(x_0_r)

                # we do not want to correct the vertical tip position, as it is not supposed to be zero
                static_tip_offset[2] = 0.
                static_tip_offset[3] = 0.

            if static_tip_offset is not None:
                x_0_r = x_0_r - static_tip_offset

            q = self.cc_kinematics.inverse_kinematics(x_0_r, R_0_r, self.l_r)

            q_list.append(q)
            q_tensor = torch.hstack(q_list)

        q_tensor = torch.squeeze(q_tensor, 0)

        return q_tensor
