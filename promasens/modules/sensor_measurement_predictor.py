import torch
from torch import nn
from typing import *

from promasens.enums.joint_nn_mode import JointNNMode
from .kinematics.robot_kinematics import RobotKinematics
from .kinematics.sensor_magnet_kinematics import SensorMagnetKinematics


class SensorMeasurementPredictor(nn.Module):
    def __init__(self, kinematic_params: dict, nn: Union[torch.nn.Module, List[torch.nn.Module]] = None,
                 joint_nn_mode: JointNNMode = JointNNMode.ALL):
        super().__init__()

        self.robot_kinematics = RobotKinematics(**kinematic_params)
        self.sensor_magnet_kinematics = SensorMagnetKinematics(robot_kinematics=self.robot_kinematics)

        self.num_sensors = self.robot_kinematics.num_sensors
        self.nn, self.joint_nn_mode = nn, joint_nn_mode

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        xi = self.sensor_magnet_kinematics(q)

        if self.joint_nn_mode == JointNNMode.ALL:
            # a single neural network for all sensors
            u = self.nn(xi)
        elif self.joint_nn_mode == JointNNMode.EACH_SENSOR:
            # a separate neural network for each sensor
            u_j_list = []
            for sensor_id, nn_model in enumerate(self.nn):
                xi_j = xi[sensor_id, ...]
                u_j = nn_model(xi_j.unsqueeze(dim=0)).squeeze(0)
                u_j_list.append(u_j)

            u = torch.stack(u_j_list)
        elif self.joint_nn_mode == JointNNMode.EACH_SEGMENT:
            # a separate neural network for each segment
            u = q.new_zeros((self.num_sensors, ))
            for segment_id in range(1, len(self.nn) + 1):
                nn_model = self.nn[segment_id - 1]
                u_selector = self.robot_kinematics.segment_id_per_sensor == segment_id
                xi_i = xi[u_selector, ...]
                u[u_selector] = nn_model(xi_i)
        else:
            raise NotImplementedError

        return u

    def load_torch_nn(self, nn_statedict_paths: Union[str, List[str]], *args, **kwargs):
        if type(nn_statedict_paths) == str:
            self.joint_nn_mode = JointNNMode.ALL
            self.nn = self.load_single_torch_nn(nn_statedict_path=nn_statedict_paths, *args, **kwargs)
        elif type(nn_statedict_paths) == list:
            self.joint_nn_mode = JointNNMode.EACH_SENSOR
            self.nn = []
            for path in nn_statedict_paths:
                self.nn.append(self.load_single_torch_nn(nn_statedict_path=path, *args, **kwargs))
        else:
            raise ValueError('nn_statedict_paths must be a str or a list of str')

    def load_single_torch_nn(self, nn_class, nn_statedict_path: str, nn_params: dict,
                             use_swa: bool = True) -> torch.nn.Module:
        nn_model = nn_class(**nn_params)
        if use_swa:
            nn_model = torch.optim.swa_utils.AveragedModel(nn_model)

        nn_model.load_state_dict(torch.load(nn_statedict_path))
        nn_model.eval()

        return nn_model
