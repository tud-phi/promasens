import magpylib as magpy
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from typing import *

from promasens.simulation.base_simulator import BaseSimulator


class MagpyRobotSimulator(BaseSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        magnet_dim = (self.magnet_inner_diameter*10**3, self.magnet_outer_diameter*10**3,
                      self.magnet_thickness*10**3, 0, 360)

        magnetization = tuple((self.magnetization * 10**3).tolist())  # conversion from tesla to millitesla

        self.col = magpy.Collection()
        for j in range(self.robot_kinematics.num_sensors):
            self.col.add(magpy.Sensor())
        for k in range(self.robot_kinematics.num_magnets):
            self.col.add(magpy.magnet.CylinderSegment(magnetization=magnetization, dimension=magnet_dim))

        self.apply_configuration(np.zeros((self.robot_kinematics.num_segments, 3)))

    def apply_configuration(self, q: Union[np.array, torch.tensor] = None):
        if q is not None:
            # q_array: i x q_i
            if type(q) is np.ndarray:
                q = torch.tensor(q)
            self.robot_kinematics.forward(q)

        # T_magnets: n_m x 4 x 4, T_sensors: n_s x 4 x 4
        T_magnets, T_sensors = self.robot_kinematics.T_magnets, self.robot_kinematics.T_sensors

        assert T_magnets.size(0) == len(self.col.sources)
        for k in range(T_magnets.size(0)):
            T_magnet = T_magnets[k].cpu().numpy()
            magnet_bottom_position = T_magnet @ np.array([[0], [0], [-self.magnet_thickness/2], [1]])
            self.col.sources[k].position = magnet_bottom_position[:3, 0] * 10 ** 3  # m to mm
            self.col.sources[k].orientation = Rotation.from_matrix(T_magnet[:3, :3])

        assert T_sensors.size(0) == len(self.col.sensors)
        for j in range(T_sensors.size(0)):
            T_sensor = T_sensors[j].cpu().numpy()
            self.col.sensors[j].position = T_sensor[:3, 3] * 10 ** 3  # m to mm
            self.col.sensors[j].orientation = Rotation.from_matrix(T_sensor[:3, :3])

    def follow_trajectory(self, q_array: np.array):
        # q_array: t x i x q_i
        raise NotImplementedError

    def getB(self, *args, **kwargs):
        B = magpy.getB(self.col, *args, **kwargs)
        if self.add_earth_magnetic_field:
            B_e = self.get_earth_magnetic_field() * 10 ** 3  # T to mT
            B += B_e
        return B

    def getH(self, *args, **kwargs):
        if self.add_earth_magnetic_field:
            raise NotImplementedError("MagpySimulator::getH() does not consider the earth magnetic field yet")
        return self.col.getH(*args, **kwargs)

    def show(self, *args, **kwargs):
        self.col.show(*args, **kwargs)

    def get_sensor_measurements(self) -> np.array:
        B_e = None
        if self.add_earth_magnetic_field:
            B_e = self.get_earth_magnetic_field_in_sensor_frames() * 10 ** 3  # T to mT

        u = np.zeros((len(self.col.sensors, )))
        for j in range(len(self.col.sensors)):
            B_j = self.col.sensors[j].getB(self.col)  # in millitesla

            if self.add_earth_magnetic_field:
                # add earth magnetic field
                B_j += B_e[j]

            u[j] = B_j[2]

        return u
