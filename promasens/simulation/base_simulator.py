import numpy as np
import torch

from promasens.modules.kinematics.sensor_magnet_kinematics import SensorMagnetKinematics


class BaseSimulator:
    def __init__(self, sensor_magnet_kinematics: SensorMagnetKinematics,
                 D_m_in: float, D_m_out: float, t_m: float,
                 add_earth_magnetic_field: bool = True, phi_e: float = 0., n_e: np.array = None,
                 **kwargs):
        self.magnet_thickness = t_m
        self.magnet_inner_diameter = D_m_in
        self.magnet_outer_diameter = D_m_out

        # Our magnet (N50): https://www.magnetenspecialist.nl/en/store/ring-magnets/ringmagnet-12-x-6-x-6-mm/
        # https://www.supermagnete.de/eng/physical-magnet-data
        # Neodymium grade N50 has approx. 1.42-1.47 Tesla
        self.magnetization = np.array([0, 0, 1.45])

        self.sensor_magnet_kinematics = sensor_magnet_kinematics
        self.robot_kinematics = sensor_magnet_kinematics.robot_kinematics

        self.add_earth_magnetic_field = add_earth_magnetic_field
        self.n_e = n_e
        self.set_earth_magnetic_field_dir(phi_e, n_e)

    def set_earth_magnetic_field_dir(self, phi_e: float = 0., n_e: np.array = None):
        if n_e is None:
            n_e = np.stack([np.cos(phi_e), np.sin(phi_e), 0.], axis=0)

        self.n_e, n_e_tensor = n_e, torch.tensor(n_e)
        self.robot_kinematics.n_e, self.sensor_magnet_kinematics.n_e = n_e_tensor, n_e_tensor

    def get_earth_magnetic_field(self) -> np.array:
        # earth magnetic field
        # varies between 25 and 65 microtesla on the earth surface
        # https://en.wikipedia.org/wiki/Earth%27s_magnetic_field
        B_e_abs = 65 * 10 ** (-6)  # magnitude of earth magnetic field in Tesla
        B_e_inertial = B_e_abs * self.n_e  # earth magnetic field in the inertial frame

        return B_e_inertial

    def get_earth_magnetic_field_in_sensor_frames(self) -> np.array:
        B_e_inertial = self.get_earth_magnetic_field()

        T_sensors = self.robot_kinematics.T_sensors

        B_e = np.zeros((T_sensors.size(0), B_e_inertial.shape[0]))
        for j in range(T_sensors.size(0)):
            R = T_sensors[j, :3, :3].cpu().numpy()
            B_e[j] = R.T @ B_e_inertial

        return B_e
