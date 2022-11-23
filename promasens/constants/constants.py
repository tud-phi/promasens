import numpy as np
import torch


SIMULATION_ROBOT_PARAMS = {
    "num_segments": 3,
    "kinematic_parametrization": "cc",
    "L0": 110E-3,
    "d": 22E-3,
    "d_m": 55E-3,
    "num_sensors_per_segment": 3,
    "d_s_r": 13E-3,
    "d_s_a": 110E-3,
    "n_e": np.array([1., 0., 0.]),
    "phi_off": 0./180*np.pi,
    "psi_s": 0./180*np.pi,
    "D_m_in": 0.006,  # magnet inner diameter [m]
    "D_m_out": 0.012,  # magnet outer diameter [m]
    "t_m": 0.006,  # magnet thickness [m]
}

SIMULATION_PERTURBED_ROBOT_PARAMS = {
    "num_segments": 3,
    "kinematic_parametrization": "cc",
    "L0": 110E-3,
    "d": 22E-3,
    "d_m": 55E-3,
    "num_sensors_per_segment": 3,
    "d_s_r": 16E-3,
    "d_s_a": 110E-3,
    "n_e": np.array([1., 0., 0.]),
    "phi_off": 60./180*np.pi,
    "psi_s": -10./180*np.pi,
}

SIMULATION_AFFINE_CURVATURE_SEGMENT_PARAMS = {
    "num_segments": 1,
    "kinematic_parametrization": "ac",
    "L0": 200E-3,
    "d": 22E-3,
    "d_m": np.array([50E-3, 150E-3]),
    "num_sensors_per_segment": 9,
    "d_s_r": 13E-3,
    "d_s_a": np.array([0E-3, 100E-3, 200E-3, 0E-3, 100E-3, 200E-3, 0E-3, 100E-3, 200E-3]),
    "phi_off": 0./180*np.pi,
    "psi_s": 0./180*np.pi,
    "D_m_in": 0.006,  # magnet inner diameter [m]
    "D_m_out": 0.012,  # magnet outer diameter [m]
    "t_m": 0.006,  # magnet thickness [m]
}

# earth magnetic field vector in North, East, Down (NED) frame
# in Delft, Netherlands in April 2022
# 51° 59' 56" N, 4° 22' 51" E, 0.0 m
# https://www.ngdc.noaa.gov/geomag/calculators/magcalc.shtml#igrfwmm
B_e = torch.tensor([19152.8, 645.8, 45408.6])  # nT
# as we measure the angle of the magnetic north with phi_e,
# we are only interested in the ratio between horizontal and vertical components
B_e_mag = torch.linalg.vector_norm(B_e)
B_e_h, B_e_v = torch.linalg.vector_norm(B_e[:2]), B_e[2]
# planar angle of magnetic north in base frame
phi_e = -143 / 180 * np.pi
# unit vector of earth magnetic field in the base frame
n_e = torch.tensor([np.cos(phi_e) * B_e_h / B_e_mag, np.sin(phi_e) * B_e_h / B_e_mag, B_e_v / B_e_mag])

EXPERIMENT_ROBOT_PARAMS_BASE = {
    "num_segments": 1,
    "L0": 110E-3,
    "d": 22E-3,
    "d_m": 55E-3,
    "num_sensors_per_segment": 3,
    "d_s_r": 13E-3,
    "d_s_a": 110E-3,
    "phi_e": phi_e,
    "phi_off": 0.,
    "D_m_in": 0.006,  # magnet inner diameter [m]
    "D_m_out": 0.012,  # magnet outer diameter [m]
    "t_m": 0.006,  # magnet thickness [m]
}

EXPERIMENT_ROBOT_PARAMS_20211207 = EXPERIMENT_ROBOT_PARAMS_BASE.copy()
EXPERIMENT_ROBOT_PARAMS_20211207.update({
    "l_r": 0.975 * EXPERIMENT_ROBOT_PARAMS_BASE["L0"]
})

EXPERIMENT_ROBOT_PARAMS_20220211 = EXPERIMENT_ROBOT_PARAMS_BASE.copy()
L0 = 105.8E-3
EXPERIMENT_ROBOT_PARAMS_20220211.update({
    "L0": L0,
    "d_s_a": L0,
    "d_m": L0-55E-3,
    "l_r": 99E-3,
})

EXPERIMENT_ROBOT_PARAMS_20220221 = EXPERIMENT_ROBOT_PARAMS_BASE.copy()
L0 = 106.85E-3
EXPERIMENT_ROBOT_PARAMS_20220221.update({
    "L0": L0,
    "d_s_a": L0,
    "d_m": L0-55E-3,
    "l_r": 93.6E-3,
})

EXPERIMENT_ROBOT_PARAMS_20220421 = EXPERIMENT_ROBOT_PARAMS_BASE.copy()
L0 = 105.5E-3
EXPERIMENT_ROBOT_PARAMS_20220421.update({
    "L0": L0,
    "d_s_a": L0,
    "d_m": L0-60E-3,
    "l_r": L0-6E-3,
})

EXPERIMENT_ROBOT_PARAMS_20220502 = EXPERIMENT_ROBOT_PARAMS_BASE.copy()
L0 = 106.05E-3
EXPERIMENT_ROBOT_PARAMS_20220502.update({
    "L0": L0,
    "d_s_a": L0,
    "d_m": L0-60E-3,
    "l_r": L0-6E-3,
    "n_e": n_e,
})
