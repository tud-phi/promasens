import matplotlib.colors as colors
import numpy as np
import torch

from promasens.constants.constants import SIMULATION_ROBOT_PARAMS
from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.modules.kinematics import SensorMagnetKinematics
from promasens.motion_planning import TrajectoryType
from promasens.motion_planning.cc_trajectory import CcTrajectory
from promasens.simulation.fem.ngsolve_simulator import NGSolveSimulator
from promasens.visualization import plt


robot_kinematics = RobotKinematics(**SIMULATION_ROBOT_PARAMS)
sensor_magnet_kinematics = SensorMagnetKinematics(robot_kinematics)
simulator = NGSolveSimulator(sensor_magnet_kinematics, add_earth_magnetic_field=False,
                             **SIMULATION_ROBOT_PARAMS)

num_segments = SIMULATION_ROBOT_PARAMS['num_segments']

segment_trajectory = CcTrajectory(TrajectoryType.RANDOM, d=SIMULATION_ROBOT_PARAMS['d'])
q_traj_list = []
for i in range(num_segments):
    q_traj_i = segment_trajectory.plan()
    q_traj_list.append(q_traj_i)
q_traj = np.stack(q_traj_list, axis=1)

q = q_traj[0, ...]
# q = torch.zeros(num_segments, 3)
print("q: ", q)

simulator.apply_configuration(q)
u = simulator.get_sensor_measurements()

print("sensor measurements: ", u)

simulator.draw()
