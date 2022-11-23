import numpy as np
import magpylib as magpy
import torch

from promasens.constants.constants import SIMULATION_ROBOT_PARAMS
from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.modules.kinematics import SensorMagnetKinematics
from promasens.motion_planning import TrajectoryType
from promasens.motion_planning.cc_trajectory import CcTrajectory
from promasens.simulation.analytical.magpy_simulator import MagpyRobotSimulator
from promasens.visualization import plt
from promasens.visualization.pyvista_scene_visualizer import PyvistaSceneVisualizer
from promasens.visualization.streamplot_2D import streamplot_2D


if __name__ == "__main__":
    robot_kinematics = RobotKinematics(**SIMULATION_ROBOT_PARAMS)
    sensor_magnet_kinematics = SensorMagnetKinematics(robot_kinematics)
    simulator = MagpyRobotSimulator(sensor_magnet_kinematics, add_earth_magnetic_field=False,
                                    **SIMULATION_ROBOT_PARAMS)

    num_segments = SIMULATION_ROBOT_PARAMS['num_segments']

    segment_trajectory = CcTrajectory(TrajectoryType.RANDOM, d=SIMULATION_ROBOT_PARAMS['d'])
    q_traj_list = []
    for i in range(num_segments):
        q_traj_i = segment_trajectory.plan()
        q_traj_list.append(q_traj_i)
    q_traj = torch.tensor(np.stack(q_traj_list, axis=1))

    q = q_traj[0, :]
    # q = torch.zeros(num_segments, 3)
    Delta_c_max = segment_trajectory.Delta_c_max
    # q = torch.tensor([[Delta_c_max/2, 0., 0.] for i in range(num_segments)])
    # q = torch.tensor([[0., Delta_c_max, 0.], [0., -35/45*Delta_c_max, 0.], [-Delta_c_max, 0., 0.]])
    print("q: ", q)

    simulator.apply_configuration(q)
    T_sensors = robot_kinematics.T_sensors.detach().cpu().numpy()
    u = simulator.get_sensor_measurements()

    for child in simulator.col.children:
        print(child, "position:", child.position, "orientation:", child.orientation.as_quat())

    print("sensor measurements: ", u)

    # visualize scene with objects
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(projection='3d')
    simulator.show(canvas=ax)
    plt.tight_layout()
    plt.show()

    # create a grid
    L0 = SIMULATION_ROBOT_PARAMS["L0"]
    n_b = SIMULATION_ROBOT_PARAMS["num_segments"]
    xlim = (-L0*n_b * 10 ** 3, L0*n_b * 10 ** 3)
    zlim = (-0.1*(L0 * n_b) * 10 ** 3, 1.1*(L0 * n_b) * 10 ** 3)
    ts_x = np.linspace(xlim[0], xlim[1], 100)
    ts_z = np.linspace(zlim[0], zlim[1], 100)
    grid = np.array([[(x, 0, z) for x in ts_x] for z in ts_z])

    # compute field on grid
    # B_func = lambda grid: simulator.getB(observers=grid)

    # generate streamplot
    streamplot_2D(grid=grid, B=simulator.getB(grid), T_sensors=T_sensors, L0=L0)

    # 3D visualization with pyvista
    visualizer = PyvistaSceneVisualizer(robot_kinematics=robot_kinematics, B_func=simulator.getB,
                                        **SIMULATION_ROBOT_PARAMS, show_silicone=True, enable_shadows=False)
    visualizer.run(q_gt=q, filepath="scene_snapshots/analytical_simulation_three_segment.pdf")
