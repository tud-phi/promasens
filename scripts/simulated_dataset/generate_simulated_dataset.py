import numpy as np
import pandas as pd
import torch

# set default tensor type and device
torch.set_default_tensor_type('torch.FloatTensor')

from promasens.constants.constants import SIMULATION_ROBOT_PARAMS, SIMULATION_AFFINE_CURVATURE_SEGMENT_PARAMS
from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.modules.kinematics import SensorMagnetKinematics
from promasens.motion_planning import TrajectoryType
from promasens.motion_planning.ac_trajectory import AcTrajectory
from promasens.motion_planning.cc_trajectory import CcTrajectory
from promasens.simulation.analytical.magpy_simulator import MagpyRobotSimulator

nominal_robot_params = SIMULATION_ROBOT_PARAMS
# nominal_robot_params = SIMULATION_AFFINE_CURVATURE_SEGMENT_PARAMS

try:
    from promasens.simulation.fem.ngsolve_simulator import NGSolveSimulator
except ImportError as e:
    print(e)
    print("FEM simulation is not available. Please install ngsolve.")
    NGSolveSimulator = None

simulator = MagpyRobotSimulator
if simulator == MagpyRobotSimulator:
    sim_str = 'analytical'
elif simulator == NGSolveSimulator:
    sim_str = 'fem'
    if NGSolveSimulator is None:
        raise ImportError("FEM simulation is not available. Please install ngsolve.")
else:
    raise ValueError('Invalid simulator')

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

traj_type = TrajectoryType.FULL_LEMNISCATE
traj_num_steps = 400

# kinematic parametrization
kinematic_parametrization = nominal_robot_params.get("kinematic_parametrization", "cc")

add_earth_magnetic_field = False
random_earth_magnetic_field_direction = False
if random_earth_magnetic_field_direction:
    assert add_earth_magnetic_field is True
n_e = nominal_robot_params.get('n_e', np.array([1.0, 0.0, 0.0]))

random_sensor_params = {"phi_off": True, "psi_s": False, "d_s_r": False, "d_s_a": False}


def save(db_name: str, data: np.array, last_save: bool = True):
    headers = ["time_idx", "segment_id", "sensor_id"]
    for i in range(num_segments):
        if kinematic_parametrization == "cc":
            headers.extend([f"q_dx_{i}", f"q_dy_{i}", f"q_dL_{i}"])
        elif kinematic_parametrization == "ac":
            headers.extend([f"q_kappa0_{i}", f"q_kappa1_{i}", f"q_phi_{i}", f"q_dL_{i}"])
        else:
            raise ValueError("Invalid kinematic parametrization")

    headers.append("lambda")
    for k in range(num_magnets):
        headers.extend([f"d_{k}", f"alpha_{k}", f"beta_{k}", f"theta_{k}"])
    headers.append("u")

    df = pd.DataFrame(data, columns=headers)
    df['time_idx'], df['segment_id'], df['sensor_id'] = \
        df['time_idx'].astype(int), df['segment_id'].astype(int), df['sensor_id'].astype(int)

    if last_save:
        print(df)

    db_path = f"{sim_str}_databases/{db_name}.csv"
    print(f"Saving dataset to {db_path}")
    df.to_csv(db_path, index=False)


if __name__ == "__main__":
    robot_kinematics = RobotKinematics(**nominal_robot_params)
    sensor_magnet_kinematics = SensorMagnetKinematics(robot_kinematics)
    simulator = simulator(sensor_magnet_kinematics, add_earth_magnetic_field=add_earth_magnetic_field,
                          **nominal_robot_params)

    num_segments = nominal_robot_params['num_segments']
    num_sensors, num_magnets = robot_kinematics.num_sensors, robot_kinematics.num_magnets

    if add_earth_magnetic_field:
        if random_earth_magnetic_field_direction:
            emf_str = "_random_emf"
        else:
            # emf_str = f"_emf_({format(n_e[0], '.3f')},{format(n_e[1], '.3f')},{format(n_e[2], '.3f')})"
            emf_str = f"_emf_({n_e[0]},{n_e[1]},{n_e[2]})"
    else:
        emf_str = ""
    rand_phi_off_str = "_rand_phi_off" if random_sensor_params.get("phi_off", False) else ""
    rand_psi_s_str = "_rand_psi_s" if random_sensor_params.get("psi_s", False) else ""
    rand_d_s_r_str = "_rand_d_s_r" if random_sensor_params.get("d_s_r", False) else ""
    rand_d_s_a_str = "_rand_d_s_a" if random_sensor_params.get("d_s_a", False) else ""
    db_name = f"{sim_str}_db_{kinematic_parametrization}_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_" \
              f"T{traj_type.value}_n_t-{traj_num_steps}{emf_str}" \
              f"{rand_phi_off_str}{rand_psi_s_str}{rand_d_s_r_str}{rand_d_s_a_str}"

    if kinematic_parametrization == "cc":
        segment_trajectory = CcTrajectory(traj_type, num_steps=traj_num_steps, d=nominal_robot_params['d'],
                                          seed=seed)
    elif kinematic_parametrization == "ac":
        segment_trajectory = AcTrajectory(traj_type, num_steps=traj_num_steps, seed=seed)
    else:
        raise ValueError("Invalid kinematic parametrization")

    q_traj_list = []
    for i in range(num_segments):
        q_traj_i = segment_trajectory.plan(phase=2 * np.pi * (i / num_segments))
        q_traj_list.append(q_traj_i)
    q_traj = np.stack(q_traj_list, axis=1)

    num_timesteps = q_traj.shape[0]

    data = None
    robot_params = nominal_robot_params.copy()
    for time_idx in range(num_timesteps):
        q = q_traj[time_idx]

        print(f"time_idx {time_idx} / {num_timesteps}, q = {q}")

        if random_sensor_params.get("phi_off", False):
            phi_off = np.random.uniform(0, 2 * np.pi / robot_kinematics.num_sensors_per_segment)
            robot_params.update({"phi_off": phi_off})
        if random_sensor_params.get("psi_s", False):
            psi_s = np.random.uniform(-20 / 180 * np.pi, 20 / 180 * np.pi)
            robot_params.update({"psi_s": psi_s})
        if random_sensor_params.get("d_s_r", False):
            nominal_d_s_r = nominal_robot_params["d_s_r"]
            d_s_r = np.random.uniform(2 / 3 * nominal_d_s_r, 4 / 3 * nominal_d_s_r)
            robot_params.update({"d_s_r": d_s_r})
        if random_sensor_params.get("d_s_a", False):
            nominal_d_s_a = nominal_robot_params["d_s_a"]
            abs_d_sm = np.abs(nominal_d_s_a - nominal_robot_params["d_m"])
            d_s_a = np.random.uniform(nominal_d_s_a - abs_d_sm / 4, nominal_d_s_a + abs_d_sm / 4)
            robot_params.update({"d_s_a": d_s_a})
        if add_earth_magnetic_field and random_earth_magnetic_field_direction:
            n_e = -np.ones(3) + 2 * np.random.rand(3)
            n_e /= np.linalg.norm(n_e)
            robot_params.update({"n_e": n_e})
            simulator.set_earth_magnetic_field_dir(n_e=n_e)
        robot_kinematics.set_robot_params(**robot_params)

        robot_kinematics.forward(torch.tensor(q))

        xi = sensor_magnet_kinematics.forward().detach().cpu().numpy()

        # try:
        #     simulator.apply_configuration()
        # except Exception as e:
        #     print(f"Error applying configuration q={q}:")
        #     print(e)
        #     continue

        simulator.apply_configuration()

        u = simulator.get_sensor_measurements()

        print("u", u)

        assert xi.shape[0] == u.shape[0]

        for sensor_id in range(num_sensors):
            segment_id = robot_kinematics.segment_id_per_sensor[sensor_id].item()
            row = np.array([time_idx, segment_id, sensor_id])
            row = np.concatenate([row, q.flatten(), xi[sensor_id], u[sensor_id:sensor_id + 1]])

            if data is None:
                data = np.zeros((num_timesteps * num_sensors, row.shape[0]))

            data[time_idx * num_sensors + sensor_id, :] = row

        if time_idx % 500 == 0:
            save(db_name, data[:(time_idx * num_sensors)], False)

    save(db_name, data, True)
