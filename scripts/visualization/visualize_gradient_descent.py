import numpy as np
import pandas as pd
import progressbar
import torch
from typing import *

from promasens.enums.joint_nn_mode import JointNNMode
from promasens.modules.proprioceptor import Proprioceptor
from promasens.utils.df_to_tensor_utils import database_df_to_tensors
from promasens.utils.load_predictor import load_predictor
from promasens.visualization import plt
from promasens.visualization.loss_landscape_plotter import LossLandscapePlotter
from promasens.visualization.pyvista_scene_visualizer import PyvistaSceneVisualizer

dataset_type = "real-world"
if dataset_type == "analytical":
    from promasens.constants.constants import SIMULATION_ROBOT_PARAMS as kinematic_params

    num_segments = kinematic_params["num_segments"]
    num_sensors = num_segments * kinematic_params["num_sensors_per_segment"]
    num_magnets = num_segments

    joint_nn_mode = JointNNMode.EACH_SEGMENT

    train_dataset_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_" \
                         f"rand_phi_off_rand_psi_s_rand_d_s_r"
    test_dataset_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T3_n_t-400"
    test_dataset_path = f"datasets/analytical_simulation/{test_dataset_name}.csv"

    # gradient descent settings
    max_num_iterations = 20
    if num_segments == 1:
        gamma = 3.5E-4  # gradient descent learning rate
    elif num_segments == 2:
        gamma = 3E-3
        gamma = gamma * torch.ones((num_segments, 3))
        gamma[:, 2] = 0.1 * gamma[:, 2]  # discount gradient for delta L
    elif num_segments == 3:
        gamma = 2E-3
        gamma = gamma * torch.ones((num_segments, 3))
        gamma[:, 2] = 0.1 * gamma[:, 2]  # discount gradient for delta L
    else:
        raise NotImplementedError
    mu = 0.3  # gradient descent momentum
else:
    from promasens.constants.constants import EXPERIMENT_ROBOT_PARAMS_20220502 as kinematic_params

    joint_nn_mode = JointNNMode.EACH_SENSOR

    phi_off_deg = 0
    phi_off = phi_off_deg / 180. * np.pi
    # dataset_name = f"2022-02-2{dataset_name}1_spiral2_480s_start_200mb_max_425mb_f0_20"
    train_dataset_name = f"2022-05-02_FLOWER_SLOW_NOMINAL_P{int(phi_off_deg)}_R1"
    test_dataset_name = f"2022-05-02_FLOWER_SLOW_NOMINAL_P{int(phi_off_deg)}_R1"
    test_dataset_path = f"merged_databases/{test_dataset_name}_test.csv"

    # gradient descent settings
    max_num_iterations = 20
    gamma = 1.5E-8  # gradient descent learning rate
    mu = 0.2  # gradient descent momentum
    if train_dataset_name == "2022-05-02_RAND_NOMINAL_600s_P0_R1":
        gamma = 1.E-8  # gradient descent learning rate

# init random seed
seed = 0
np.random.seed(seed)
sample_rate = 40

# dataset that is used
df = pd.read_csv(test_dataset_path).dropna(axis=0)
num_sensors = len(df["sensor_id"].unique())
q_gt_ts, xi_gt_ts, u_gt_ts = database_df_to_tensors(df)
q_min, q_max = q_gt_ts.min(dim=0)[0], q_gt_ts.max(dim=0)[0]

q_optim_bool = torch.zeros((q_gt_ts.size(1), q_gt_ts.size(2)), dtype=torch.bool)
q_optim_bool[0, 0], q_optim_bool[0, 1] = True, True  # Delta_x, Delta_y optimization of first segment

predictor = load_predictor(train_dataset_name, kinematic_params, num_segments=1, num_sensors=num_sensors,
                           seed=seed, joint_nn_mode=JointNNMode.EACH_SENSOR)
proprioceptor = Proprioceptor(predictor=predictor, q_optim_bool=q_optim_bool, q_min=q_min, q_max=q_max,
                              max_num_iterations=max_num_iterations, gamma=gamma, mu=mu)
proprioceptor.set_time_optim_params(sample_rate, global_optim_freq=0., global_optim_delay=0.)

plotter = LossLandscapePlotter(q_optim_bool=q_optim_bool, rmse_u_limit=100)

time_idx = 3 * 40
frame_rate = 2  # number of gradient descent steps per second of movie time
q_hat_init = torch.zeros(size=q_optim_bool.size(), dtype=q_gt_ts.dtype)
if __name__ == "__main__":
    t = time_idx / sample_rate

    # get data for one time step
    q_gt, xi_gt, u_gt = database_df_to_tensors(df, time_idx)

    q_hat_init[~q_optim_bool] = q_gt[~q_optim_bool]

    q_hat_global, u_hat_global = proprioceptor.run_global_optimization(q_hat_init=q_hat_init, u_gt=u_gt, num_samples=50)

    q_hat, u_hat = proprioceptor.forward(q_hat_init, u_gt, optimize_globally=False, q_gt=q_gt)
    q_hat_its, u_hat_its = proprioceptor.q_hat_its, proprioceptor.u_hat_its

    # loss landscape plot
    loss_landscape_plotter = LossLandscapePlotter(q_optim_bool=q_optim_bool, rmse_u_limit=100)
    filepath_landscape = f"videos/gradient_descent_{test_dataset_name}_t={'{:.2f}'.format(t)}_loss_landscape.mp4"
    loss_landscape_plotter.setup_movie(filepath=filepath_landscape, frame_rate=frame_rate)

    # 3D visualization with pyvista
    pyvista_visualizer = PyvistaSceneVisualizer(robot_kinematics=predictor.robot_kinematics,
                                                **kinematic_params, show_silicone=True, enable_shadows=False)
    filepath_pyvista = f"videos/gradient_descent_{test_dataset_name}_t={'{:.2f}'.format(t)}_pyvista.mp4"
    pyvista_visualizer.setup_movie(filepath=filepath_pyvista, sample_rate=frame_rate, frame_rate=frame_rate)

    print("Rendering movie frames...")
    for it in progressbar.progressbar(range(q_hat_its.size(0))):
        pyvista_visualizer.run_timestep(q_gt=q_gt, q_hat=q_hat_its[it])
        loss_landscape_plotter.run_step(t=t,
                                        samples_q=proprioceptor.brute_grid.cpu().numpy(),
                                        samples_rmse_u=proprioceptor.brute_cost.cpu().numpy(),
                                        q_hat=q_hat_its[it].cpu().numpy(), u_hat=u_hat_its[it].cpu().numpy(),
                                        q_gt=q_gt.cpu().numpy(), u_gt=u_gt.cpu().numpy(),
                                        # q_hat_global=q_hat_global.cpu().numpy(),
                                        # u_hat_global=u_hat_global.cpu().numpy(),
                                        q_hat_its=q_hat_its[:(it+1)].cpu().numpy(),
                                        u_hat_its=u_hat_its[:(it+1)].cpu().numpy()
                                        )

    loss_landscape_plotter.finish_movie()
