import numpy as np
import pandas as pd
import torch

from promasens.enums.joint_nn_mode import JointNNMode
from promasens.modules.proprioceptor import Proprioceptor
from promasens.utils.df_to_tensor_utils import database_df_to_tensors, tensors_to_inference_df
from promasens.utils.load_predictor import load_predictor
from promasens.plotting.plot_configuration_estimates import (
    plot_cc_configuration_estimates, plot_ac_configuration_estimates
)
from promasens.plotting.plot_sensor_predictions import plot_sensor_predictions

# kinematic parameters of robot
from promasens.constants.constants import (
    SIMULATION_ROBOT_PARAMS
)

kinematic_params = SIMULATION_ROBOT_PARAMS
kinematic_parametrization = kinematic_params.get("kinematic_parametrization", "cc")
num_segments = kinematic_params["num_segments"]
num_magnets_per_segment = 1 if type(kinematic_params["d_m"]) == float else len(kinematic_params["d_m"])
num_magnets = num_segments * num_magnets_per_segment
num_sensors = num_segments * kinematic_params["num_sensors_per_segment"]

if kinematic_parametrization == "cc":
    train_dataset_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_" \
                         f"rand_phi_off_rand_psi_s_rand_d_s_r"
    test_dataset_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T3_n_t-400"
elif kinematic_parametrization == "ac":
    train_dataset_name = f"analytical_db_ac_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_" \
                         f"rand_phi_off"
    test_dataset_name = f"analytical_db_ac_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T5_n_t-400"
else:
    raise NotImplementedError(f"kinematic parametrization {kinematic_parametrization} not implemented")

test_dataset_path = f"{test_dataset_name}"
# dataset that is used
df = pd.read_csv(f'analytical_databases/{test_dataset_path}.csv').dropna(axis=0)
num_sensors = len(df["sensor_id"].unique())

# init random seed
seed = 0
np.random.seed(seed)

# device
device = torch.device("cpu")

# other settings
manual_superposition = False
sample_rate = 40

# global optimization settings
global_optim_freq = 0.  # Hz
global_optim_delay = 0.  # in seconds

q_gt_ts, xi_gt_ts, u_gt_ts = database_df_to_tensors(df, device=device)
q_min, q_max = q_gt_ts.min(dim=0)[0], q_gt_ts.max(dim=0)[0]

# gradient descent settings
q_optim_bool = torch.ones((q_gt_ts.size(1), q_gt_ts.size(2)), dtype=torch.bool)
max_num_iterations = 20
if kinematic_parametrization == "cc":
    joint_nn_mode = JointNNMode.EACH_SEGMENT
    if num_segments == 1:
        gamma = 3.5E-4  # gradient descent learning rate
    elif num_segments == 2:
        gamma = 3E-3
        gamma = gamma * torch.ones((num_segments, q_gt_ts.shape[-1]))
        gamma[:, -1] = 0.1 * gamma[:, -1]  # discount gradient for delta L
    elif num_segments == 3:
        gamma = 2E-3
        gamma = gamma * torch.ones((num_segments, q_gt_ts.shape[-1]))
        gamma[:, -1] = 0.1 * gamma[:, -1]  # discount gradient for delta L
    else:
        raise NotImplementedError
elif kinematic_parametrization == "ac":
    joint_nn_mode = JointNNMode.EACH_SENSOR
    gamma = torch.tensor([1E0, 5E0, 1E0, 2E-4])
    # q_optim_bool[:, 0] = False  # do not optimize theta0
    # q_optim_bool[:, 1] = False  # do not optimize theta1
    # q_optim_bool[:, -2] = False  # do not optimize phi
    # q_optim_bool[:, -1] = False  # do not optimize dL
else:
    raise NotImplementedError
mu = 0.3  # gradient descent momentum

predictor = load_predictor(train_dataset_name, kinematic_params, num_segments, num_sensors, seed,
                           joint_nn_mode=joint_nn_mode, manual_superposition=manual_superposition, device=device)
proprioceptor = Proprioceptor(predictor=predictor, q_optim_bool=q_optim_bool, q_min=q_min, q_max=q_max,
                              max_num_iterations=max_num_iterations, gamma=gamma, mu=mu, verbose=False)
proprioceptor.set_time_optim_params(sample_rate, global_optim_freq, global_optim_delay)

optim_variable_list = list(map(str, list(q_optim_bool.view(-1).cpu().numpy())))
inference_filename = f"{train_dataset_name}_to_{test_dataset_name}_" \
                     f"inference_seed_{seed}.csv"
# _optim_{'_'.join(optim_variable_list)}

if __name__ == "__main__":
    for time_idx in df["time_idx"].unique():
        print(f"Run inference at t={format(time_idx / sample_rate, '.3f')}s "
              f"of t_f={len(df['time_idx'].unique()) / sample_rate}s")
        # get data for one time step
        q_gt, xi_gt, u_gt = database_df_to_tensors(df, time_idx, device=device)

        q_hat, u_hat = proprioceptor.run_time_step(u_gt, q_gt)

    df = tensors_to_inference_df(proprioceptor.q_gt_ts, proprioceptor.q_hat_ts, proprioceptor.u_gt_ts,
                                 proprioceptor.u_hat_ts, proprioceptor.rmse_u_ts,
                                 kinematic_parametrization=kinematic_parametrization)
    print(df)
    df.to_csv(f"inference_data/{inference_filename}", index=False)

    RMSE_q = torch.sqrt(torch.mean((proprioceptor.q_gt_ts - proprioceptor.q_hat_ts) ** 2, dim=0))
    rel_RMSE_q = RMSE_q / (q_max - q_min)
    RMSE_u = torch.sqrt(torch.mean((proprioceptor.u_gt_ts - proprioceptor.u_hat_ts) ** 2, dim=0))

    print(f"Finished configuration optimization for seed={seed} of {test_dataset_name}")
    print(f"RMSE_q: {RMSE_q.cpu().numpy()} m")
    print(f"Relative RMSE_q: {rel_RMSE_q.cpu().numpy() * 100} %")
    print(f"RMSE_u: {RMSE_u.cpu().numpy()} mT")

    plot_sensor_predictions(sample_rate, proprioceptor.u_gt_ts.cpu().numpy(), proprioceptor.u_hat_ts.cpu().numpy())

    if kinematic_parametrization == "cc":
        plot_cc_configuration_estimates(
            sample_rate, 
            proprioceptor.q_gt_ts.cpu().numpy(), 
            proprioceptor.q_hat_ts.cpu().numpy()
        )
    elif kinematic_parametrization == "ac":
        plot_ac_configuration_estimates(
            sample_rate, 
            proprioceptor.q_gt_ts.cpu().numpy(), 
            proprioceptor.q_hat_ts.cpu().numpy()
        )
    else:
        raise ValueError("Unknown kinematic parametrization: {}".format(kinematic_parametrization))
