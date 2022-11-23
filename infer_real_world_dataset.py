import numpy as np
import pandas as pd
import torch
from typing import *

from promasens.enums.joint_nn_mode import JointNNMode
from promasens.modules.proprioceptor import Proprioceptor
from promasens.utils.check_freq_activation import check_freq_activation
from promasens.utils.load_predictor import load_predictor
from promasens.visualization import plt
from promasens.visualization.loss_landscape_plotter import LossLandscapePlotter

# device ='cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cpu')
print('Using device:', device)
dtype = torch.float32

phi_off_deg = 0
phi_off = phi_off_deg / 180. * np.pi
# dataset_name = f"2022-02-2{dataset_name}1_spiral2_480s_start_200mb_max_425mb_f0_20"
train_dataset_name = f"2022-05-02_FLOWER_SLOW_NOMINAL_P{int(phi_off_deg)}_R1"
test_dataset_name = f"2022-05-02_T3_P{int(phi_off_deg)}_R1"
test_dataset_path = f"{test_dataset_name}_test"

# init random seed
seed = 0
np.random.seed(seed)

# dataset that is used
df = pd.read_csv(f'merged_databases/{test_dataset_path}.csv').dropna(axis=0)

# parse to old df format
df = df.rename(columns={'q_dx_0': 'q_dx', 'q_dy_0': 'q_dy', 'q_dL_0': 'q_dL'})

from promasens.constants.constants import EXPERIMENT_ROBOT_PARAMS_20220502 as kinematic_params
num_sensors = len(df["sensor_id"].unique())
sample_rate = 40

# smooth the data
# df_sensor_list = []
# for j in range(num_sensors):
#     df_sensor = df[df["sensor_id"] == j]
#     window_size = 10
#     df_sensor = df_sensor.rolling(window=window_size, min_periods=1).mean()
#     df_sensor_list.append(df_sensor)
# df = pd.concat(df_sensor_list).sort_index()

joint_model_all_sensors = False
use_pl_model = True

# Averaging Weights Leads to Wider Optima and Better Generalization
# https://arxiv.org/abs/1803.05407
use_swa = True

optim_dx = True
optim_dy = True
optim_dL = False
optim_variable_list = []
if optim_dx:
    optim_variable_list.append("q_dx")
if optim_dy:
    optim_variable_list.append("q_dy")
if optim_dL:
    optim_variable_list.append("q_dL")

# global optimization settings
global_optim_freq = 0.
global_optim_delay = 0.  # in seconds

# gradient descent settings
max_num_iterations = 20
gamma = 1.5E-8  # gradient descent learning rate
mu = 0.2  # gradient descent momentum
if train_dataset_name == "2022-05-02_RAND_NOMINAL_600s_P0_R1":
    gamma = 1.E-8  # gradient descent learning rate
inference_filename = f"{train_dataset_name}_to_{test_dataset_name}_inference_optim_{'_'.join(optim_variable_list)}_seed_{seed}.csv"

num_samples_loss_landscape = 50
plot_loss_landscape_freq = 0.
plot_loss_landscape_state_history = False

predictor = load_predictor(train_dataset_name, kinematic_params, num_segments=1, num_sensors=num_sensors,
                           seed=seed, joint_nn_mode=JointNNMode.EACH_SENSOR,
                           use_pl_model=use_pl_model, use_swa=use_swa, device=device)
q_min = torch.tensor([[df["q_dx"].min(), df["q_dy"].min(), df["q_dL"].min()]], device=device)
q_max = torch.tensor([[df["q_dx"].max(), df["q_dy"].max(), df["q_dL"].max()]], device=device)
q_optim_bool = torch.tensor([[optim_dx, optim_dy, optim_dL]], device=device)
proprioceptor = Proprioceptor(predictor=predictor, q_optim_bool=q_optim_bool, q_min=q_min, q_max=q_max,
                              max_num_iterations=max_num_iterations, gamma=gamma, mu=mu).to(device)
proprioceptor.set_time_optim_params(sample_rate, global_optim_freq, global_optim_delay)


def predict_sensor_measurements(q_hat: np.array) -> np.array:
    if type(q_hat) is not torch.Tensor:
        q_hat = torch.tensor(q_hat, dtype=dtype, device=device)
    else:
        raise ValueError

    u_hat = predictor(q_hat).detach().cpu().numpy()

    return u_hat


def estimate_q_hat(time_idx: int, q_prior: np.array, q_gt: np.array, u_gt: np.array,
                   df_history: pd.DataFrame = None) -> Tuple[np.array, np.array, np.array, np.array]:
    q_gt = np.expand_dims(q_gt, axis=0)
    u_gt_tensor = torch.tensor(u_gt, dtype=dtype, device=device)
    q_gt_tensor = torch.tensor(q_gt, dtype=dtype, device=device)

    q_hat, u_hat = proprioceptor.run_time_step(u_gt_tensor, q_gt_tensor)
    q_hat_its, u_hat_its = proprioceptor.q_hat_its, proprioceptor.u_hat_its
    q_hat, u_hat = q_hat.cpu().numpy(), u_hat.cpu().numpy()
    q_hat_its, u_hat_its = q_hat_its.cpu().numpy(), u_hat_its.cpu().numpy()
    q_gt_ts, q_hat_ts = proprioceptor.q_gt_ts.cpu().numpy(), proprioceptor.q_hat_ts.cpu().numpy()

    q_global_min, u_global_min = None, None
    if proprioceptor.q_hat_global is not None:
        q_global_min, u_global_min = proprioceptor.q_hat_global.squeeze(), proprioceptor.u_hat_global

    rel_q_error = (q_hat - q_gt) / np.array([(df["q_dx"].max() - df["q_dx"].min()),
                                             (df["q_dy"].max() - df["q_dy"].min()),
                                             (df["q_dL"].max() - df["q_dL"].min())])
    rmse_rel_q = np.sqrt(np.mean(rel_q_error[q_optim_bool] ** 2))
    print("Relative RMSE of q_hat:", rmse_rel_q * 100, "%")

    if (plot_loss_landscape_freq > 0. and check_freq_activation(time_idx, sample_rate / plot_loss_landscape_freq)) \
            or rmse_rel_q > np.nan:

        q_hat_global, u_hat_global = proprioceptor.run_global_optimization(q_hat_init=q_gt_tensor, u_gt=u_gt_tensor,
                                                                           num_samples=num_samples_loss_landscape)
        q_hat_global, u_hat_global = q_hat_global.cpu().numpy(), u_hat_global.cpu().numpy()
        samples_q, samples_rmse_u = proprioceptor.brute_grid.cpu().numpy(), proprioceptor.brute_cost.cpu().numpy()

        loss_landscape_plotter = LossLandscapePlotter(q_optim_bool=q_optim_bool.cpu().numpy(), rmse_u_limit=100)

        loss_landscape_kwargs = dict(t=proprioceptor.t,
                                     samples_q=samples_q, samples_rmse_u=samples_rmse_u,
                                     q_hat=q_hat, u_hat=u_hat, q_gt=q_gt, u_gt=u_gt,
                                     q_hat_global=q_hat_global, u_hat_global=u_hat_global,
                                     q_hat_its=q_hat_its, u_hat_its=u_hat_its,
                                     )
        if plot_loss_landscape_state_history:
            loss_landscape_kwargs["q_gt_ts"], loss_landscape_kwargs["q_hat_ts"] = q_gt_ts, q_hat_ts
        loss_landscape_plotter.plot(**loss_landscape_kwargs)

    return q_hat.squeeze(), u_hat, q_global_min, u_global_min


def plot_q_traj():
    print("Plotting q_traj...")
    # plot configuration values over dataset
    df_sensor_1 = df[df["sensor_id"] == 1]
    plt.plot(df_sensor_1["time_idx"] / sample_rate, df_sensor_1["q_dx"], label="$\Delta_x$")
    plt.plot(df_sensor_1["time_idx"] / sample_rate, df_sensor_1["q_dy"], label="$\Delta_y$")
    plt.plot(df_sensor_1["time_idx"] / sample_rate, df_sensor_1["q_dL"], label="$\delta L$")
    plt.xlabel("Time [s]")
    plt.ylabel("$q$")
    plt.legend()
    plt.show()


def plot_neural_network_predictions():
    samples = []
    for time_idx in df["time_idx"].unique():
        df_at_t = df[df["time_idx"] == time_idx]
        q_gt = df_at_t[["q_dx", "q_dy", "q_dL"]].iloc[0].to_numpy()

        u_gt = df_at_t["u"].to_numpy()
        assert u_gt.shape[0] == num_sensors

        u_hat = predict_sensor_measurements(q_gt)

        error_u = u_hat - u_gt
        rmse = np.sqrt(np.mean(error_u ** 2))

        sample = {}
        sample["q_gt_dx"] = q_gt[0].item()
        sample["q_gt_dy"] = q_gt[1].item()
        sample["q_gt_dL"] = q_gt[2].item()

        for sensor_idx in range(u_hat.shape[0]):
            sample["time_idx"] = time_idx
            sample[f"u_gt_{sensor_idx}"] = u_gt[sensor_idx].item()
            sample[f"u_hat_{sensor_idx}"] = u_hat[sensor_idx].item()
            sample[f"error_u_{sensor_idx}"] = error_u[sensor_idx].item()

        sample["RMSE_u"] = rmse.item()

        samples.append(sample)

    df_samples = pd.DataFrame(samples)

    fig, axes = plt.subplots(2)
    for sensor_idx in range(num_sensors):
        axes[0].plot(df_samples["time_idx"] / sample_rate, df_samples[f"u_gt_{sensor_idx}"],
                     label="$u_" + str(sensor_idx) + "$")

    axes[0].set_prop_cycle(None)

    for sensor_idx in range(num_sensors):
        axes[0].plot(df_samples["time_idx"] / sample_rate, df_samples[f"u_hat_{sensor_idx}"], ':',
                     label="$\hat{u}_" + str(sensor_idx) + "$")

    axes[0].set_title("Neural network predictions")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("$u$ [mV]")
    axes[0].legend()

    axes[1].set_prop_cycle(None)

    axes[1].set_title("RMSE")
    axes[1].plot(df_samples["time_idx"] / sample_rate, df_samples[f"RMSE_u"], label="$\mathrm{RMSE}_u$")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("$\mathrm{RMSE}_u$ [mV]")
    axes[1].legend()

    fig.tight_layout()

    plt.show()

    print("RMSE for sensor measurement prediction for all samples in trajectory: ",
          np.sqrt(np.mean(df_samples["RMSE_u"].to_numpy() ** 2)).item())


def plot_q_hat():
    samples = []
    q_hat_prior = None
    q_global_min_prior = None
    df_samples = None
    for time_idx in df["time_idx"].unique():
        print(f"Optimize q_hat at time={format(time_idx / sample_rate, '.3f')}s "
              f"of {format(len(df['time_idx'].unique()) / sample_rate, '.3f')}s")

        df_at_t = df[df["time_idx"] == time_idx]
        q_gt = df_at_t[["q_dx", "q_dy", "q_dL"]].iloc[0].to_numpy()

        assert q_gt.shape[0] == num_sensors
        u_gt = df_at_t["u"].to_numpy()

        q_hat, u_hat, q_global_min, u_global_min = estimate_q_hat(time_idx, q_hat_prior, q_gt, u_gt, df_samples)

        error_u = u_hat - u_gt
        rmse = np.sqrt(np.mean(error_u ** 2))

        sample = {}
        sample["time_idx"] = time_idx
        sample["q_gt_dx_0"] = q_gt[0].item()
        sample["q_gt_dy_0"] = q_gt[1].item()
        sample["q_gt_dL_0"] = q_gt[2].item()
        sample["q_hat_dx_0"] = q_hat[0].item()
        sample["q_hat_dy_0"] = q_hat[1].item()
        sample["q_hat_dL_0"] = q_hat[2].item()
        if q_global_min is not None:
            sample["q_global_min_dx"] = q_global_min[0].item()
            sample["q_global_min_dy"] = q_global_min[1].item()
            sample["q_global_min_dL"] = q_global_min[2].item()

        for sensor_idx in range(u_hat.shape[0]):
            sample["time_idx"] = time_idx
            sample[f"u_gt_{sensor_idx}"] = u_gt[sensor_idx].item()
            sample[f"u_hat_{sensor_idx}"] = u_hat[sensor_idx].item()
            sample[f"error_u_{sensor_idx}"] = error_u[sensor_idx].item()
            if u_global_min is not None:
                sample[f"u_global_min_{sensor_idx}"] = u_global_min[sensor_idx].item()

        sample["rmse_u"] = rmse.item()

        samples.append(sample)
        df_samples = pd.DataFrame(samples)

    df_samples = pd.DataFrame(samples)
    inference_filepath = f"inference_data/{inference_filename}"
    print("Saving inference data to file:", inference_filepath)
    df_samples.to_csv(inference_filepath)

    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_gt_dx_0"], label=f"q_gt_dx")
    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_gt_dy_0"], label=f"q_gt_dy")
    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_gt_dL_0"], label=f"q_gt_dL")

    plt.gca().set_prop_cycle(None)

    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_hat_dx_0"], ':', label=f"q_hat_dx")
    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_hat_dy_0"], ':', label=f"q_hat_dy")
    plt.plot(df_samples["time_idx"] / sample_rate, df_samples[f"q_hat_dL_0"], ':', label=f"q_hat_dL")

    plt.title("Optimal configuration estimates")
    plt.xlabel("time [s]")
    plt.ylabel("q")
    plt.legend()
    plt.show()

    RMSE_dx = np.sqrt(np.mean((df_samples["q_hat_dx_0"] - df_samples["q_gt_dx_0"]).to_numpy() ** 2)).item()
    RMSE_dy = np.sqrt(np.mean((df_samples["q_hat_dy_0"] - df_samples["q_gt_dy_0"]).to_numpy() ** 2)).item()
    RMSE_dL = np.sqrt(np.mean((df_samples["q_hat_dL_0"] - df_samples["q_gt_dL_0"]).to_numpy() ** 2)).item()

    rel_RMSE_dx = RMSE_dx / (df_samples["q_gt_dx_0"].max() - df_samples["q_gt_dx_0"].min())
    rel_RMSE_dy = RMSE_dy / (df_samples["q_gt_dy_0"].max() - df_samples["q_gt_dy_0"].min())
    rel_RMSE_dL = RMSE_dL / (df_samples["q_gt_dL_0"].max() - df_samples["q_gt_dL_0"].min())

    print(f"Finished configuration optimization for seed={seed} of {test_dataset_name}")
    print(f"RMSE_q: RMSE_dx={RMSE_dx}m, RMSE_dy={RMSE_dy}m, RMSE_dL={RMSE_dL}m")
    print(f"Relative RMSE_q: RMSE_dx={rel_RMSE_dx*100}%, RMSE_dy={rel_RMSE_dy*100}%, RMSE_dL={rel_RMSE_dL*100}%")


if __name__ == "__main__":
    plot_q_traj()
    plot_neural_network_predictions()
    plot_q_hat()
