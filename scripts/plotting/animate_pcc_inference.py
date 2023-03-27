import matplotlib
matplotlib.use("Qt5Cairo")
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from promasens.utils.df_to_tensor_utils import inference_df_to_tensors

# latex text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

dataset_type = "simulated"  # "simulated" or "experimental"
if dataset_type == "simulated":
    dataset_name = "analytical_db_n_b-3_n_s-9_n_m-3_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r" \
                   "_to_" \
                   "analytical_db_n_b-3_n_s-12_n_m-3_T3_n_t-400_inference_sensor_failure"
    legend_size = 6
else:
    dataset_name = "2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1" \
                   "_to_" \
                   "2022-05-02_T3_90deg_P0_R1_inference_optim_q_dx_q_dy"
    legend_size = 8

sample_rate = 40
step_skip = 2

seeds = [0, 1, 2]

alpha_error_band = 0.4

colors = [
    {"gt": "dodgerblue", "hat_mean": "deepskyblue", "hat_std": "cyan"},
    {"gt": "lightcoral", "hat_mean": "coral", "hat_std": "coral"},
    {"gt": "forestgreen", "hat_mean": "limegreen", "hat_std": "palegreen"},
    {"gt": "darkblue", "hat_mean": "mediumblue", "hat_std": "steelblue"},
    {"gt": "sienna", "hat_mean": "chocolate", "hat_std": "sandybrown"},
    {"gt": "seagreen", "hat_mean": "mediumseagreen", "hat_std": "springgreen"},
    {"gt": "royalblue", "hat_mean": "cornflowerblue", "hat_std": "lightskyblue"},
    {"gt": "maroon", "hat_mean": "firebrick", "hat_std": "indianred"},
    {"gt": "darkolivegreen", "hat_mean": "olivedrab", "hat_std": "yellowgreen"},
    {"gt": "magenta", "hat_mean": "orchid", "hat_std": "violet"},
    {"gt": "purple", "hat_mean": "mediumorchid", "hat_std": "plum"},
    {"gt": "indigo", "hat_mean": "rebeccapurple", "hat_std": "mediumpurple"},
]

if __name__ == "__main__":
    q_gt_ts, q_hat_ts = [], []
    u_gt_ts, u_hat_ts = [], []
    for seed in seeds:
        df_seed = pd.read_csv(f'datasets/inference/{dataset_name}_seed_{seed}.csv')

        q_gt_ts_seed, q_hat_ts_seed, u_gt_ts_seed, u_hat_ts_seed, rmse_u_ts_seed = inference_df_to_tensors(
            df=df_seed,
            device=torch.device("cpu")
        )

        q_gt_ts.append(q_gt_ts_seed)
        q_hat_ts.append(q_hat_ts_seed)
        u_gt_ts.append(u_gt_ts_seed)
        u_hat_ts.append(u_hat_ts_seed)

    q_gt_ts = torch.stack(q_gt_ts, dim=0)
    q_hat_ts = torch.stack(q_hat_ts, dim=0)
    u_gt_ts = torch.stack(u_gt_ts, dim=0)
    u_hat_ts = torch.stack(u_hat_ts, dim=0)

    q_gt_ts_mean = q_gt_ts.mean(dim=0)
    q_hat_ts_mean = q_hat_ts.mean(dim=0)
    u_gt_ts_mean = u_gt_ts.mean(dim=0)
    u_hat_ts_mean = u_hat_ts.mean(dim=0)
    q_gt_ts_stdev = q_gt_ts.std(dim=0)
    q_hat_ts_stdev = q_hat_ts.std(dim=0)
    u_gt_ts_stdev = u_gt_ts.std(dim=0)
    u_hat_ts_stdev = u_hat_ts.std(dim=0)

    time = np.arange(q_gt_ts_mean.shape[0]) / sample_rate
    num_segments = q_gt_ts_mean.shape[-2]
    num_sensors = u_gt_ts_mean.shape[-1]

    fig, axes = plt.subplots(2, 1, num="PCC inference", figsize=(6, 4.5), dpi=200)
    ax1 = axes[0]
    ax2 = axes[1]

    u_gt_lines = []
    for j in range(num_sensors):
        line, = ax1.plot(
            [],
            [],
            color=colors[j]["gt"],
            label=r"$u_{" + str(j+1) + "}$"
        )
        u_gt_lines.append(line)

    ax1.set_prop_cycle(None)
    u_hat_mean_lines = []
    for j in range(num_sensors):
        line, = ax1.plot(
            [],
            [],
            linestyle="--",
            color=colors[j]["hat_mean"],
            label=r"$\hat{u}_{" + str(j + 1) + "}$"
        )
        u_hat_mean_lines.append(line)

    ax1.set_prop_cycle(None)
    u_hat_stdev_collections = []
    for j in range(num_sensors):
        col = ax1.fill_between(
            [],
            [],
            [],
            alpha=alpha_error_band,
            color=colors[j]["hat_std"],
        )
        u_hat_stdev_collections.append(col)

    u_gt_not_nan = u_gt_ts[~torch.isnan(u_gt_ts)]  # remove NaN measurements
    u_range = u_gt_not_nan.max() - u_gt_not_nan.min()
    ax1.set_xlim([0, time[-1]])
    ax1.set_ylim([
        u_gt_not_nan.min() - 0.1 * u_range,
        u_gt_not_nan.max() + 0.1 * u_range
    ])

    ax1.set_title("Sensor measurements")
    # ax1.set_xlabel(r"Time [s]")
    ax1_ylabel = r"$u$ [mV]" if dataset_type == "experimental" else r"$u$ [mT]"
    ax1.set_ylabel(ax1_ylabel)
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=ax1_handles[:u_gt_ts_mean.shape[-1]],
        labels=ax1_labels[:u_gt_ts_mean.shape[-1]],
        loc="upper right",
        prop={'size': legend_size},
        bbox_to_anchor=(1.005, 1.03),
        # bbox_to_anchor=(1.15, 1.04), outside of plot
    )
    ax1.grid(True)

    q_gt_mean_lines = []
    for i in range(num_segments):
        Delta_x_gt_mean_line, = ax2.plot(
            [], [],
            color=colors[3 * i]["gt"],
            label=r"$\Delta_x$"
        )
        Delta_y_gt_mean_line, = ax2.plot(
            [], [],
            color=colors[3 * i + 1]["gt"],
            label=r"$\Delta_y$"
        )
        dL_gt_mean_line, = ax2.plot(
            [], [],
            color=colors[3 * i + 2]["gt"],
            label=r"$\delta L$"
        )
        q_gt_mean_lines.extend([Delta_x_gt_mean_line, Delta_y_gt_mean_line, dL_gt_mean_line])

    plt.gca().set_prop_cycle(None)
    q_hat_mean_lines = []
    for i in range(num_segments):
        Delta_x_hat_mean_line, = ax2.plot(
            [], [],
            linestyle="--",
            color=colors[3 * i]["hat_mean"],
            label=r"$\hat{\Delta}_{x," + str(i + 1) + "} L$ [mm]"
        )
        Delta_y_hat_mean_line, = ax2.plot(
            [], [],
            linestyle="--",
            color=colors[3 * i + 1]["hat_mean"],
            label=r"$\hat{\Delta}_{y," + str(i + 1) + "} L$ [mm]"
        )
        dL_hat_mean_line, = ax2.plot(
            [], [],
            linestyle="--",
            color=colors[3 * i + 2]["hat_mean"],
            label=r"$\hat{\delta}_{" + str(i + 1) + "} L$ [mm]"
        )
        q_hat_mean_lines.extend([Delta_x_hat_mean_line, Delta_y_hat_mean_line, dL_hat_mean_line])

    plt.gca().set_prop_cycle(None)
    q_hat_stdev_cols = []
    for i in range(num_segments):
        Delta_x_hat_stdev_col = ax2.fill_between(
            [], [], [],
            alpha=alpha_error_band,
            color=colors[3 * i]["hat_std"],
        )
        Delta_y_hat_stdev_col = ax2.fill_between(
            [], [], [],
            alpha=alpha_error_band,
            color=colors[3 * i + 1]["hat_std"],
        )
        dL_hat_stdev_col = ax2.fill_between(
            [], [], [],
            alpha=alpha_error_band,
            color=colors[3 * i + 2]["hat_std"],
        )
        q_hat_stdev_cols.extend([Delta_x_hat_stdev_col, Delta_y_hat_stdev_col, dL_hat_stdev_col])

    q_gt_ts_mean_for_lim = q_gt_ts_mean.clone() * 1e3
    q_range = q_gt_ts_mean_for_lim.max() - q_gt_ts_mean_for_lim.min()
    ax2.set_xlim([0, time[-1]])
    ax2.set_ylim([
        q_gt_ts_mean_for_lim.min() - 0.1 * q_range,
        q_gt_ts_mean_for_lim.max() + 0.1 * q_range
    ])

    ax2.set_title("Configuration")
    ax2.set_xlabel(r"Time [s]")
    ax2.set_ylabel(r"$q$ [mm]")
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles=ax2_handles[:len(q_gt_mean_lines)],
        labels=ax2_labels[:len(q_gt_mean_lines)],
        loc="upper right",
        prop={'size': legend_size},
    )
    ax2.grid(True)

    plt.tight_layout()

    # frames
    frames = np.arange(0, time.shape[0], step=step_skip)

    pbar = tqdm(total=time.shape[0])

    def animate(time_idx):
        pbar.update(step_skip)
        for j, u_gt_line in enumerate(u_gt_lines):
            u_gt_line.set_data(time[:time_idx], u_gt_ts_mean[:time_idx, j])
        for j, u_hat_mean_line in enumerate(u_hat_mean_lines):
            u_hat_mean_line.set_data(time[:time_idx], u_hat_ts_mean[:time_idx, j])
        for j, u_hat_stdev_col in enumerate(u_hat_stdev_collections):
            u_hat_stdev_col.remove()
            col = ax1.fill_between(
                [],
                [],
                [],
                alpha=alpha_error_band,
                color=colors[j]["hat_std"],
            )
            u_hat_stdev_collections[j] = col

        for i in range(num_segments):
            q_gt_mean_lines[3 * i].set_data(time[:time_idx], q_gt_ts_mean[:time_idx, i, 0] * 1e3)
            q_gt_mean_lines[3 * i + 1].set_data(time[:time_idx], q_gt_ts_mean[:time_idx, i, 1] * 1e3)
            q_gt_mean_lines[3 * i + 2].set_data(time[:time_idx], q_gt_ts_mean[:time_idx, i, 2] * 1e3)

            q_hat_mean_lines[3 * i].set_data(time[:time_idx], q_hat_ts_mean[:time_idx, i, 0] * 1e3)
            q_hat_mean_lines[3 * i + 1].set_data(time[:time_idx], q_hat_ts_mean[:time_idx, i, 1] * 1e3)
            q_hat_mean_lines[3 * i + 2].set_data(time[:time_idx], q_hat_ts_mean[:time_idx, i, 2] * 1e3)

            q_hat_stdev_cols[3 * i].remove()
            q_hat_stdev_cols[3 * i] = ax2.fill_between(
                time[:time_idx],
                (q_hat_ts_mean[:time_idx, i, 0] - q_hat_ts_stdev[:time_idx, i, 0]) * 1e3,
                (q_hat_ts_mean[:time_idx, i, 0] + q_hat_ts_stdev[:time_idx, i, 0]) * 1e3,
                alpha=alpha_error_band,
                color=colors[3 * i]["hat_std"],
            )

            q_hat_stdev_cols[3 * i + 1].remove()
            q_hat_stdev_cols[3 * i + 1] = ax2.fill_between(
                time[:time_idx],
                (q_hat_ts_mean[:time_idx, i, 1] - q_hat_ts_stdev[:time_idx, i, 1]) * 1e3,
                (q_hat_ts_mean[:time_idx, i, 1] + q_hat_ts_stdev[:time_idx, i, 1]) * 1e3,
                alpha=alpha_error_band,
                color=colors[3 * i + 1]["hat_std"],
            )

            q_hat_stdev_cols[3 * i + 2].remove()
            q_hat_stdev_cols[3 * i + 2] = ax2.fill_between(
                time[:time_idx],
                (q_hat_ts_mean[:time_idx, i, 2] - q_hat_ts_stdev[:time_idx, i, 2]) * 1e3,
                (q_hat_ts_mean[:time_idx, i, 2] + q_hat_ts_stdev[:time_idx, i, 2]) * 1e3,
                alpha=alpha_error_band,
                color=colors[3 * i + 2]["hat_std"],
            )

        lines = (
                u_gt_lines + u_hat_mean_lines + u_hat_stdev_collections
                + q_gt_mean_lines + q_hat_mean_lines + q_hat_stdev_cols
        )
        return lines

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=frames,
        interval=step_skip*1000/sample_rate,
        blit=True,
    )

    movie_writer = animation.FFMpegWriter(fps=sample_rate)
    video_folder = "pcc_simulations" if dataset_type == "simulated" else "experiments"
    ani.save(f"videos/{video_folder}/{dataset_name}_plot.mp4")

    plt.show()
    pbar.close()
