import matplotlib
matplotlib.use("Qt5Cairo")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from promasens.utils.df_to_tensor_utils import inference_df_to_tensors

# latex text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

dataset_name = "analytical_db_ac_n_b-1_n_s-9_n_m-2_T0_n_t-120000_rand_phi_off" \
               "_to_" \
               "analytical_db_ac_n_b-1_n_s-9_n_m-2_T5_n_t-400"
sample_rate = 40
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
        df_seed = pd.read_csv(f'datasets/inference/{dataset_name}_inference_seed_{seed}.csv')

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
    num_sensors = u_gt_ts_mean.shape[-1]

    # plt.figure(num="Affine curvature inference", figsize=(8, 6))
    axes = plt.subplots(2, 1, num="Affine curvature inference", figsize=(6, 4))
    ax1 = axes[1][0]
    ax2 = axes[1][1]

    for j in range(num_sensors):
        ax1.plot(
            time,
            u_gt_ts_mean[:, j],
            color=colors[j]["gt"],
            label=r"$u_" + str(j+1) + "$"
        )
    ax1.set_prop_cycle(None)
    for j in range(num_sensors):
        ax1.plot(
            time,
            u_hat_ts_mean[:, j],
            linestyle="--",
            color=colors[j]["hat_mean"],
            label=r"$\hat{u}_" + str(j + 1) + "$"
        )
    ax1.set_prop_cycle(None)
    for j in range(num_sensors):
        ax1.fill_between(
            time,
            u_hat_ts_mean[:, j] - u_hat_ts_stdev[:, j],
            u_hat_ts_mean[:, j] + u_hat_ts_stdev[:, j],
            alpha=alpha_error_band,
            color=colors[j]["hat_std"],
        )

    # ax1.set_xlabel(r"Time [s]")
    ax1.set_ylabel(r"$u$ [mT]")
    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles=ax1_handles[:u_gt_ts_mean.shape[-1]],
        labels=ax1_labels[:u_gt_ts_mean.shape[-1]],
        loc="upper right",
        prop={'size': 8},
        bbox_to_anchor=(1.005, 1.03),
        # bbox_to_anchor=(1.15, 1.04), outside of plot
    )

    ax2.plot(
        time,
        q_gt_ts_mean[:, 0, 0],
        color=colors[0]["gt"],
        label=r"$\kappa_0$ [rad/m]"
    )
    ax2.plot(
        time,
        q_gt_ts_mean[:, 0, 1],
        color=colors[1]["gt"],
        label=r"$\kappa_1$ [rad/m\textsuperscript{2}]"
    )
    ax2.plot(
        time,
        q_gt_ts_mean[:, 0, 2],
        color=colors[2]["gt"],
        label=r"$\phi$ [rad]"
    )
    ax2.plot(
        time,
        q_gt_ts_mean[:, 0, 3] * 1e3,
        color=colors[3]["gt"],
        label=r"$\delta L$ [mm]"
    )
    plt.gca().set_prop_cycle(None)
    ax2.plot(
        time, q_hat_ts_mean[:, 0, 0],
        linestyle="--",
        color=colors[0]["hat_mean"],
        label=r"$\hat{\kappa}_0$ [rad/m]"
    )
    ax2.plot(
        time,
        q_hat_ts_mean[:, 0, 1],
        linestyle="--",
        color=colors[1]["hat_mean"],
        label=r"$\hat{\kappa}_1$ [rad/m\textsuperscript{2}]"
    )
    ax2.plot(
        time,
        q_hat_ts_mean[:, 0, 2],
        linestyle="--",
        color=colors[2]["hat_mean"],
        label=r"$\hat{\phi}$ [rad]"
    )
    ax2.plot(
        time,
        q_hat_ts_mean[:, 0, 3] * 1e3,
        linestyle="--",
        color=colors[3]["hat_mean"],
        label=r"$\hat{\delta} L$ [mm]"
    )
    plt.gca().set_prop_cycle(None)
    ax2.fill_between(
        time,
        q_hat_ts_mean[:, 0, 0] - q_hat_ts_stdev[:, 0, 0],
        q_hat_ts_mean[:, 0, 0] + q_hat_ts_stdev[:, 0, 0],
        alpha=alpha_error_band,
        color=colors[0]["hat_std"],
    )
    ax2.fill_between(
        time,
        q_hat_ts_mean[:, 0, 1] - q_hat_ts_stdev[:, 0, 1],
        q_hat_ts_mean[:, 0, 1] + q_hat_ts_stdev[:, 0, 1],
        alpha=alpha_error_band,
        color=colors[1]["hat_std"],
    )
    ax2.fill_between(
        time,
        q_hat_ts_mean[:, 0, 2] - q_hat_ts_stdev[:, 0, 2],
        q_hat_ts_mean[:, 0, 2] + q_hat_ts_stdev[:, 0, 2],
        alpha=alpha_error_band,
        color=colors[2]["hat_std"],
    )
    ax2.fill_between(
        time,
        (q_hat_ts_mean[:, 0, 3] - q_hat_ts_stdev[:, 0, 3]) * 1e3,
        (q_hat_ts_mean[:, 0, 3] + q_hat_ts_stdev[:, 0, 3]) * 1e3,
        alpha=alpha_error_band,
        color=colors[3]["hat_std"],
    )

    ax2.set_xlabel(r"Time [s]")
    ax2.set_ylabel(r"Configuration $q$")
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    ax2.legend(
        handles=ax2_handles[:q_gt_ts_mean.shape[-1]],
        labels=ax2_labels[:q_gt_ts_mean.shape[-1]],
        loc="upper right",
        prop={'size': 8},
    )

    plt.tight_layout()
    plt.show()
