import numpy as np

from . import plt


def plot_dataset(sample_rate: float, q_gt_ts: np.array, u_gt_ts: np.array):
    ts = np.arange(0, u_gt_ts.shape[0] / sample_rate, 1 / sample_rate)
    n_b = q_gt_ts.shape[1]
    n_s = u_gt_ts.shape[1]

    # for j in range(n_s):
    #     plt.plot(ts, u_gt_ts[:, j], label="$\hat{u}_" + str(j) + "$")
    #
    # plt.xlabel("Time [s]")
    # plt.ylabel("Magnetic flux density [mT]")
    # plt.legend()
    #
    # plt.show()

    fig, ax1 = plt.subplots(num="Configurations and sensor measurements")
    for i in range(1, n_b+1):
        print("q_gt_ts", q_gt_ts.shape)
        for q_i_idx in range(q_gt_ts.shape[2]):
            if q_i_idx == 0:
                q_i_name = "$\Delta_{x, " + str(i) + "}$"
            elif q_i_idx == 1:
                q_i_name = "$\Delta_{y, " + str(i) + "}$"
            elif q_i_idx == 2:
                q_i_name = "$\delta_{L, " + str(i) + "}$"
            else:
                raise ValueError
            plt.plot(ts, q_gt_ts[:, i-1, q_i_idx] * 1000, label=q_i_name)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Configuration $q$ [mm]")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for j in range(n_s):
        ax2.plot(ts, u_gt_ts[:, j], ':', label="$\hat{u}_" + str(j+1) + "$")
    ax2.set_ylabel("Magnetic flux density $u$ [mT]")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()
