import numpy as np

from . import plt


def plot_cc_configuration_estimates(sample_rate: float, q_gt_ts: np.array, q_hat_ts: np.array = None):
    ts = np.arange(0, q_gt_ts.shape[0] / sample_rate, 1 / sample_rate)
    n_b = q_gt_ts.shape[1]

    for i in range(1, n_b+1):
        for q_i_idx in range(q_gt_ts.shape[2]):
            if q_i_idx == 0:
                q_i_name = r"$\Delta_{x, " + str(i) + r"}$"
            elif q_i_idx == 1:
                q_i_name = r"$\Delta_{y, " + str(i) + r"}$"
            elif q_i_idx == 2:
                q_i_name = r"$\delta_{L, " + str(i) + r"}$"
            else:
                raise ValueError
            plt.plot(ts, q_gt_ts[:, i-1, q_i_idx]*1000, label=q_i_name)

    plt.gca().set_prop_cycle(None)

    if q_hat_ts is not None:
        for i in range(1, n_b+1):
            for q_i_idx in range(q_gt_ts.shape[2]):
                if q_i_idx == 0:
                    q_i_name = r"$\hat{\Delta}_{x, " + str(i) + r"}$"
                elif q_i_idx == 1:
                    q_i_name = r"$\hat{\Delta}_{y, " + str(i) + r"}$"
                elif q_i_idx == 2:
                    q_i_name = r"$\hat{\delta}_{L, " + str(i) + r"}$"
                else:
                    raise ValueError
                plt.plot(ts, q_hat_ts[:, i-1, q_i_idx]*1000, ':', label=q_i_name)

    plt.xlabel("Time [s]")
    plt.ylabel("Configuration [mm]")
    plt.legend()

    plt.show()


def plot_ac_configuration_estimates(sample_rate: float, q_gt_ts: np.array, q_hat_ts: np.array = None):
    ts = np.arange(0, q_gt_ts.shape[0] / sample_rate, 1 / sample_rate)
    n_b = q_gt_ts.shape[1]

    for i in range(1, n_b+1):
        for q_i_idx in range(q_gt_ts.shape[2]):
            q_i_values = q_gt_ts[:, i-1, q_i_idx]
            if q_i_idx == 0:
                q_i_name = r"$\kappa_{0, " + str(i) + r"}$ [rad/m]"
            elif q_i_idx == 1:
                q_i_name = r"$\kappa_{1, " + str(i) + r"}$ [rad/(m m)]"
            elif q_i_idx == 2:
                q_i_name = r"$\phi_{" + str(i) + r"}$ [rad]"
            elif q_i_idx == 3:
                q_i_name = r"$\delta_{L, " + str(i) + r"}$ [mm]"
                q_i_values *= 1000
            else:
                raise ValueError
            plt.plot(ts, q_i_values, label=q_i_name)

    plt.gca().set_prop_cycle(None)

    if q_hat_ts is not None:
        for i in range(1, n_b+1):
            for q_i_idx in range(q_hat_ts.shape[2]):
                q_i_values = q_hat_ts[:, i-1, q_i_idx]
                if q_i_idx == 0:
                    q_i_name = r"$\hat{\kappa}_{0, " + str(i) + r"}$ [rad/m]"
                elif q_i_idx == 1:
                    q_i_name = r"$\hat{\kappa}_{1, " + str(i) + r"}$ [rad/(m m)]"
                elif q_i_idx == 2:
                    q_i_name = r"$\hat{\phi}_{" + str(i) + r"}$ [rad]"
                elif q_i_idx == 3:
                    q_i_name = r"$\hat{\delta}_{L, " + str(i) + r"}$ [mm]"
                    q_i_values *= 1000
                else:
                    raise ValueError
                plt.plot(ts, q_i_values, ':', label=q_i_name)

    plt.xlabel("Time [s]")
    plt.ylabel("Configuration")
    plt.legend()

    plt.show()
