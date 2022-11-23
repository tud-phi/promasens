import numpy as np

from . import plt


def plot_sensor_predictions(sample_rate: float, u_gt_ts: np.array, u_hat_ts: np.array = None):
    ts = np.arange(0, u_gt_ts.shape[0] / sample_rate, 1 / sample_rate)
    n_s = u_gt_ts.shape[1]

    for j in range(n_s):
        plt.plot(ts, u_gt_ts[:, j], label=r"$u_{" + str(j+1) + r"}$")

    plt.gca().set_prop_cycle(None)

    if u_hat_ts is not None:
        for j in range(n_s):
            plt.plot(ts, u_hat_ts[:, j], ':', label=r"$\hat{u}_{" + str(j+1) + r"}$")

    plt.xlabel("Time [s]")
    plt.ylabel("Magnetic flux density [mT]")
    plt.legend(ncol=2)

    plt.show()
