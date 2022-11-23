from contextlib import contextmanager
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.axes import Axes
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from typing import *

from promasens.visualization import plt


class LossLandscapePlotter:
    def __init__(self, q_optim_bool: np.array, rmse_u_limit: float = None):
        self.q_optim_bool = q_optim_bool
        self.rmse_u_limit = rmse_u_limit

        # number of degrees of freedom
        self.dof = self.q_optim_bool.sum().item()
        optim_indices_array = np.array(np.nonzero(self.q_optim_bool))  # np.array: dof x 2
        # cast to list([tuple([x, y])]) with len(list) = dof
        self.optim_indices = []
        for idx in range(optim_indices_array.shape[0]):
            self.optim_indices.append(tuple(optim_indices_array[idx]))

        self.fig: Figure = None
        self.ax: Axes = None
        self.clev, self.cmap = None, None
        self.cs: ContourSet = None
        self.lines: dict = {}
        self.movie_writer = None

        self.latex_var_mapping = ["$\Delta_{x,i}$", "$\Delta_{y,i}$", "$\delta L_{i}$"]

    @contextmanager
    def movie(self, *args, **kwargs):
        try:
            self.setup_movie(*args, **kwargs)
            yield self
        finally:
            self.finish_movie()

    def setup_movie(self, filepath: str, frame_rate: int = 1):
        self.fig = plt.figure(f"Animated loss landscape")

        # self.movie_writer = animation.writers['ffmpeg'](fps=frame_rate)
        metadata = dict(title='Loss landscape', artist='Maximilian Stölzle')
        self.movie_writer = FFMpegWriter(fps=frame_rate, metadata=metadata)
        self.movie_writer.setup(self.fig, outfile=filepath, dpi=600)

    def finish_movie(self):
        self.movie_writer.finish()

    def plot(self, t: float = None, *args, **kwargs):
        """
        Plot the loss landscape.
        Currently, the implementation only supports the optimization of one or two variables / DoFs.
        n: number of samples per variable
        """

        time_str = " at time t = {:.2f} s".format(t) if t is not None else ""
        self.fig = plt.figure(f"Loss landscape{time_str}")

        self.draw_plot(*args, **kwargs)

        plt.show()

    def run_step(self, *args, **kwargs):
        if self.ax is None:
            self.draw_plot(*args, **kwargs)
        else:
            self.update_plot(*args, **kwargs)

        self.movie_writer.grab_frame()

    def draw_plot(self, samples_q: np.array, samples_rmse_u: np.array,
                  q_hat: np.array = None, u_hat: np.array = None,
                  q_gt: np.array = None, u_gt: np.array = None,
                  q_hat_global: np.array = None, u_hat_global: np.array = None,
                  q_hat_its: np.array = None, u_hat_its: np.array = None,
                  q_gt_ts: np.array = None, q_hat_ts: np.array = None,
                  t: float = None):
        samples_q = np.expand_dims(samples_q, axis=0) if len(samples_q.shape) == 1 else samples_q
        self.validate_samples(samples_q, samples_rmse_u)

        self.ax = self.fig.add_subplot(111)

        # plot loss landscape
        if self.dof == 1:
            self.lines["rmse_u"] = self.ax.plot(samples_q[0], samples_rmse_u, marker=".", label="$\mathrm{RMSE}_u$")[0]

            if q_hat_its is not None and u_hat_its is not None:
                self.lines["q_hat_init"] = self.ax.plot(q_hat_its[0, self.optim_indices[0][0], self.optim_indices[0][1]],
                                                        np.sqrt(np.mean((u_hat_its - u_gt) ** 2, axis=1))[0], "cv",
                                                        label="$\hat{q}_0$")[0]
                self.lines["q_hat_its"] = self.ax.plot(q_hat_its[:, self.optim_indices[0][0], self.optim_indices[0][1]],
                                                       np.sqrt(np.mean((u_hat_its - u_gt) ** 2, axis=1)), "r",
                                                       label="$\hat{q}_l$")[0]

            if q_gt is not None:
                # u_hat_q_gt = predict_sensor_measurements(q_gt)
                # rmse_q_gt = np.sqrt(np.mean((u_hat_q_gt - u_gt) ** 2)).item()
                rmse_q_gt = np.zeros((q_gt.shape[0],))
                self.lines["q_gt"] = self.ax.plot(q_gt[self.q_optim_bool], rmse_q_gt, "gs", label="$q$")[0]

            if q_hat_global is not None and u_hat_global is not None and u_gt is not None:
                rmse_u_hat_global = np.sqrt(np.mean((u_hat_global - u_gt) ** 2)).item()
                self.lines["q_hat_global"] = self.ax.plot(q_hat_global[self.q_optim_bool], rmse_u_hat_global, "mP",
                                                          label="$\hat{q}_\mathrm{global}^*$")[0]

            if q_hat is not None and u_hat is not None and u_gt is not None:
                rmse_q_hat = np.sqrt(np.mean((u_hat - u_gt) ** 2)).item()
                self.lines["q_hat"] = self.ax.plot(q_hat[self.q_optim_bool], rmse_q_hat, "r^", label="$\hat{q}$")[0]

            xlabel = self.latex_var_mapping[self.optim_indices[0][1]].replace("i", str(self.optim_indices[0][0]))
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel("$\mathrm{RMSE}_u$ [mV]")
            self.ax.set_ylim(0, min(self.rmse_u_limit, samples_rmse_u.max()))
        elif self.dof == 2:
            # 3D visualization
            # x = df_samples[optim_variable_list_hat[0]]
            # y = df_samples[optim_variable_list_hat[1]]
            # z = df_samples["RMSE_u"]
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.plot_trisurf(x, y, z, cmap=plt.cm.jet, alpha=0.5)
            # ax.scatter(x, y, z, c='black', s=5)  # plot loss landscape
            # # ground-truth configuration
            # ax.scatter(df_samples[optim_variable_list_gt[0]], df_samples[optim_variable_list_gt[1]], RMSE_q_gt, c='red',
            #            s=50)
            # ax.set_title(f"Loss landscape for sample at time {time_idx / sample_rate}s")
            # ax.set_xlabel(optim_variable_list[0])
            # ax.set_ylabel(optim_variable_list[1])
            # ax.set_zlabel("RMSE for u")
            # ax.set_zlim(0, min(RMSE_limit, df_samples["RMSE_u"].max()))
            # ax.set_box_aspect((np.ptp(x), np.ptp(y), np.max([np.ptp(x), np.ptp(y)])))
            # plt.show()

            # cs = ax.contourf(mat_dim_0, mat_dim_1, mat_RMSE_u, cmap=plt.cm.jet)
            rmse_u_limit = samples_rmse_u.max() if self.rmse_u_limit is None else self.rmse_u_limit
            max_rmse = max(min(rmse_u_limit, samples_rmse_u.max()), samples_rmse_u.min())
            color_discretization = (max_rmse - samples_rmse_u.min()) / 100
            self.clev = np.arange(samples_rmse_u.min(), max_rmse, step=color_discretization)
            self.cmap = plt.get_cmap("jet")
            self.cs = self.ax.contourf(samples_q[0] * 10 ** 3, samples_q[1] * 10 ** 3, samples_rmse_u,
                                       self.clev, cmap=self.cmap)

            # ground-truth configuration
            if q_hat_its is not None:
                self.lines["q_hat_init"] = \
                    self.ax.plot(q_hat_its[0, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                                 q_hat_its[0, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3,
                                 "rv", label="$\hat{q}_0$")[0]
                self.lines["q_hat_its"] = \
                    self.ax.plot(q_hat_its[:, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                                 q_hat_its[:, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3,
                                 "r", marker='.', label="$\hat{q}_l$")[0]

            if q_gt_ts is not None:
                self.lines["q_gt_ts"] = \
                    self.ax.plot(q_gt_ts[:, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                                 q_gt_ts[:, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3,
                                 "g", marker='.', label="$q(t)$")[0]

            if q_hat_ts is not None:
                self.lines["q_hat_ts"] = \
                    self.ax.plot(q_hat_ts[:, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                                 q_hat_ts[:, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3,
                                 "c", marker='.', label="$\hat{q}(t)$")[0]

            if q_hat is not None:
                self.lines["q_hat"] = self.ax.plot(q_hat[self.optim_indices[0]] * 10 ** 3,
                                                   q_hat[self.optim_indices[1]] * 10 ** 3,
                                                   "c^", label="$\hat{q}_{l^*}$")[0]

            if q_hat_global is not None:
                self.lines["q_hat_global"] = self.ax.plot(q_hat_global[self.optim_indices[0]] * 10 ** 3,
                                                          q_hat_global[self.optim_indices[1]] * 10 ** 3, "mP",
                                                          label="$\hat{q}_\mathrm{global}^*$")[0]

            if q_gt is not None:
                self.lines["q_gt"] = self.ax.plot(q_gt[self.optim_indices[0]] * 10 ** 3,
                                                  q_gt[self.optim_indices[1]] * 10 ** 3,
                                                  'gs', label="$q$")[0]

            self.ax.set_xlim(samples_q[0].min() * 10 ** 3, samples_q[0].max() * 10 ** 3)
            self.ax.set_ylim(samples_q[1].min() * 10 ** 3, samples_q[1].max() * 10 ** 3)

            xlabel = self.latex_var_mapping[self.optim_indices[0][1]]
            xlabel = xlabel.replace("i", str(self.optim_indices[0][0])) + " [mm]"
            ylabel = self.latex_var_mapping[self.optim_indices[1][1]]
            ylabel = ylabel.replace("i", str(self.optim_indices[1][0])) + " [mm]"
            self.ax.set_xlabel(xlabel)
            self.ax.set_ylabel(ylabel)
            self.ax.set_aspect('equal', 'box')

            self.fig.colorbar(self.cs, label="RMSE $\hat{u}$ [mV]")

        else:
            raise NotImplementedError

        self.ax.legend()
        self.fig.tight_layout()

    def update_plot(self, samples_q: np.array = None, samples_rmse_u: np.array = None,
                    q_hat: np.array = None, u_hat: np.array = None,
                    q_gt: np.array = None, u_gt: np.array = None,
                    q_hat_global: np.array = None, u_hat_global: np.array = None,
                    q_hat_its: np.array = None, u_hat_its: np.array = None,
                    q_gt_ts: np.array = None, q_hat_ts: np.array = None,
                    t: float = None):
        samples_q = np.expand_dims(samples_q, axis=0) if len(samples_q.shape) == 1 else samples_q
        self.validate_samples(samples_q, samples_rmse_u)

        if self.dof == 1:
            if samples_q is not None and samples_rmse_u is not None and "rmse_u" in self.lines:
                self.lines["rmse_u"].set_data(samples_q[0], samples_rmse_u)

            if q_hat_its is not None and u_hat_its is not None and "q_hat_init" in self.lines:
                self.lines["q_hat_init"].set_data(q_hat_its[0, self.optim_indices[0][0], self.optim_indices[0][1]],
                                                  np.sqrt(np.mean((u_hat_its - u_gt) ** 2, axis=1))[0])
            if q_hat_its is not None and u_hat_its is not None and "q_hat_its" in self.lines:
                self.lines["q_hat_its"].set_data(q_hat_its[:, self.optim_indices[0][0], self.optim_indices[0][1]],
                                                 np.sqrt(np.mean((u_hat_its - u_gt) ** 2, axis=1)))

            if q_gt is not None and "q_gt" in self.lines:
                # u_hat_q_gt = predict_sensor_measurements(q_gt)
                # rmse_q_gt = np.sqrt(np.mean((u_hat_q_gt - u_gt) ** 2)).item()
                rmse_q_gt = np.zeros((q_gt.shape[0],))
                self.lines["q_gt"].set_data(q_gt[self.q_optim_bool], rmse_q_gt)

            if q_hat_global is not None and u_hat_global is not None and u_gt is not None \
                    and "q_hat_global" in self.lines:
                rmse_u_hat_global = np.sqrt(np.mean((u_hat_global - u_gt) ** 2)).item()
                self.lines["q_hat_global"].set_data(q_hat_global[self.q_optim_bool], rmse_u_hat_global)

            if q_hat is not None and u_hat is not None and u_gt is not None and "q_hat" in self.lines:
                rmse_q_hat = np.sqrt(np.mean((u_hat - u_gt) ** 2)).item()
                self.lines["q_hat"].set_data(q_hat[self.q_optim_bool], rmse_q_hat)

        elif self.dof == 2:
            if samples_q is not None and samples_rmse_u is not None and self.cs is not None:
                self.validate_samples(samples_q, samples_rmse_u)

                for coll in self.cs.collections:
                    plt.gca().collections.remove(coll)
                self.cs = self.ax.contourf(samples_q[0] * 10 ** 3, samples_q[1] * 10 ** 3, samples_rmse_u,
                                           self.clev, cmap=self.cmap)

            # ground-truth configuration
            if q_hat_its is not None and "q_hat_init" in self.lines:
                self.lines["q_hat_init"].set_data(
                    q_hat_its[0, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                    q_hat_its[0, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3)

            if q_hat_its is not None and "q_hat_its" in self.lines:
                self.lines["q_hat_its"].set_data(q_hat_its[:, self.optim_indices[0][0],
                                                 self.optim_indices[0][1]] * 10 ** 3,
                                                 q_hat_its[:, self.optim_indices[1][0],
                                                 self.optim_indices[1][1]] * 10 ** 3)

            if q_gt_ts is not None and "q_gt_ts" in self.lines:
                self.lines["q_gt_ts"].set_data(q_gt_ts[:, self.optim_indices[0][0], self.optim_indices[0][1]] * 10 ** 3,
                                               q_gt_ts[:, self.optim_indices[1][0], self.optim_indices[1][1]] * 10 ** 3)

            if q_hat_ts is not None and "q_hat_ts" in self.lines:
                self.lines["q_hat_ts"].set_data(q_hat_ts[:, self.optim_indices[0][0], self.optim_indices[0][1]]*10**3,
                                                q_hat_ts[:, self.optim_indices[1][0], self.optim_indices[1][1]]*10**3)

            if q_hat is not None and "q_hat" in self.lines:
                self.lines["q_hat"].set_data(q_hat[self.optim_indices[0]] * 10 ** 3, q_hat[self.optim_indices[1]]*10**3)

            if q_hat_global is not None and "q_hat_global" in self.lines:
                self.lines["q_hat_global"].set_data(q_hat_global[self.optim_indices[0]] * 10 ** 3,
                                                    q_hat_global[self.optim_indices[1]] * 10 ** 3)

            if q_gt is not None and "q_gt" in self.lines:
                self.lines["q_gt"].set_data(q_gt[self.optim_indices[0]] * 10 ** 3, q_gt[self.optim_indices[1]]*10**3)

    def validate_samples(self, samples_q: np.array, samples_rmse_u: np.array):
        # the matplotlib contour function expects the data to be in the form of a regular grid with side length n
        assert samples_q.shape[0] == self.dof
        assert samples_rmse_u.shape == samples_q.shape[1:]
