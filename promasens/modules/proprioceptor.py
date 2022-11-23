import numpy as np
from scipy import optimize
import torch
from torch import nn
from torch.autograd.functional import jacobian
from typing import *

from promasens.utils.check_freq_activation import check_freq_activation
from .sensor_measurement_predictor import SensorMeasurementPredictor


def np_loss_fnc(q_hat_array: np.array, predictor: SensorMeasurementPredictor, q_hat_init: torch.Tensor,
                u_gt: torch.Tensor, optim_selector: torch.Tensor) -> np.array:
    q_hat = q_hat_init.detach().clone()
    q_hat[optim_selector] = q_hat.new_tensor(q_hat_array)
    u_hat = predictor.forward(q_hat)

    rmse_loss = torch.sqrt(torch.mean((u_hat - u_gt) ** 2))

    return rmse_loss.detach().cpu().numpy()


class Proprioceptor(nn.Module):
    def __init__(self, predictor: SensorMeasurementPredictor,
                 q_optim_bool: torch.Tensor = None, q_min: torch.Tensor = None, q_max: torch.Tensor = None,
                 max_num_iterations: int = 10, gamma: Union[float, torch.Tensor] = 1.E-8, mu: float = 0.2,
                 grid_search_num_samples: int = 25, verbose: bool = False):
        super(Proprioceptor, self).__init__()

        self.predictor = predictor
        self.verbose = verbose

        self.num_segments = self.predictor.robot_kinematics.num_segments
        self.num_sensors = self.predictor.robot_kinematics.num_sensors

        if q_optim_bool is None:
            self.q_optim_bool = torch.ones((self.num_segments, 3), dtype=torch.bool)
        else:
            self.q_optim_bool = q_optim_bool
        self.q_min, self.q_max = q_min, q_max
        assert q_optim_bool.size() == self.q_min.size() == self.q_max.size()

        q_optim_bool_flat = self.q_optim_bool.flatten()
        q_min_flat, q_max_flat = self.q_min.flatten(), self.q_max.flatten()
        self.q_ranges_list = []
        for idx in range(q_optim_bool_flat.size(0)):
            if q_optim_bool_flat[idx]:
                self.q_ranges_list.append((q_min_flat[idx].item(), q_max_flat[idx].item()))

        # initialize time
        self.t = 0.

        self.gamma = gamma  # learning rate
        self.mu = mu  # momentum
        self.use_optimizer = False
        self.max_num_iterations = max_num_iterations
        self.limit_gd_to_dataset_range = False

        self.early_stopping = False
        self.early_stopping_patience = 5
        assert self.max_num_iterations > self.early_stopping_patience
        self.delta_RMSE_u_thresh = 0.001

        # global optimization variables
        self.global_optimization_method = "brute"
        self.grid_search_num_samples = grid_search_num_samples
        # grid and cost function values of grid search (e.g. brute search)
        self.brute_grid, self.brute_cost = None, None

        # initialize memory variables
        self.q_hat_param = nn.UninitializedParameter()
        # best guess of current configuration together with estimate of u
        self.q_hat, self.u_hat, self.rmse_u = None, None, None
        # current ground-truth configuration and sensor value
        self.q_gt, self.u_gt = None, None
        # list of intermediate results of gradient descent iterations
        self.q_hat_its, self.u_hat_its, self.rmse_u_its = None, None, None
        # time history of proprioceptive signals
        self.q_hat_ts, self.u_hat_ts, self.rmse_u_ts = None, None, None
        # time history of ground-truth
        self.q_gt_ts, self.u_gt_ts = None, None
        # global optimization result
        self.q_hat_global, self.u_hat_global = None, None
        # gradient descent update
        self.b_prior = None

        # time optim params
        self.sample_rate, self.global_optim_freq, self.global_optim_delay = None, None, None

    def set_time_optim_params(self, sample_rate: float, global_optim_freq: float = 0., global_optim_delay: float = 0.):
        if global_optim_freq > 0.:
            assert global_optim_freq <= sample_rate
            assert global_optim_delay <= (1 / global_optim_freq)
            assert check_freq_activation(global_optim_delay, 1 / sample_rate)
        elif global_optim_freq == 0.:
            assert global_optim_delay == 0.
        else:
            raise ValueError

        self.sample_rate = sample_rate
        self.global_optim_freq, self.global_optim_delay = global_optim_freq, global_optim_delay
        if self.global_optim_freq > 0.:
            self.limit_gd_to_dataset_range = True

    def forward(self, q_hat_init: torch.Tensor, u_gt: torch.Tensor, optimize_globally: bool = False,
                q_gt: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        q_hat = q_hat_init.clone()
        if q_gt is not None:
            q_hat[~self.q_optim_bool] = q_gt[~self.q_optim_bool]
        self.q_gt, self.u_gt = q_gt, u_gt

        if optimize_globally:
            q_hat, u_hat = self.run_global_optimization(q_hat, u_gt)
            print("Global optimum of grid search at q_hat=", q_hat.numpy())

        q_hat, u_hat, rmse_u = self.run_gradient_descent(q_hat, u_gt)

        self.q_hat, self.u_hat, self.rmse_u = q_hat, u_hat, rmse_u

        return q_hat, u_hat

    def run_time_step(self, u_gt: torch.Tensor, q_gt: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize q_init at the beginning of trajectory to ground-truth configuration
        if self.q_hat is None:
            q_hat = q_gt.clone()
        else:
            q_hat = self.q_hat.clone()

        if self.global_optim_freq > 0. and self.global_optim_delay > 0. and self.q_hat_global is not None and \
                check_freq_activation(self.t - self.global_optim_delay, 1 / self.global_optim_freq):
            print(f"Update q_hat_prior to delayed q_global_min={self.q_hat_global} from {self.global_optim_delay}s ago")
            q_hat = self.q_hat_global.clone()

        # Set elements of q_hat, that are not optimized, to q_gt
        if q_gt is not None:
            q_hat[~self.q_optim_bool] = q_gt[~self.q_optim_bool]

        if self.global_optim_freq > 0.:
            optimize_globally = check_freq_activation(self.t, 1 / self.global_optim_freq)
            if optimize_globally:
                self.q_hat_global, self.u_hat_global = self.run_global_optimization(q_hat, u_gt)
                if self.global_optim_delay == 0.:
                    q_hat = self.q_hat_global
                    print("Updating q_init to global optimum as delay is set 0s")

        q_hat, u_hat = self.forward(q_hat, u_gt, optimize_globally=False, q_gt=q_gt)

        q_hat_cp, u_hat_cp = q_hat.clone().unsqueeze(0), u_hat.clone().unsqueeze(0)
        rmse_u_cp, u_gt_cp = self.rmse_u.clone().unsqueeze(0), u_gt.clone().unsqueeze(0)
        if self.t == 0.:
            self.q_hat_ts, self.u_hat_ts, self.rmse_u_ts = q_hat_cp, u_hat_cp, rmse_u_cp
            self.u_gt_ts = u_gt_cp
        else:
            self.q_hat_ts = torch.cat((self.q_hat_ts, q_hat_cp), dim=0)
            self.u_hat_ts = torch.cat((self.u_hat_ts, u_hat_cp), dim=0)
            self.u_gt_ts = torch.cat((self.u_gt_ts, u_gt_cp), dim=0)
            self.rmse_u_ts = torch.cat((self.rmse_u_ts, rmse_u_cp), dim=0)

        if q_gt is not None:
            q_gt_cp = q_gt.clone().unsqueeze(0)
            if self.t == 0.:
                self.q_gt_ts = q_gt_cp
            else:
                self.q_gt_ts = torch.cat((self.q_gt_ts, q_gt_cp), dim=0)

        self.t += 1. / self.sample_rate

        return q_hat, u_hat

    def run_gradient_descent(self, q_hat_init: torch.Tensor, u_gt: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_hat = q_hat_init.clone().requires_grad_()

        q_optim_float = self.q_optim_bool.to(dtype=u_gt.dtype).to(device=q_hat.device)
        if isinstance(self.gamma, torch.Tensor):
            self.gamma = self.gamma.to(device=q_hat.device)

        optimizer = None
        if self.use_optimizer:
            self.q_hat_param = nn.Parameter(q_hat)
            optimizer = torch.optim.SGD([self.q_hat_param], lr=self.gamma, momentum=self.mu)

        q_hat_list, u_hat_list, rmse_u_list = [], [], []
        it = 0
        while it < self.max_num_iterations:
            u_hat = self.predictor.forward(q_hat)
            RMSE_u = torch.sqrt(torch.mean((u_hat - u_gt) ** 2)).detach()

            q_hat_list.append(q_hat.detach().clone())
            u_hat_list.append(u_hat.detach().clone())
            rmse_u_list.append(RMSE_u)

            if self.early_stopping and len(rmse_u_list) > self.early_stopping_patience:
                if torch.abs(rmse_u_list[-self.early_stopping_patience] - rmse_u_list[-1]) < self.delta_RMSE_u_thresh:
                    print(f"Stopped gradient descent at iteration {it} after detecting delta RMSE_u threshold")
                    break

            if it == self.max_num_iterations - 1:
                # we do not want to change q_hat anymore in the last iteration
                print(f"Stopped gradient descent at iteration {it} after surpassing the maximum number of iterations")
                break

            if optimizer is None:
                # solution with autograd of loss function
                # def loss_func(q_hat):
                #     u_hat = self.predictor.forward(q_hat)
                #     mse_loss = torch.mean((u_hat - u) ** 2)
                #     return mse_loss
                # loss_grad = jacobian(loss_func, q_hat)
                # q_hat = q_hat - optim_selector_float * self.gamma * loss_grad

                # manual solution with hand-derived gradient of MSE loss function
                jac = 2. / self.num_sensors * jacobian(func=self.predictor.forward, inputs=q_hat, vectorize=True)
                jac_transposed = jac.transpose(0, 1).transpose(1, 2)
                b = q_optim_float * torch.matmul(jac_transposed, (u_hat - u_gt))
                if self.mu > 0. and self.b_prior is not None:
                    b = b + q_optim_float * self.mu * self.b_prior
                q_hat = q_hat - self.gamma * b
                self.b_prior = b.detach()

                # import timeit
                # num_its = 100
                # lambda_func = lambda: jacobian(func=self.predictor.forward, inputs=q_hat, vectorize=True)
                # print("Mean elapsed time: ", timeit.timeit(lambda_func, number=num_its) / num_its * 1000, " ms")
            else:
                self.q_hat_param.data.copy_(q_hat.data)
                u_hat = self.predictor.forward(self.q_hat_param)
                loss = torch.mean((u_hat - u_gt) ** 2)
                loss.backward()
                optimizer.step()
                q_hat[self.q_optim_bool] = self.q_hat_param[self.q_optim_bool]

            if torch.isnan(q_hat).any():
                # the gradient descent diverged
                print(f"Stopped gradient descent at iteration {it} because it diverged")
                break

            if self.limit_gd_to_dataset_range:
                q_hat[self.q_optim_bool] = torch.clamp(q_hat[self.q_optim_bool], self.q_min[self.q_optim_bool],
                                                       self.q_max[self.q_optim_bool])
            if self.verbose:
                print(f"it={it} with RMSE_u={RMSE_u} and relative q error\n", (q_hat - self.q_gt) / (self.q_max - self.q_min))
            it += 1

        self.q_hat_its, self.u_hat_its = torch.stack(q_hat_list), torch.stack(u_hat_list)
        self.rmse_u_its = torch.stack(rmse_u_list)

        best_it = torch.argmin(self.rmse_u_its)
        q_hat, u_hat, rmse_u = q_hat_list[best_it].clone(), u_hat_list[best_it].clone(), self.rmse_u_its[best_it]
        if self.q_gt is None:
            print(f"Found best RMSE_u at it={best_it} "
                  f"with RMSE_u={rmse_u.cpu().numpy()} and q_hat=\n{q_hat.cpu().numpy()}")
        else:
            error_q_rel = (q_hat - self.q_gt) / (self.q_max - self.q_min)
            print(f"Found best RMSE_u at it={best_it} with RMSE_u={rmse_u.cpu().numpy()} "
                  f"and relative q error=\n{error_q_rel.cpu().numpy()}")

        return q_hat, u_hat, rmse_u

    def run_global_optimization(self, q_hat_init: torch.Tensor, u_gt: torch.Tensor,
                                num_samples: int = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples is None:
            num_samples = self.grid_search_num_samples

        if self.global_optimization_method == "brute":
            brute_output = optimize.brute(np_loss_fnc, ranges=tuple(self.q_ranges_list),
                                          args=(self.predictor, q_hat_init, u_gt, self.q_optim_bool),
                                          Ns=num_samples, finish=None, full_output=True)
            x0, fval, grid, Jout = brute_output
            self.brute_grid, self.brute_cost = u_gt.new_tensor(grid), u_gt.new_tensor(Jout)
        else:
            raise NotImplementedError

        q_hat = q_hat_init.detach().clone()
        q_hat[self.q_optim_bool] = q_hat.new_tensor(x0)
        u_hat = self.predictor(q_hat).detach()

        return q_hat, u_hat
