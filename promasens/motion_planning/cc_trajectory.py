import numpy as np

from . import TrajectoryType


class CcTrajectory:
    def __init__(self, trajectory_type: TrajectoryType, d: float, num_steps: int = 1000, seed: int = 0):
        self.trajectory_type = trajectory_type

        self.num_steps = num_steps
        self.d = d

        self.Delta_c_max = 45. / 180 * np.pi * self.d  # m
        self.delta_L_max = 0.0055  # [m], roughly 5% of the length of the robot

        self.rng = np.random.default_rng(seed)

    def plan(self, phase: float = 0.) -> np.array:
        steps = np.linspace(0, 1, self.num_steps)
        if self.trajectory_type == TrajectoryType.RANDOM:
            Delta_x = self.rng.uniform(-1.2*self.Delta_c_max, 1.2*self.Delta_c_max, self.num_steps)
            Delta_y = self.rng.uniform(-1.2*self.Delta_c_max, 1.2*self.Delta_c_max, self.num_steps)
            delta_L = self.rng.uniform(0, 1.2*self.delta_L_max, self.num_steps)
        elif self.trajectory_type == TrajectoryType.BENDING_1D:
            Delta_x = self.Delta_c_max*steps
            Delta_y = np.zeros_like(Delta_x)
            delta_L = 0.5 * self.delta_L_max * (1 + np.sin(2 * np.pi * steps + phase))
        elif self.trajectory_type == TrajectoryType.HALF_LEMNISCATE:
            Delta_x = self.Delta_c_max * np.sin(1 * 2 * np.pi * steps + phase)
            Delta_y = self.Delta_c_max * np.sin(0.5 * 2 * np.pi * steps + phase)
            delta_L = 0.5 * self.delta_L_max * (1 + np.sin(2 * np.pi * steps + phase))
        elif self.trajectory_type == TrajectoryType.FULL_LEMNISCATE:
            Delta_x = self.Delta_c_max * np.sin(2 * 2 * np.pi * steps + phase)
            Delta_y = self.Delta_c_max * np.sin(1 * 2 * np.pi * steps + phase)
            delta_L = 0.5 * self.delta_L_max * (1 + np.sin(2 * np.pi * steps + phase))
        # elif self.trajectory_type == TrajectoryType.SPIRAL:
        #     v_q = 1  # m/s (velocity of q)
        #
        #     steps = np.linspace(0, 1, self.num_steps)
        #     A = steps * self.Delta_max
        #     2*pi*A / p = v
        #     2*pi*A * omega = v
        #     f =
        #     Delta_x = A * np.cos(v_q / (2*pi*A) * steps)
        #     Delta_y = A * np.sin(2*np.pi*steps)
        #     delta_L = np.zeros_like(Delta_x)
        else:
            raise NotImplementedError

        q_traj = np.stack((Delta_x, Delta_y, delta_L), axis=1)
        return q_traj
