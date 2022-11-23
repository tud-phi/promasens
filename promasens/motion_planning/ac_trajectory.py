import numpy as np

from . import TrajectoryType


class AcTrajectory:
    def __init__(self, trajectory_type: TrajectoryType, num_steps: int = 1000, seed: int = 0):
        self.trajectory_type = trajectory_type

        self.num_steps = num_steps

        self.kappa0_max = 45. / 180 * np.pi  # rad / m
        self.kappa1_max = 180. / 180 * np.pi  # rad / m^2
        self.delta_L_max = 0.0055  # [m], roughly 5% of the length of the robot

        self.rng = np.random.default_rng(seed)

    def plan(self, phase: float = 0.) -> np.array:
        steps = np.linspace(0, 1, self.num_steps)
        if self.trajectory_type == TrajectoryType.RANDOM:
            kappa0 = self.rng.uniform(-1.2 * self.kappa0_max, 1.2 * self.kappa0_max, self.num_steps)
            kappa1 = self.rng.uniform(-1.2 * self.kappa1_max, 1.2 * self.kappa1_max, self.num_steps)
            phi = self.rng.uniform(0, 2 * np.pi, self.num_steps)
            delta_L = self.rng.uniform(0, 1.2*self.delta_L_max, self.num_steps)
        elif self.trajectory_type == TrajectoryType.FULL_LEMNISCATE:
            dx = - np.sin(1 * 2 * np.pi * steps + phase)
            dy = np.sin(2 * 2 * np.pi * steps + phase)

            kappa0 = -self.kappa0_max * np.sqrt(dx**2 + dy**2)
            kappa1 = self.kappa1_max * np.sqrt(dx**2 + dy**2)

            phi = np.arctan2(dy, dx)
            delta_L = 0.5 * self.delta_L_max * (1 + np.sin(2 * np.pi * steps + phase))
        elif self.trajectory_type == TrajectoryType.FLOWER:
            kappa0 = self.kappa0_max * (0.0 + 1.0 * (0.5 - 0.5 * np.cos(3 * 2 * np.pi * steps + phase)))
            kappa1 = self.kappa1_max * (0.1 + 0.9 * (0.5 - 0.5 * np.cos(3 * 2 * np.pi * steps + phase)))
            phi = 2 * np.pi * steps + phase
            delta_L = 0.5 * self.delta_L_max * (1 + np.sin(2 * np.pi * steps + phase))
        else:
            raise NotImplementedError

        q_traj = np.stack((kappa0, kappa1, phi, delta_L), axis=1)
        return q_traj
