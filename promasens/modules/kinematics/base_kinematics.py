from pytorch3d.transforms import euler_angles_to_matrix
import torch
from torch import linalg, nn


class BaseKinematics(nn.Module):
    def __init__(self):
        super().__init__()

        self.eps = 1.19209e-07

    def forward_kinematics(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Forward kinematics method for using the abstract forward_kinematics_batched function for 1D tensors
        :param q: torch tensor of size (4, ) containing the configuration variables
        :param v: torch tensor of size (, ) containing the position of the point in the interval [0, 1]
        :return: torch tensor of size (4, 4) containing the transformation matrix
        """
        q_batched, s_batched = q.unsqueeze(0), v.unsqueeze(0)
        T = self.forward_kinematics_batched(q_batched, s_batched).squeeze(0)
        return T

    @staticmethod
    def compute_fixed_in_plane_transformations(
            n: int,
            d_r: torch.Tensor = None,
            phi: torch.Tensor = None,
            psi: torch.Tensor = None,
            polarization: torch.Tensor = None,
            device: torch.device = None,
    ) -> torch.Tensor:
        """
        Compute the fixed in-plane transformations for the given batch of n points
        :param n: batch size
        :param d_r: torch tensor of size (n, ) containing the radial distance of the point from the backbone
        :param phi: torch tensor of size (n, ) containing the azimuth angle of the point
            (e.g. the direction of the radial offset)
        :param psi: torch tensor of size (n, ) containing the tilt angle of the local coordinate system
        :param polarization: torch tensor of size (n, ) containing the polarization
            (e.g. 1 or -1). (-1) corresponds to a 180-degree rotation around the local x-axis
        """
        if d_r is None:
            d_r = torch.zeros(n, device=device)
        if phi is None:
            phi = torch.zeros(n, device=device)
        if psi is None:
            psi = torch.zeros(n, device=device)
        if polarization is None:
            polarization = torch.ones(n, device=device)

        assert d_r.size(0) == n, "d_r must be of size (n, )"
        assert phi.size(0) == n, "phi must be of size (n, )"
        assert psi.size(0) == n, "psi must be of size (n, )"
        assert polarization.size(0) == n, "polarization must be of size (n, )"

        t_fixed = torch.zeros((n, 3, 1), device=device)
        t_fixed[:, 0, 0] = d_r * torch.cos(phi)
        t_fixed[:, 1, 0] = d_r * torch.sin(phi)

        euler_zyx_r_planar = torch.zeros((n, 3))
        # first we rotate around z-axis
        euler_zyx_r_planar[:, 0] = phi
        # then we rotate around y-axis
        # rotation around tangential axis to backbone circle
        euler_zyx_r_planar[:, 1] = psi
        # application of polarization
        euler_zyx_r_planar[:, 2] = - (polarization - 1.0) / 2 * torch.pi
        R_fixed = euler_angles_to_matrix(euler_zyx_r_planar, 'ZYX')

        T_fixed = torch.zeros((n, 4, 4), device=device)
        T_fixed[..., :3, :3] = R_fixed
        T_fixed[..., :3, 3:4] = t_fixed
        T_fixed[:, 3, 3] = torch.ones(n, device=device)

        return T_fixed
