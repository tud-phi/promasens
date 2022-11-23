import torch
from torch.autograd.functional import jacobian

from promasens.modules.kinematics import AcKinematics
from promasens.visualization.plot_segment_shape import plot_segment_shape


q = torch.tensor([-60 / 180 * torch.pi, 180 / 180 * torch.pi, 45 / 180 * torch.pi, 0.025])

if __name__ == '__main__':
    ac_kinematics = AcKinematics(L0=0.1)

    s = torch.linspace(0, 1, 20)
    q_batched = q.unsqueeze(dim=0).repeat((s.size(0), 1))

    T = ac_kinematics.forward_kinematics_batched(q_batched, s)

    T_np = T.detach().cpu().numpy()
    plot_segment_shape(T_np, oal=0.01)

    jac_T = jacobian(ac_kinematics.forward_kinematics, (q, s[-1]), vectorize=True)
    print("jac_T", jac_T[0].size(), jac_T[1].size())
