import numpy as np
import torch

from ngsolve import *
from netgen.csg import *

from promasens.modules.kinematics.sensor_magnet_kinematics import SensorMagnetKinematics
from promasens.simulation.base_simulator import BaseSimulator


class NGSolveSimulator(BaseSimulator):
   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.geo, self.mesh, self.gfu, self.B, self.H = None, None, None, None, None

        self.mu0 = 4 * np.pi * 1e-7
        self.mur_air = 1.00000037
        # permeability of Neodymium magnets: 1.05
        # https://www.engineeringtoolbox.com/permeability-d_1923.html
        self.mur_magnet = 1.05

        # self.apply_configuration(np.zeros((self.robot_kinematics.num_segments, 3)))

    def apply_configuration(self, q: np.array = None):
        if q is not None:
            # q_array: i x q_i
            self.robot_kinematics.forward(torch.tensor(q))

        # T_magnets: n_m x 4 x 4
        T_magnets = self.robot_kinematics.T_magnets.cpu().numpy()
        T_sensors = self.robot_kinematics.T_sensors.cpu().numpy()

        num_segments, L0 = self.robot_kinematics.num_segments, self.robot_kinematics.L0
        l_m, D_m_in, D_m_out = self.magnet_thickness, self.magnet_inner_diameter, self.magnet_outer_diameter

        T_joint = np.concatenate((T_magnets, T_sensors), axis=0)
        # max_T_joint_abs = np.abs(T_joint[:, :3, 3]).max(axis=0)
        min_T_joint = T_joint[:, :3, 3].min(axis=0)
        max_T_joint = T_joint[:, :3, 3].max(axis=0)

        geo = CSGeometry()
        box_lower_pnt = min_T_joint - L0
        box_upper_pnt = max_T_joint + L0
        box = OrthoBrick(Pnt(box_lower_pnt[0], box_lower_pnt[1], box_lower_pnt[2]),
                         Pnt(box_upper_pnt[0], box_upper_pnt[1], box_upper_pnt[2])).bc("outer")
        air = box
        for k in range(T_magnets.shape[0]):
            T_magnet = T_magnets[k]
            t_m, R_m, = T_magnet[:3, 3], T_magnet[:3, :3]
            n_m, e_m, o_m = T_magnet[:3, 0:1], T_magnet[:3, 1:2],  T_magnet[:3, 2:3]

            magnet_lower_pnt = t_m - R_m @ np.array([0., 0., l_m / 2.])
            magnet_upper_pnt = t_m + R_m @ np.array([0., 0., l_m / 2.])

            magnet_lower_pnt = Pnt(magnet_lower_pnt[0], magnet_lower_pnt[1], magnet_lower_pnt[2])
            magnet_upper_pnt = Pnt(magnet_upper_pnt[0], magnet_upper_pnt[1], magnet_upper_pnt[2])
            magnet_outer_cyl = Cylinder(magnet_lower_pnt, magnet_upper_pnt, D_m_out)  # cylinder of infinite length
            magnet_inner_cyl = Cylinder(magnet_lower_pnt, magnet_upper_pnt, D_m_in)  # cylinder of infinite length

            upper_half_space = Plane(magnet_lower_pnt, Vec(-o_m[0], -o_m[1], -o_m[2]))
            lower_half_space = Plane(magnet_upper_pnt, Vec(o_m[0], o_m[1], o_m[2]))

            magnet = (magnet_outer_cyl - magnet_inner_cyl) * upper_half_space * lower_half_space

            air -= magnet
            geo.Add(magnet.mat(f"m_{k}").maxh(D_m_out/2),
                    col=(0.3+0.05*k, 0.3-0.05*k, 0.1))  # col=(0.3, 0.3, 0.1): color of magnet in RGB

        geo.Add(air.mat("air"), transparent=True)
        # geo.Draw()

        mesh = Mesh(geo.GenerateMesh(maxh=2*L0/num_segments, curvaturesafety=1))
        mesh.Curve(3)

        fes = HCurl(mesh, order=3, dirichlet="outer", nograds=True)
        print("ndof =", fes.ndof)
        u, v = fes.TnT()

        mur_material_dict = {}
        magnetization_material_dict = {}
        for k in range(T_magnets.shape[0]):
            mur_material_dict[f"m_{k}"] = self.mur_magnet
            R = T_magnets[k, :3, :3]

            # Magpylib says that M = magnetization / mu0
            # https://magpylib.readthedocs.io/en/latest/
            magnetization_inertial_frame = tuple((R @ (self.magnetization/self.mu0)).tolist())
            magnetization_material_dict[f"m_{k}"] = magnetization_inertial_frame
        mur = mesh.MaterialCF(mur_material_dict, default=self.mur_air)  # permeability
        # magnetization in inertial frame (x, y, z)
        mag = mesh.MaterialCF(magnetization_material_dict, default=(0, 0, 0))

        a = BilinearForm(fes)
        a += 1 / (self.mu0 * mur) * curl(u) * curl(v) * dx + 1e-8 / (self.mu0 * mur) * u * v * dx
        c = Preconditioner(a, "bddc")

        f = LinearForm(fes)
        curl_v = curl(v)
        for k in range(T_magnets.shape[0]):
            f += mag * curl_v * dx(f"m_{k}")

        with TaskManager():
            a.Assemble()
            f.Assemble()

        gfu = GridFunction(fes)
        with TaskManager():
            solvers.CG(sol=gfu.vec, rhs=f.vec, mat=a.mat, pre=c.mat)

        # the vector potential is not supposed to look nice
        # Draw(gfu, mesh, "vector-potential", draw_surf=False, clipping=True)

        B, H = curl(gfu), 1 / (self.mu0 * mur) * curl(gfu) - mag

        # Draw(B, mesh, "B-field", draw_surf=False, clipping=True)
        # Draw(H, mesh, "H-field", draw_surf=False, clipping=True)

        self.geo, self.mesh, self.gfu, self.B, self.H = geo, mesh, gfu, B, H

    def draw(self):
        self.geo.Draw()
        Draw(self.B, self.mesh, "B-field", draw_surf=False, clipping=True)
        Draw(self.H, self.mesh, "H-field", draw_surf=False, clipping=True)

    def getB(self):
        return self.B

    def get_sensor_measurements(self) -> np.array:
        B_e = None
        if self.add_earth_magnetic_field:
            B_e = self.get_earth_magnetic_field_in_sensor_frames()

        # T_sensors: n_s x 4 x 4
        T_sensors = self.robot_kinematics.T_sensors.cpu().numpy()

        u = np.zeros((T_sensors.shape[0], ))
        for j in range(T_sensors.shape[0]):
            t_s = T_sensors[j, :3, 3]
            R_s = T_sensors[j, :3, :3]

            mip = self.mesh(t_s[0], t_s[1], t_s[2])
            B_j = R_s.T @ self.B(mip)

            if self.add_earth_magnetic_field:
                # add earth magnetic field
                B_j += B_e[j]

            u[j] = B_j[2] * 10**3  # convert from T to mT and only take z-component

        return u
