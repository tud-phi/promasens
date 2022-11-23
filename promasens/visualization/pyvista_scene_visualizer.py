import numpy as np
import pyvista as pv
import seaborn as sns
import torch
from typing import *
import warnings


from promasens.utils.check_freq_activation import check_freq_activation


class PyvistaSceneVisualizer:
    """
    A visualizer for the soft robot, magnets, sensors and magnetic field using PyVista.
    """
    def __init__(self, robot_kinematics,
                 d: float, D_m_in: float, D_m_out: float, t_m: float,
                 B_func = None,
                 show_backbone: bool = True, show_silicone: bool = True,
                 show_sensors: bool = True, show_magnets: bool = True,
                 q_hat_opacity: float = 0.3, enable_shadows: bool = False,
                 base_down_view: bool = True,
                 **kwargs):
        self.robot_kinematics = robot_kinematics
        self.d, self.D_m_in, self.D_m_out, self.t_m = d, D_m_in, D_m_out, t_m
        self.L0 = self.robot_kinematics.L0 * 10 ** 3  # convert to mm

        if enable_shadows:
            assert show_silicone is False, "Shadows are not supported by PyVista / VTK for meshes with opacity"
            warnings.warn('Shadows are not supported by PyVista / VTK for meshes with opacity')

        self.show_backbone, self.show_silicone = show_backbone, show_silicone
        self.show_sensors, self.show_magnets = show_sensors, show_magnets
        self.q_hat_opacity = q_hat_opacity  # opacity of estimated soft robot configuration
        self.enable_shadows, self.base_down_view = enable_shadows, base_down_view
        self.num_points_backbone_spline = 1000

        self.filepath = None
        self.t, self.sample_rate, self.frame_rate = 0., 1., 1.

        self.outputs_q_gt, self.outputs_q_hat = None, None

        # pyvista plotter
        self.pl = None

        # prepare magnetic field
        self.B_func = B_func
        if self.B_func is not None:
            grid_dim = 75
            grid = pv.UniformGrid(
                dims=(grid_dim, grid_dim, grid_dim),
                spacing=(2 * self.L0 / grid_dim, 2 * self.L0 / grid_dim, 1.5 * self.L0 / grid_dim),
                origin=(-self.L0, -self.L0, 0.),
            )
            self.magnetic_field_grid = grid

    def run(self, q_gt: torch.Tensor = None, q_hat: torch.Tensor = None, filepath: str = None):
        self.draw_scene(q_gt=q_gt, q_hat=q_hat)

        self.pl.show(auto_close=False)

        if filepath is not None:
            # self.pl.window_size = (2500, 2500)
            self.pl.ren_win.SetOffScreenRendering(True)
            self.pl.save_graphic(filepath)

        self.pl.close()

    def run_timestep(self, q_gt: torch.Tensor = None, q_hat: torch.Tensor = None):
        if self.t == 0:
            # somehow diffusion is very inconsistent with overlapping meshes from frame to frame
            diffuse = 0.5
            if q_gt is not None and q_hat is not None:
                diffuse = 0.

            self.draw_scene(q_gt=q_gt, q_hat=q_hat, diffuse=diffuse)

            self.pl.open_movie(self.filepath, framerate=self.frame_rate, quality=9)

            self.pl.show(auto_close=False)

            self.pl.ren_win.SetOffScreenRendering(True)

            self.pl.write_frame()  # Write this frame

        elif check_freq_activation(self.t, 1 / self.frame_rate):
            self.update_scene(q_gt=q_gt, q_hat=q_hat)

            self.pl.write_frame()  # Write this frame

        self.t = self.t + 1. / self.sample_rate

    def setup_movie(self, filepath: str,
                    sample_rate: float = 40, frame_rate: float = 20):
        # this method should be run once at the start when creating a movie
        assert frame_rate <= sample_rate, "frame rate of movie should be less than or equal to sample rate"
        assert sample_rate % frame_rate == 0, "sample rate of movie should be a multiple of frame rate"

        self.filepath, self.sample_rate, self.frame_rate = filepath, sample_rate, frame_rate

    def draw_scene(self, q_gt: torch.Tensor = None, q_hat: torch.Tensor = None, diffuse: float = 0.5):
        # create plotter
        plotter_kwargs = {"window_size": [1500, 1500], "lighting": "none"}
        self.pl = pv.Plotter(**plotter_kwargs)

        if self.B_func is not None:
            # compute B-field and add as data to grid
            self.magnetic_field_grid['B'] = self.B_func(self.magnetic_field_grid.points)

        if q_gt is not None:
            self.outputs_q_gt = self.draw_meshes(q=q_gt, diffuse=diffuse, opacity=1.,
                                                 show_magnetic_field_streamlines=True)
        if q_hat is not None:
            self.outputs_q_hat = self.draw_meshes(q=q_hat, diffuse=diffuse, opacity=self.q_hat_opacity,
                                                  show_magnetic_field_streamlines=False)

        # add light
        light = pv.Light(position=(0, 0. * self.L0, 5 * self.L0), show_actor=False, positional=True,
                         cone_angle=60, exponent=20, intensity=2)
        self.pl.add_light(light)

        # add coordinate axis at origin of base frame
        # self.pl.add_axes_at_origin()
        marker_args = dict(cone_radius=0.6, shaft_length=0.7, tip_length=0.3, ambient=0.5,
                           label_size=(0.25, 0.1))
        _ = self.pl.add_axes(line_width=10, marker_args=marker_args, color="black")

        # add floor
        # floor = pl.add_floor(face='-z', opacity=0.5, lighting=True, pad=10.0)
        if self.base_down_view:
            # floor = pv.Plane(i_size=1.0 * self.L0, j_size=1.0 * self.L0, i_resolution=10, j_resolution=10)
            floor = pv.Plane(i_size=101.6, j_size=101.6, i_resolution=10, j_resolution=10)
            self.pl.add_mesh(floor, ambient=0., diffuse=1., specular=0.8, color='black', opacity=0.3)
        else:
            floor = pv.Plane(i_size=1.5 * self.L0, j_size=1.5 * self.L0, i_resolution=5, j_resolution=5)
            self.pl.add_mesh(floor, ambient=0., diffuse=0.5, specular=0.8, color='white', opacity=1.0)

        # display settings
        if self.enable_shadows:
            self.pl.enable_shadows()  # add shadows
        self.pl.set_background("white")
        self.pl.camera_position = "xz"
        self.pl.camera.roll = 180.  # flipping +z to -z --> we are now in (-x, -z) plane
        self.pl.camera.elevation = 18.  # slightly tilt upwards to look from above onto the robot
        self.pl.camera.azimuth = -90.  # rotate into (-y, -z) plane with x-axis coming out of the screen

    def update_scene(self, q_gt: torch.Tensor = None, q_hat: torch.Tensor = None):
        if self.B_func is not None:
            # compute B-field and add as data to grid
            self.magnetic_field_grid['B'] = self.B_func(self.magnetic_field_grid.points)

        if q_gt is not None:
            self.update_meshes(q=q_gt, outputs=self.outputs_q_gt, opacity=1.)
        if q_hat is not None:
            self.update_meshes(q=q_hat, outputs=self.outputs_q_hat, opacity=self.q_hat_opacity)

    def draw_meshes(self, q: torch.Tensor, diffuse: float = 0.5, opacity: float = 1.,
                    show_magnetic_field_streamlines: bool = True) -> Dict:
        # acquire data
        T_bases, T_tips = self.robot_kinematics.apply_configuration(q)

        outputs = {q: q, "T_bases": T_bases, "T_tips": T_tips}

        if self.show_magnets or (show_magnetic_field_streamlines is True and self.B_func is not None):
            T_magnets = self.robot_kinematics.forward_kinematics_magnets().cpu().numpy()
            # add magnets
            magnet_meshes = []
            for k in range(T_magnets.shape[0]):
                T_magnet = T_magnets[k]
                magnet_center = (T_magnet @ np.array([[0], [0], [-self.t_m / 2], [1]])) * 10 ** 3
                ring = pv.CylinderStructured(center=magnet_center[:3, 0], direction=T_magnet[:3, 2],
                                             radius=np.linspace(self.D_m_in / 2, self.D_m_out / 2, 5) * 10 ** 3,
                                             height=self.t_m * 10 ** 3,
                                             theta_resolution=50)
                magnet_meshes.append(ring)
                self.pl.add_mesh(ring, color="green", opacity=opacity, diffuse=0.5, ambient=1.0)
            outputs.update({"T_magnets": T_magnets, "magnet_meshes": magnet_meshes})

        if self.show_sensors:
            T_sensors = self.robot_kinematics.forward_kinematics_sensors().cpu().numpy()
            # add sensors
            sensor_meshes = []
            for j in range(T_sensors.shape[0]):
                T_sensor = T_sensors[j].copy()
                T_sensor[:3, 3] = T_sensor[:3, 3] *10 ** 3
                arrow = pv.Arrow(start=T_sensor[:3, 3], direction=T_sensor[:3, 2],
                                 tip_length=0.25, tip_radius=0.2, shaft_radius=0.1, scale=10)
                # sensor_meshes.append(arrow)
                # self.pl.add_mesh(arrow, color="blue", opacity=opacity, diffuse=diffuse, ambient=1.0)
                sensor_mesh = pv.read("promasens/assets/sensor.stl")
                # scale x length to 10mm
                sensor_mesh.scale(10.5 / (sensor_mesh.bounds[1] - sensor_mesh.bounds[0]) * np.ones((3,)), inplace=True)
                # translate to center of x-axis
                sensor_mesh.translate((-(sensor_mesh.bounds[1] - sensor_mesh.bounds[0]) / 2, 0., 0.), inplace=True)
                # translate to center of y-axis
                sensor_mesh.translate((0., -(sensor_mesh.bounds[3] - sensor_mesh.bounds[2]) / 2, 0.), inplace=True)
                # flip z-axis
                sensor_mesh.flip_z(inplace=True)
                # rotate around z-axis for x-axis of sensor to match x-axis of robot
                sensor_mesh.rotate_z(-90, inplace=True)
                # transform to sensor position
                sensor_mesh.transform(T_sensor, inplace=True)
                merged_sensor_mesh = sensor_mesh.merge(arrow)
                sensor_meshes.append(merged_sensor_mesh)
                self.pl.add_mesh(merged_sensor_mesh, color="blue", opacity=opacity, diffuse=diffuse, ambient=1.0)

            outputs.update({"T_sensors": T_sensors, "sensor_meshes": sensor_meshes})

        # add backbone
        # segment_colors = ["yellow", "orange", "red"]
        # segment_colors = sns.color_palette("rocket", n_colors=robot_kinematics.num_segments)
        segment_colors = sns.dark_palette("blueviolet", n_colors=self.robot_kinematics.num_segments)
        backbone_splines = []
        backbone_meshes, silicone_meshes = [], []
        for i in range(1, self.robot_kinematics.num_segments + 1):
            v_i = torch.linspace(float(i - 1), float(i), 20, dtype=q.dtype, device=q.device)
            segment_points = self.robot_kinematics.get_translations(v_i).squeeze(dim=2).detach().cpu().numpy()*10**3

            # Create spline with 1000 interpolation points
            spline = pv.Spline(segment_points, self.num_points_backbone_spline)
            backbone_splines.append(spline)

            if self.show_backbone:
                tube = spline.tube(radius=self.D_m_in / 2 * 10 ** 3)
                backbone_meshes.append(tube)
                diffusion = 1.0
                if opacity < 1.0:
                    diffusion = 0.
                self.pl.add_mesh(tube, color=segment_colors[i - 1], opacity=opacity, smooth_shading=True,
                                 diffuse=diffuse, ambient=1.0)
            if self.show_silicone:
                tube = spline.tube(radius=self.d * 10 ** 3)
                silicone_meshes.append(tube)
                self.pl.add_mesh(tube, color=segment_colors[i - 1], opacity=opacity/2, smooth_shading=True,
                                 diffuse=diffuse, ambient=1.0)
        outputs.update({"backbone_splines": backbone_splines})
        if self.show_backbone:
            outputs.update({"backbone_meshes": backbone_meshes})
        if self.show_silicone:
            outputs.update({"silicone_meshes": silicone_meshes})

        if show_magnetic_field_streamlines and self.B_func is not None:
            outputs["strl_mesh"] = self.generate_magnetic_field_streamlines(T_magnets)

            # add field lines and legend to scene
            legend_args = {
                'title': 'B [mT]',
                'title_font_size': 30,
                'color': 'black',
                "height": 0.7,
                'position_y': 0.15,
                'vertical': True,
                "label_font_size": 25,
            }

            self.pl.add_mesh(
                outputs["strl_mesh"],
                cmap="coolwarm",
                scalar_bar_args=legend_args,
                opacity=opacity,
                log_scale=True,
                ambient=0.95,
                diffuse=0.05,
                specular=0.,
            )

        return outputs

    def update_meshes(self, q: np.array, outputs: Dict, opacity: float = 1.):
        # acquire data
        L0 = self.robot_kinematics.L0 * 10 ** 3
        T_bases, T_tips = self.robot_kinematics.apply_configuration(q)

        if self.show_magnets or "strl_mesh" in outputs:
            T_magnets = self.robot_kinematics.forward_kinematics_magnets().cpu().numpy()
            for k in range(T_magnets.shape[0]):
                T_magnet = T_magnets[k]
                T_magnet_prior = outputs["T_magnets"][k]
                T_rel = T_magnet @ np.linalg.inv(T_magnet_prior)
                T_rel[:3, 3] *= 10 ** 3  # convert translations to mm
                outputs["magnet_meshes"][k].transform(T_rel, inplace=True)
            outputs["T_magnets"] = T_magnets

        if self.show_sensors:
            T_sensors = self.robot_kinematics.forward_kinematics_sensors().cpu().numpy()
            for j in range(T_sensors.shape[0]):
                T_sensor = T_sensors[j]
                T_sensor_prior = outputs["T_sensors"][j]
                T_rel = T_sensor @ np.linalg.inv(T_sensor_prior)
                T_rel[:3, 3] *= 10 ** 3  # convert translations to mm
                outputs["sensor_meshes"][j].transform(T_rel, inplace=True)
            outputs["T_sensors"] = T_sensors

        # update backbone
        for i in range(1, self.robot_kinematics.num_segments + 1):
            s_i = torch.linspace(float(i - 1), float(i), 20, dtype=q.dtype, device=q.device)
            segment_points = self.robot_kinematics.get_translations(s_i).squeeze(dim=2).detach().cpu().numpy() * 10 ** 3

            spline = pv.Spline(segment_points, self.num_points_backbone_spline)
            outputs["backbone_splines"][i - 1] = spline

            if self.show_backbone:
                new_tube = spline.tube(radius=self.D_m_in / 2 * 10 ** 3)
                tube = outputs["backbone_meshes"][i - 1]
                self.pl.update_coordinates(points=new_tube.points, mesh=tube, render=False)

            if self.show_silicone:
                new_tube = spline.tube(radius=self.d * 10 ** 3)
                tube = outputs["silicone_meshes"][i - 1]
                self.pl.update_coordinates(points=new_tube.points, mesh=tube, render=False)

        outputs.update({"q": q, "T_bases": T_bases, "T_tips": T_tips})

        if "strl_mesh" in outputs:
            strl_mesh = self.generate_magnetic_field_streamlines(T_magnets)
            outputs["strl_mesh"].overwrite(strl_mesh)
            outputs["strl_mesh"] = strl_mesh

    def generate_magnetic_field_streamlines(self, T_magnets: np.array) -> pv.PolyData:
        merged_seed = None
        for k in range(T_magnets.shape[0]):
            # seed = pv.Disc(center=T_magnets[j, :3, 3] * 10 ** 3, normal=T_magnets[j, :3, 2],
            #                       inner=0 * D_m_in / 2 * 10 ** 3, outer=2 * D_m_out / 2 * 10 ** 3,
            #                       r_res=15, c_res=20)
            seed = pv.Sphere(center=T_magnets[k, :3, 3] * 10 ** 3, radius=5 * self.D_m_out / 2 * 10 ** 3,
                             theta_resolution=10, phi_resolution=10)
            merged_seed = merged_seed.merge(seed) if merged_seed is not None else seed

        strl = self.magnetic_field_grid.streamlines_from_source(
            merged_seed,
            vectors='B',
            max_time=self.L0,
            integration_direction='both',
            progress_bar=True,
        )
        # strl = self.magnetic_field_grid.streamlines(
        #     vectors='B',
        #     source_center=T_magnets[0, :3, 3]*10**3,
        #     source_radius=5*D_m_out/2*10**3,
        #     n_points=200,
        #     max_time=self.L0,
        #     integration_direction='both',
        #     progress_bar=True,
        # )

        strl_mesh = strl.tube(radius=.4)

        return strl_mesh

    def close(self):
        self.pl.close()


