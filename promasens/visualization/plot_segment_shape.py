import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D


class Arrow3D(FancyArrowPatch):
    """
    Source: https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
    """

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def plot_segment_shape(T: np.ndarray, T_hat: np.array = None, oal: float = 0.1):
    """
    Plot (discretized) segment shape.
    Expecting numpy array with shape (N, 4, 4) containing se(3) transformation matrices
    where N is the number of discretization points.
    :param T: np.ndarray of shape (N, 4, 4)
        Transformation matrices for each discretization point along rod.
    :param T_hat: np.ndarray of shape (N, 4, 4)
        Transformation matrices of estimated configuration for each discretization point along rod.
    :param oal: float
        length of the orientation arrow
    :return:
    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    t = T[:, :3, 3]

    _draw_segment_shape(ax, T, oal, opacity=1.0)
    if T_hat is not None:
        _draw_segment_shape(ax, T_hat, oal, opacity=0.5)

    # set axis limits - needs to be done before setting aspect ratio
    ax.set_xlim(-0.01 + 1.5 * np.min(t[:, 0]), 0.01 + 1.5 * np.max(t[:, 0]))
    ax.set_ylim(-0.01 + 1.5 * np.min(t[:, 1]), 0.01 + 1.5 * np.max(t[:, 1]))
    ax.set_zlim(np.min(t[:, 2]), 1.2 * np.max(t[:, 2]))

    ax.set_box_aspect([1, 1, 1])
    _set_axes_equal(ax)  # IMPORTANT - this is also required

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    fig.tight_layout()
    plt.show()


def _draw_segment_shape(ax: plt.Axes, T: np.ndarray, oal: float, opacity: float = 1.0):
    t = T[:, :3, 3]
    R = T[:, :3, :3]
    for idx in [
        0,
        int(T.shape[0] // 4),
        int(T.shape[0] // 2),
        int(T.shape[0] * 3 / 4),
        -1,
    ]:
        _draw_node_orientation(ax, t[idx], R[idx], oal, opacity=opacity)
    # for idx in range(T.shape[0]):
    #     plot_node_orientation(t[idx], R[idx])

    ax.plot3D(
        t[:, 0],
        t[:, 1],
        t[:, 2],
        linestyle=":",
        color="black",
        marker="o",
        markersize=3,
        alpha=opacity,
    )


def _draw_node_orientation(
    ax: plt.Axes, t_n: np.ndarray, R_n: np.ndarray, oal: float, opacity: float = 1.0
):
    """
    :param ax: plt.Axes
    :param t_n: translation vector of the node
    :param R_n: rotation matrix of the link / node
    :param oal: orientation arrow length
    :return:
    """
    normal = oal * R_n[:, 0]
    binormal = oal * R_n[:, 1]
    tangent = oal * R_n[:, 2]
    ax.arrow3D(
        x=t_n[0],
        y=t_n[1],
        z=t_n[2],
        dx=normal[0],
        dy=normal[1],
        dz=normal[2],
        color="red",
        mutation_scale=5,
        alpha=opacity,
    )
    ax.arrow3D(
        x=t_n[0],
        y=t_n[1],
        z=t_n[2],
        dx=binormal[0],
        dy=binormal[1],
        dz=binormal[2],
        color="green",
        mutation_scale=5,
    )
    ax.arrow3D(
        x=t_n[0],
        y=t_n[1],
        z=t_n[2],
        dx=tangent[0],
        dy=tangent[1],
        dz=tangent[2],
        color="blue",
        mutation_scale=5,
    )


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    """Add an 3d arrow to an `Axes3D` instance."""

    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


# Functions from @Mateen Ulhaq and @karlo
def _set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


setattr(Axes3D, "arrow3D", _arrow3D)
