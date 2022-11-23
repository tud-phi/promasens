import matplotlib.colors as colors
import numpy as np

from promasens.visualization import plt

def streamplot_2D(grid: np.array, B: np.array, T_sensors: np.array = None, L0 = None):
    amp = np.linalg.norm(B, axis=2)

    fig = plt.figure()
    ax = fig.add_subplot()

    # display field in figure with matplotlib
    print("B field amplitude min: ", np.min(amp), "max: ", np.max(amp))
    stream = ax.streamplot(grid[:, :, 0], grid[:, :, 2], B[:, :, 0], B[:, :, 2],
                           density=2, color=amp, linewidth=1, cmap='plasma',
                           norm=colors.LogNorm(vmin=np.min(amp), vmax=np.max(amp)))
    ax.set_xlabel('$x$ [mm]')
    ax.set_ylabel('$z$ [mm]')
    ax.axis('equal')
    plt.colorbar(stream.lines, ax=ax, label='$|B|$ [mT]')

    if T_sensors is not None and L0 is not None:
        # plot sensor measurement direction
        arrow_length = 0.05 * L0 * 10 ** 3  # in mm, 5% of segment length

        for j in range(T_sensors.shape[0]):
            arrow_x_y = T_sensors[j, [0, 2], 3] * 10 ** 3
            o_xz = T_sensors[j, [0, 2], 2]
            arrow_dx_dy = (o_xz / np.linalg.norm(o_xz)) * arrow_length
            plt.arrow(x=arrow_x_y[0], y=arrow_x_y[1], dx=arrow_dx_dy[0], dy=arrow_dx_dy[1],
                      width=1.5, head_width=5., head_length=0.4 * arrow_length, facecolor='black')
            ax.text(arrow_x_y[0] + 0.2 * arrow_length, arrow_x_y[1] + 0.2 * arrow_length, "$s_{" + str(j) + "}$",
                    color='black',
                    fontsize="x-large")

    ax.set_xlim(grid[:, :, 0].min(), grid[:, :, 0].max())
    ax.set_ylim(grid[:, :, 2].min(), grid[:, :, 2].max())

    plt.tight_layout()
    plt.show()
