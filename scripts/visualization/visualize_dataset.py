import pandas as pd
import progressbar

from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.utils.df_to_tensor_utils import database_df_to_tensors
from promasens.visualization.pyvista_scene_visualizer import PyvistaSceneVisualizer

dataset_type = "analytical"
if dataset_type == "analytical":
    from promasens.constants.constants import SIMULATION_AFFINE_CURVATURE_SEGMENT_PARAMS as kinematic_params

    # kinematic parameters of robot
    num_segments = kinematic_params["num_segments"]
    if isinstance(kinematic_params["d_m"], float):
        num_magnets = num_segments
    else:
        num_magnets = len(kinematic_params["d_m"]) * num_segments
    num_sensors = num_segments * kinematic_params["num_sensors_per_segment"]

    inference_name = f"analytical_db_ac_" \
                     f"n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_" \
                     f"T3_n_t-400"
else:
    from promasens.constants.constants import EXPERIMENT_ROBOT_PARAMS_20220502 as kinematic_params

    inference_name = "2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1_to_" \
                     "2022-05-02_T3_90deg_P0_R1_inference_optim_q_dx_q_dy_seed_0"

df = pd.read_csv(f"analytical_databases/{inference_name}.csv")
q_gt_ts, _, _ = database_df_to_tensors(df)

robot_kinematics = RobotKinematics(**kinematic_params)

if __name__ == '__main__':
    visualizer = PyvistaSceneVisualizer(robot_kinematics=robot_kinematics, **kinematic_params,
                                        show_silicone=True, enable_shadows=False)
    visualizer.setup_movie(filepath=f"videos/{inference_name}.mp4", sample_rate=40, frame_rate=20)

    print("Rendering movie frames...")
    for t in progressbar.progressbar(range(q_gt_ts.size(0))):
        visualizer.run_timestep(q_gt=q_gt_ts[t])

    visualizer.close()
