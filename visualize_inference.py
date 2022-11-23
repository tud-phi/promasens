import pandas as pd
import progressbar

from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.utils.df_to_tensor_utils import inference_df_to_tensors
from promasens.visualization.pyvista_scene_visualizer import PyvistaSceneVisualizer

dataset_type = "analytical"
if dataset_type == "analytical":
    from promasens.constants.constants import SIMULATION_ROBOT_PARAMS as kinematic_params

    # kinematic parameters of robot
    kinematic_parametrization = kinematic_params.get("kinematic_parametrization", "cc")
    num_segments = kinematic_params["num_segments"]
    num_magnets_per_segment = 1 if type(kinematic_params["d_m"]) == float else len(kinematic_params["d_m"])
    num_magnets = num_segments * num_magnets_per_segment
    num_sensors = num_segments * kinematic_params["num_sensors_per_segment"]

    if kinematic_parametrization == "cc":
        inference_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r" \
                         f"_to_" \
                         f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T3_n_t-400_inference_seed_0"
        # inference_name = f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_random_emf_rand_phi_off" \
        #                   "_to_" \
        #                  f"analytical_db_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T3_n_t-400_emf_(1.0,0.0,0.0)_inference_seed_0"
    elif kinematic_parametrization == "ac":
        inference_name = f"analytical_db_ac_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T0_n_t-120000_rand_phi_off" \
                         f"_to_" \
                         f"analytical_db_ac_n_b-{num_segments}_n_s-{num_sensors}_n_m-{num_magnets}_T5_n_t-400_inference_seed_0"
    else:
        raise NotImplementedError(f"kinematic parametrization {kinematic_parametrization} not implemented")
else:
    from promasens.constants.constants import EXPERIMENT_ROBOT_PARAMS_20220502 as kinematic_params

    inference_name = "2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1_to_" \
                     "2022-05-02_T3_90deg_P0_R1_inference_optim_q_dx_q_dy_seed_0"

df = pd.read_csv(f"inference_data/{inference_name}.csv")
q_gt_ts, q_hat_ts, _, _, _ = inference_df_to_tensors(df)

robot_kinematics = RobotKinematics(**kinematic_params)

if __name__ == '__main__':
    visualizer = PyvistaSceneVisualizer(robot_kinematics=robot_kinematics, **kinematic_params,
                                        show_silicone=True, enable_shadows=False)
    visualizer.setup_movie(filepath=f"videos/{inference_name}.mp4", sample_rate=40, frame_rate=20)

    print("Rendering movie frames...")
    for t in progressbar.progressbar(range(q_gt_ts.size(0))):
        visualizer.run_timestep(q_gt=q_gt_ts[t], q_hat=q_hat_ts[t])

    visualizer.close()
