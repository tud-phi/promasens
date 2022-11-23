import pandas as pd
import progressbar
import torch

from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.utils.df_to_tensor_utils import inference_df_to_tensors
from promasens.visualization.pyvista_scene_visualizer import PyvistaSceneVisualizer

# use affine curvature
from promasens.constants.constants import SIMULATION_AFFINE_CURVATURE_SEGMENT_PARAMS as kinematic_params


q = torch.tensor([[45 / 180 * torch.pi, 180 / 180 * torch.pi, 45 / 180 * torch.pi, 0.025]])

if __name__ == '__main__':
    robot_kinematics = RobotKinematics(**kinematic_params)
    visualizer = PyvistaSceneVisualizer(robot_kinematics=robot_kinematics, **kinematic_params,
                                        show_silicone=True, enable_shadows=False)
    visualizer.run(q_gt=q)

    visualizer.close()
