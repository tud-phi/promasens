import csv
import pandas as pd
import torch

from promasens.modules.kinematics.mcs_kinematics import McsKinematics
from promasens.modules.kinematics.robot_kinematics import RobotKinematics
from promasens.modules.kinematics import SensorMagnetKinematics

if __name__ == '__main__':
    # print(type(segment_kinematics))  # torch.jit.ScriptFunction
    # See the compiled graph as Python code
    # print(segment_kinematics.code)

    dtype = torch.float32

    phi_off_deg = 0
    phi_off = phi_off_deg / 180. * np.pi

    from promasens.constants.constants import EXPERIMENT_ROBOT_PARAMS_20220502 as params
    params['phi_off'] = phi_off

    dataset_name = f'2022-05-02_T3_P{phi_off_deg}_R1'
    # dataset_name = f'2022-02-11_GBN_180s_320mb_REF_False_P{phi_off_deg}_R1'
    dataset_path = f'MCS_databases/{dataset_name}_Quat.csv'

    # check whether the dataset is a raw extract from Motive, or if it is already processed
    is_motive_dataset = False
    with open(dataset_path, 'r', newline='', encoding='UTF8') as f:
        for row in csv.reader(f):
            if row[0] == 'Format Version' and row[1] == '1.23':
                # dataset is raw extract from Motive
                is_motive_dataset = True
            break

    if is_motive_dataset:
        print("Recognized raw motive dataset. Proceeding with processing...")
        column_names = ["frame_idx", "t",
                        "X_b", "Y_b", "Z_b", "W_b",	"TX_b",	"TY_b",	"TZ_b",	"mean_error_b",
                        "X_r",	"Y_r",	"Z_r",	"W_r",	"TX_r",	"TY_r",	"TZ_r",	"mean_error_r"]
        df = pd.read_csv(dataset_path, index_col=0, names=column_names, skiprows=7)
    else:
        print("Recognized processed MCS dataset with renamed column headers. Proceeding with processing...")
        df = pd.read_csv(dataset_path)

    # interpolate missing MCS data (because the MCS has lost track of the markers for an instant)
    df = df.interpolate(method='linear', limit=2)

    MCS_kinematics = McsKinematics(**params)  # initialize the class
    q_r = MCS_kinematics(df)

    robot_kinematics = RobotKinematics(**params)
    sensor_magnet_kinematics = SensorMagnetKinematics(robot_kinematics)

    # Call the function using the TorchScript interpreter
    with open(f'MCS_databases/{dataset_name}_q.csv', 'w', newline="",
              encoding='UTF8') as f:  # csv file can be found in the same file as this script is in

        header = ['sensor_id', 'q_dx_0', 'q_dy_0', 'q_dL_0', 'lambda', 'd_0', 'alpha_0', 'beta_0', 'theta_0']
        writer = csv.writer(f)
        writer.writerow(header)
        counter = 0

        for i in range(len(torch.transpose(q_r, 0, 1))):
            q = q_r[:, i]

            xi = sensor_magnet_kinematics(q)

            # detach grad and tensor so only values are printed in the csv file
            q = q.detach().numpy()
            xi = xi.detach().numpy()

            print([counter], "q:", q, "xi:\n", xi)

            for j in range(robot_kinematics.num_sensors):
                data = [j, q[0], q[1], q[2], xi[j][0], xi[j][1], xi[j][2], xi[j][3], xi[j][4]]
                writer.writerow(data)

            counter = counter + 1
