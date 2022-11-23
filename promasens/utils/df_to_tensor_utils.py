import numpy as np
import pandas as pd
import torch
from typing import *


def identify_kinematic_parametrization_from_database(df: pd.DataFrame, config_variable_interfix = "") -> Tuple[str, int]:
    intf = config_variable_interfix

    legacy_cc_regex_col_matches = df.filter(regex=f"q{intf}_dx").columns.values
    if len(legacy_cc_regex_col_matches) > 0:
        multisegment_cc_regex_col_matches = df.filter(regex=f"q{intf}_dx_[0-9]").columns.values
        if len(multisegment_cc_regex_col_matches) > 0:
            num_cc_segments = int(multisegment_cc_regex_col_matches[-1].split('_')[-1]) + 1
            return "cc", num_cc_segments
        else:
            # to enable functionality for legacy databases (e.g. in particular real-world datasets)
            num_cc_segments = 1
            return "cc", num_cc_segments

    regex_col_matches = df.filter(regex=f"q{intf}_kappa0_[0-9]").columns.values
    if len(regex_col_matches) > 0:
        num_ac_segments = int(regex_col_matches[-1].split('_')[-1]) + 1
        return "ac", num_ac_segments

    raise ValueError("Could not identify kinematic parametrization from database.")


def database_df_to_tensors(df: pd.DataFrame, time_idx: int = None,
                           separate_sensors: bool = True, manual_superposition: bool = False,
                           num_segments: int = None, num_magnets: int = None,
                           device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a pandas DataFrame to a pair of Tensors.

    :param df: The DataFrame to convert.
    :param time_idx: The index of the time column. If None, the entire dataset is returned.
    :param separate_sensors: Whether to separate the xi and u data for each sensor.
    :param manual_superposition: Add additional dimension to separate parameters for each magnet.
    :param num_magnets: The number of magnets in the dataset. Can also be determined automatically
    :param device: The device to use. If None, the tensor stays on CPU.
    :return: A tuple of Tensors (q_gt, xi_gt, u_gt)
    """
    
    if time_idx is not None:
        df = df[df['time_idx'] == time_idx].sort_values("sensor_id")

    num_sensors = df["sensor_id"].unique().shape[0]

    old_q_col_headers = ["q_dx", "q_dy", "q_dL"]
    if all(x in list(df.columns.values) for x in old_q_col_headers):
        q_gt = df[df["sensor_id"] == 0][old_q_col_headers].to_numpy().expand_dims(1)
    else:
        kinematic_parametrization, num_segments_ = identify_kinematic_parametrization_from_database(df)
        num_segments = num_segments_ if num_segments is None else num_segments

        if kinematic_parametrization == "cc":
            q_col_list = []
            for i in range(num_segments):
                q_col_list.extend([f"q_dx_{i}", f"q_dy_{i}", f"q_dL_{i}"])
            q_gt = df[df["sensor_id"] == 0][q_col_list].to_numpy().reshape(-1, num_segments, 3)
        elif kinematic_parametrization == "ac":
            q_col_list = []
            for i in range(num_segments):
                q_col_list.extend([f"q_kappa0_{i}", f"q_kappa1_{i}", f"q_phi_{i}", f"q_dL_{i}"])
            q_gt = df[df["sensor_id"] == 0][q_col_list].to_numpy().reshape(-1, num_segments, 4)
        else:
            raise ValueError(f"Unknown kinematic parametrization: {kinematic_parametrization}")

    old_xi_col_headers = ['lambda_j_k', 'd_j_k', 'alpha_j_k', 'beta_j_k', 'theta_j_k']
    if all(x in list(df.columns.values) for x in old_xi_col_headers):
        xi_gt = df[old_xi_col_headers].to_numpy().reshape(-1, num_sensors, len(old_xi_col_headers))
    else:
        if num_magnets is None:
            regex_col_matches = df.filter(regex='d_[0-9]').columns.values
            num_magnets = int(regex_col_matches[-1].split('_')[-1]) + 1
            # print(f"Recognized {num_magnets} magnets in the dataset.")

        if manual_superposition:
            xi_gt = None
            for k in range(num_magnets):
                var_list = ["lambda", f"d_{k}", f"alpha_{k}", f"beta_{k}", f"theta_{k}"]
                xi_gt_k = df[var_list].to_numpy().reshape(-1, num_sensors, len(var_list))

                if xi_gt is None:
                    xi_gt = np.zeros((xi_gt_k.shape[0], xi_gt_k.shape[1], num_magnets, xi_gt_k.shape[2]))
                xi_gt[:, :, k, :] = xi_gt_k
        else:
            var_list = ["lambda"]
            for k in range(num_magnets):
                var_list.extend([f"d_{k}", f"alpha_{k}", f"beta_{k}", f"theta_{k}"])

            xi_gt = df[var_list].to_numpy().reshape(-1, num_sensors, len(var_list))

    u_gt = df[['u']].to_numpy().reshape(-1, num_sensors)

    if not separate_sensors:
        xi_gt = xi_gt.reshape((-1,) + xi_gt.shape[2:])
        u_gt = u_gt.reshape(-1)

    if time_idx is not None:
        # remove the first dimension (time_idx)
        q_gt, xi_gt, u_gt = q_gt.squeeze(0), xi_gt.squeeze(0), u_gt.squeeze(0)

    q_gt = torch.tensor(q_gt, dtype=torch.float, device=device)
    xi_gt = torch.tensor(xi_gt, dtype=torch.float, device=device)
    u_gt = torch.tensor(u_gt, dtype=torch.float, device=device)

    return q_gt, xi_gt, u_gt


def tensors_to_inference_df(q_gt_ts: torch.Tensor, q_hat_ts: torch.Tensor,
                            u_gt_ts: torch.Tensor, u_hat_ts: torch.Tensor,
                            rmse_u_ts: torch.Tensor, kinematic_parametrization: str = "cc") -> pd.DataFrame:
    q_gt_ts, q_hat_ts = q_gt_ts.cpu().numpy(), q_hat_ts.cpu().numpy()
    u_hat_ts, rmse_u_ts = u_hat_ts.cpu().numpy(), rmse_u_ts.cpu().numpy()

    df_data = {}
    df_data["time_idx"] = np.arange(q_gt_ts.shape[0])

    if kinematic_parametrization == "cc":
        q_name_map = ["dx", "dy", "dL"]
    elif kinematic_parametrization == "ac":
        q_name_map = ["kappa0", "kappa1", "phi", "dL"]
    else:
        raise ValueError(f"Unknown kinematic parametrization: {kinematic_parametrization}")

    for i in range(q_gt_ts.shape[1]):
        for q_i_idx in range(q_gt_ts.shape[2]):
            df_data[f"q_gt_{q_name_map[q_i_idx]}_{i}"] = q_gt_ts[:, i, q_i_idx]
            df_data[f"q_hat_{q_name_map[q_i_idx]}_{i}"] = q_hat_ts[:, i, q_i_idx]

    for j in range(u_hat_ts.shape[1]):
        df_data[f"u_gt_{j}"] = u_gt_ts[:, j]
        df_data[f"u_hat_{j}"] = u_hat_ts[:, j]

    df_data["rmse_u"] = rmse_u_ts

    df = pd.DataFrame(df_data)

    return df


def inference_df_to_tensors(df: pd.DataFrame, num_segments: int = None, num_sensors: int = None,
                            device: torch.device = None) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    kinematic_parametrization, num_segments_ = identify_kinematic_parametrization_from_database(
        df,
        config_variable_interfix="_gt"
    )
    num_segments = num_segments_ if num_segments is None else num_segments

    q_gt_col_list = []
    q_hat_col_list = []
    for i in range(num_segments):
        if i == 0 and "q_gt_dx" in df.columns.values:
            q_gt_col_list.extend([f"q_gt_dx", f"q_gt_dy", f"q_gt_dL"])
            q_hat_col_list.extend([f"q_hat_dx", f"q_hat_dy", f"q_hat_dL"])
            break
        elif kinematic_parametrization == "cc":
            q_gt_col_list.extend([f"q_gt_dx_{i}", f"q_gt_dy_{i}", f"q_gt_dL_{i}"])
            q_hat_col_list.extend([f"q_hat_dx_{i}", f"q_hat_dy_{i}", f"q_hat_dL_{i}"])
        elif kinematic_parametrization == "ac":
            q_gt_col_list.extend([f"q_gt_kappa0_{i}", f"q_gt_kappa1_{i}", f"q_gt_phi_{i}", f"q_gt_dL_{i}"])
            q_hat_col_list.extend([f"q_hat_kappa0_{i}", f"q_hat_kappa1_{i}", f"q_hat_phi_{i}", f"q_hat_dL_{i}"])
        else:
            raise ValueError(f"Unknown kinematic parametrization: {kinematic_parametrization}")

    q_gt_ts = df[q_gt_col_list].to_numpy().reshape(df.shape[0], num_segments, -1)
    q_hat_ts = df[q_hat_col_list].to_numpy().reshape(df.shape[0], num_segments, -1)

    if num_sensors is None:
        regex_col_matches = df.filter(regex='u_gt_[0-9]').columns.values
        num_sensors = int(regex_col_matches[-1].split('_')[-1]) + 1
        print(f"Recognized {num_sensors} sensors in the dataset.")

    u_gt_col_list = []
    u_hat_col_list = []
    for j in range(num_sensors):
        u_gt_col_list.append(f"u_gt_{j}")
        u_hat_col_list.append(f"u_hat_{j}")
    u_gt_ts = df[u_gt_col_list].to_numpy().reshape(-1, num_sensors)
    u_hat_ts = df[u_hat_col_list].to_numpy().reshape(-1, num_sensors)

    if "rmse_u" in df.columns.values:
        rmse_u_ts = df["rmse_u"].to_numpy().reshape(-1)
    elif "RMSE_u" in df.columns.values:
        rmse_u_ts = df["RMSE_u"].to_numpy().reshape(-1)
    else:
        raise ValueError("Could not find RMSE_u in the dataset.")

    q_gt_ts = torch.tensor(q_gt_ts, dtype=torch.float, device=device)
    q_hat_ts = torch.tensor(q_hat_ts, dtype=torch.float, device=device)
    u_gt_ts = torch.tensor(u_gt_ts, dtype=torch.float, device=device)
    u_hat_ts = torch.tensor(u_hat_ts, dtype=torch.float, device=device)
    rmse_u_ts = torch.tensor(rmse_u_ts, dtype=torch.float, device=device)

    return q_gt_ts, q_hat_ts, u_gt_ts, u_hat_ts, rmse_u_ts
