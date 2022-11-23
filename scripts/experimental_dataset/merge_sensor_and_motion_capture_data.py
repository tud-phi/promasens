import numpy as np
import pandas as pd

from promasens.visualization import plt


plt.close("all")

dataset_name = '2022-05-02_T3_90deg_P0_R1'
# dataset_name = '2022-02-21_spiral2_480s_start_200mb_max_425mb_f0_20'
path_to_mcs_data = f'datasets/experimental/processed_motion_capture_data/{dataset_name}_q.csv'
path_to_sensor_data = f'datasets/experimental/sensor_data/{dataset_name}_SV.csv'

path_to_merged_data = f'datasets/experimental/merged_data/{dataset_name}.csv'
path_to_merged_data_train = f'datasets/experimental/merged_data/{dataset_name}_train.csv'
path_to_merged_data_test = f'datasets/experimental/merged_data/{dataset_name}_test.csv'

test_split = 0.3
if dataset_name in ['2022-02-21_spiral2_480s_start_200mb_max_425mb_f0_20',
                    "2022-05-02_FLOWER_FAST_200mBar_200s_P0_R1"]:
    test_split = 0.5

df_mcs = pd.read_csv(path_to_mcs_data)

num_sensors = len(df_mcs["sensor_id"].unique())
print(f"recognized {num_sensors} sensors in dataset")

sample_rate = 40  # Hz

# add time idx data
df_mcs.insert(loc=0, column="time_idx",
              value=np.repeat(np.arange(start=0, stop=len(df_mcs.index) // num_sensors, step=1), num_sensors))

df_sensor_data = pd.read_csv(path_to_sensor_data)
df_sensor_data.insert(loc=0, column="time_idx", value=np.arange(start=0, stop=len(df_sensor_data.index), step=1))

if dataset_name == '2022-04-21_T3_P0_R1':
    # the sensor measurement data from sensor u3 looks weird in particular between 3600 and 3800
    df_sensor_data = df_sensor_data.iloc[:3100, :]
elif dataset_name == '2022-04-21_SPIRAL_FORWARD_120s_P0_R1':
    # issues with sensor u3
    df_sensor_data = df_sensor_data.iloc[:1640, :]
elif dataset_name == '2022-04-21_SPIRAL_FORWARD_90s_P0_R1':
    # issues with sensor u3
    df_sensor_data = df_sensor_data.iloc[:3650, :]

# we try to find out when the segment first extends caused by the increase in pressure in the chambers
thresh_delta_L = 0.0002  # m
thresh_u = 3
# the new datasets with soldered wires have much less noise
if "2022-05-02" in dataset_name:
    thresh_u = 1
if dataset_name == '2022-04-21_SPIRAL_FORWARD_90s_P0_R1':
    # u3 is very noise at the beginning making it exceed the default threshold very quickly
    thresh_u = 6
elif dataset_name == '2022-05-02_FLOWER_FAST_NOMINAL_P0_R1':
    # the MCS data is very noisy making it exceed the default threshold very quickly
    thresh_delta_L = 0.0003


def plot_separate_datasets(df_mcs: pd.DataFrame, df_sensor_data: pd.DataFrame, q_postfix: str = '_1'):
    # plot configuration values over dataset
    df_mcs_sensor_1 = df_mcs[df_mcs["sensor_id"] == 1]
    t = df_mcs_sensor_1["time_idx"] / sample_rate

    fig, ax1 = plt.subplots(num='Unmerged dataset')
    ax1.plot(t, df_mcs_sensor_1[f"q_dx{q_postfix}"]*10**3, label="$\Delta_{x}$")
    ax1.plot(t, df_mcs_sensor_1[f"q_dy{q_postfix}"]*10**3, label="$\Delta_{y}$")
    ax1.plot(t, df_mcs_sensor_1[f"q_dL{q_postfix}"]*10**3, label="$\delta L$")
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel("$q$ [mm]")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for sensor_idx in range(num_sensors):
        ax2.plot(t, df_sensor_data[f"u{sensor_idx + 1}"], ':', label=f"$u_{sensor_idx + 1}$")
    ax2.set_ylabel("$u$ [mV]")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


def plot_merged_dataset(df: pd.DataFrame, q_postfix: str = '_1'):
    # plot configuration values over dataset
    df_sensor_1 = df[df["sensor_id"] == 1]
    u = df["u"].to_numpy().reshape(-1, num_sensors)
    t = df_sensor_1["time_idx"] / sample_rate

    fig, ax1 = plt.subplots(num='Merged dataset')
    ax1.plot(t, df_sensor_1[f"q_dx{q_postfix}"]*10**3, label="$\Delta_{x}$")
    ax1.plot(t, df_sensor_1[f"q_dy{q_postfix}"]*10**3, label="$\Delta_{y}$")
    ax1.plot(t, df_sensor_1[f"q_dL{q_postfix}"]*10**3, label="$\delta L$")
    ax1.set_xlabel("$t$ [s]")
    ax1.set_ylabel("$q$ [mm]")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    for sensor_idx in range(num_sensors):
        ax2.plot(t, u[:, sensor_idx], ':', label=f"$u_{sensor_idx + 1}$")
    ax2.set_ylabel("$u$ [mV]")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # make the code work for both old and new datasets with different header naming conventions
    q_postfix = '' if 'q_dL' in list(df_mcs.columns.values) else '_0'

    cond_delta_L = (df_mcs[f"q_dL{q_postfix}"] - df_mcs[f"q_dL{q_postfix}"][0]).abs() > thresh_delta_L
    row_idx_cond_delta_L_true = cond_delta_L.idxmax()
    time_idx_cond_delta_L_true = df_mcs['time_idx'][row_idx_cond_delta_L_true]

    print(f"Recognized delta L threshold of dL={thresh_delta_L}m at time_idx={time_idx_cond_delta_L_true}")

    # cut-off any rows with missing sensor values at the end
    u_header_list = []
    for sensor_idx in range(num_sensors):
        u_header_list.append(f"u{sensor_idx + 1}")
    sensor_rows_with_nan = df_sensor_data.isnull()[u_header_list].any(axis=1)
    if sensor_rows_with_nan.sum() > 0:
        df_sensor_data = df_sensor_data.iloc[:sensor_rows_with_nan.idxmax()]

    for sensor_idx in range(num_sensors):
        Delta_u_j = df_sensor_data[f'u{sensor_idx + 1}'] - df_sensor_data[f'u{sensor_idx + 1}'][0]
        if sensor_idx == 0:
            Delta_u_sum = Delta_u_j ** 2
        else:
            Delta_u_sum += Delta_u_j ** 2
    RMSE_Delta_u = (Delta_u_sum / num_sensors) ** (1 / 2)

    cond_u = RMSE_Delta_u > thresh_u
    row_idx_cond_u_true = cond_u.idxmax()
    time_idx_cond_u_true = df_sensor_data['time_idx'][row_idx_cond_u_true]

    print(f"Recognized u threshold of u={thresh_u} at time_idx={time_idx_cond_u_true}")

    # IMPORTANT: we assume that both MoCap and sensor data are synchronized and using the same sampling frequency
    if time_idx_cond_u_true > time_idx_cond_delta_L_true:
        # we need to cut-off some of the sensor data at the beginning to align both datasets in time
        print("We need to cut-off some of the sensor data at the beginning to align both datasets in time")
        df_sensor_data = df_sensor_data[
            df_sensor_data['time_idx'] > (time_idx_cond_u_true - time_idx_cond_delta_L_true - 0)]
        # re-seed the time idx
        df_sensor_data["time_idx"] = df_sensor_data["time_idx"] - (time_idx_cond_u_true - time_idx_cond_delta_L_true)
    elif time_idx_cond_u_true < time_idx_cond_delta_L_true:
        # we need to cut-off some of the mocap data at the beginning to align both datasets in time
        print("We need to cut-off some of the mocap data at the beginning to align both datasets in time")
        df_mcs = df_mcs[df_mcs['time_idx'] > (time_idx_cond_delta_L_true - time_idx_cond_u_true)]
        # re-seed the time idx
        df_mcs["time_idx"] = df_mcs["time_idx"] - (time_idx_cond_delta_L_true - time_idx_cond_u_true)
    else:
        # the datasets are already aligned in time, we do nothing
        print("The datasets are already aligned in time. We do not need to do anything.")
        pass

    # we make both datasets equally long and cut-off the end of the longer one
    min_dataset_length = np.min([len(df_mcs.index) // num_sensors, len(df_sensor_data.index)])
    df_mcs = df_mcs[:min_dataset_length * num_sensors]
    df_sensor_data = df_sensor_data[:min_dataset_length]

    # merge the two datasets
    df_merged = df_mcs.copy()
    list_of_u_columns = [f"u{i}" for i in range(1, num_sensors + 1)]
    df_merged["u"] = df_sensor_data[list_of_u_columns].to_numpy().flatten()

    if dataset_name == '2022-02-21_spiral2_480s_start_200mb_max_425mb_f0_20':
        # only use the middle 50% of the dataset
        data_length = len(df_merged.index)
        start_idx = int((0.25 * data_length) // num_sensors * num_sensors)
        stop_idx = int((0.75 * data_length) // num_sensors * num_sensors)
        df_merged = df_merged[start_idx:stop_idx].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]
    elif dataset_name == '2022-04-21_SPIRAL_REVERSE_90s_P0_R1':
        # issues with sensor u3
        df_merged = df_merged[df_merged["time_idx"] > 2080].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]
    elif dataset_name == '2022-04-21_T2_P0_R1':
        # issues with sensor u3
        df_merged = df_merged[df_merged["time_idx"] > 1190].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]
    elif dataset_name == '2022-04-21_SPIRAL_REVERSE_120s_P0_R1':
        # issues with sensor u3
        df_merged = df_merged[df_merged["time_idx"] > 610].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]
    elif dataset_name == '2022-04-21_STAR_NOMINAL_400s_P0_R1':
        # issues with sensor u3 between 2400 and 4800
        df_merged = df_merged[df_merged["time_idx"] > 4800].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]
    elif dataset_name == '2022-04-21_STAR_80_600s_P0_R1':
        df_merged = df_merged[(df_merged["time_idx"] > 2800) & (df_merged["time_idx"] < 9850)].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]

    if "2022-04-21" not in dataset_name:
        # cut-off the first 8s and last 5s of the dataset (inflation and deflation)
        inflation_duration = 8  # s
        deflation_duration = 5  # s
        if dataset_name == "2022-05-02_FLOWER_FAST_NOMINAL_P0_R1":
            deflation_duration = 8  # s
        flating_cond = (df_merged["time_idx"] >= inflation_duration*sample_rate) \
                       & (df_merged["time_idx"] < df_merged["time_idx"].max() - deflation_duration*sample_rate)
        df_merged = df_merged[flating_cond].reset_index(drop=True)
        df_merged["time_idx"] = df_merged["time_idx"] - df_merged["time_idx"][0]

    print("final merged dataset:")
    print(df_merged)

    print("Exporting dataset to:", path_to_merged_data)
    df_merged.to_csv(path_to_merged_data, index=False)

    time_idx_split = (df_merged["time_idx"].max() - df_merged["time_idx"].min()) * (1 - test_split)

    df_train = df_merged[df_merged["time_idx"] < time_idx_split].reset_index(drop=True)
    df_train["time_idx"] = df_train["time_idx"] - df_train["time_idx"].iloc[0]

    df_test = df_merged[df_merged["time_idx"] >= time_idx_split].reset_index(drop=True)
    df_test["time_idx"] = df_test["time_idx"] - df_test["time_idx"].iloc[0]

    df_train.to_csv(path_to_merged_data_train, index=False)
    df_test.to_csv(path_to_merged_data_test, index=False)

    plot_separate_datasets(df_mcs, df_sensor_data, q_postfix)
    plot_merged_dataset(df_merged, q_postfix)
