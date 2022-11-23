import pandas as pd

from promasens.utils.df_to_tensor_utils import database_df_to_tensors
from promasens.visualization.plot_configuration_estimates import plot_cc_configuration_estimates
from promasens.visualization.plot_sensor_predictions import plot_sensor_predictions
from promasens.visualization.plot_dataset import plot_dataset

dataset_name = f"analytical_db_n_b-1_n_s-3_n_m-1_T3_n_t-400"
# dataset that is used
df = pd.read_csv(f'analytical_databases/{dataset_name}.csv').dropna(axis=0)
num_sensors = len(df["sensor_id"].unique())

sample_rate = 40

if __name__ == "__main__":
    q_gt_ts, xi_gt_ts, u_gt_ts = database_df_to_tensors(df)
    plot_sensor_predictions(sample_rate, u_gt_ts.cpu().numpy(), u_hat_ts=None)
    plot_cc_configuration_estimates(sample_rate, q_gt_ts.cpu().numpy(), q_hat_ts=None)
    plot_dataset(sample_rate, q_gt_ts.cpu().numpy(), u_gt_ts.cpu().numpy())
