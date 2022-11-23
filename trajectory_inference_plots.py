import pandas as pd
import torch
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive

interactive(True)
fontsize_axis = 20
fontsize_label_di = 25
fontsize_label_time=20

figsize = (16.0,4.0)

# latex text
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size":fontsize_axis,
    })

device ='cpu'
dtype = torch.float32

##############################################################################
"""Define all variables for plotting loss landscape"""
#define neural network architecture
#from end_to_end.neural_networks.arch_R_plot import NetNiet_E2E
#from src.modules.neural_networks.arch_N import NetNiet

"set evaluated_db to True if it is not evaluated before" 
evaluated_db = False

#dataset_name = '2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1_to_2022-05-02_T2_P0_R1'
dataset_name = '2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1_to_2022-05-02_FLOWER_SLOW_NOMINAL_P0_R1'

# init random seed
seed = 2

#dataset that is used
#df = pd.read_csv(f'Merged_databases/{dataset_name}_test.csv')

#num_sensors = len(df["sensor_id"].unique())

def mean_std_band():
    
    df_samples_seed_0 = pd.read_csv(f'inference_data/{dataset_name}_inference_optim_q_dx_q_dy_seed_0.csv')
    df_samples_seed_1 = pd.read_csv(f'inference_data/{dataset_name}_inference_optim_q_dx_q_dy_seed_1.csv')
    df_samples_seed_2 = pd.read_csv(f'inference_data/{dataset_name}_inference_optim_q_dx_q_dy_seed_2.csv')
    
    samples = []
    
    for time_idx in df_samples_seed_0["time_idx"].unique():
       
        df_at_t_s0 = df_samples_seed_0[df_samples_seed_0["time_idx"] == time_idx]
        df_at_t_s1 = df_samples_seed_1[df_samples_seed_0["time_idx"] == time_idx]
        df_at_t_s2 = df_samples_seed_2[df_samples_seed_0["time_idx"] == time_idx]
        
        q_hat_s0 = df_at_t_s0[["q_hat_dx", "q_hat_dy", "q_hat_dL"]].iloc[0].to_numpy()
        q_hat_s1 = df_at_t_s1[["q_hat_dx", "q_hat_dy", "q_hat_dL"]].iloc[0].to_numpy()
        q_hat_s2 = df_at_t_s2[["q_hat_dx", "q_hat_dy", "q_hat_dL"]].iloc[0].to_numpy()
        
        mean_dx = np.mean(np.array([[q_hat_s0[0]],[q_hat_s1[0]],[q_hat_s2[0]]]))
        mean_dy = np.mean(np.array([[q_hat_s0[1]],[q_hat_s1[1]],[q_hat_s2[1]]]))
        mean_dL = np.mean(np.array([[q_hat_s0[2]],[q_hat_s1[2]],[q_hat_s2[2]]]))      
               
        std_dx = np.std(np.array([[q_hat_s0[0]],[q_hat_s1[0]],[q_hat_s2[0]]]))
        std_dy = np.std(np.array([[q_hat_s0[1]],[q_hat_s1[1]],[q_hat_s2[1]]]))
        std_dL = np.std(np.array([[q_hat_s0[2]],[q_hat_s1[2]],[q_hat_s2[2]]]))      
        
        sample = {}
        sample["time_idx"] = time_idx

        sample["mean_dx"] = mean_dx.item()
        sample["mean_dy"] = mean_dy.item()
        sample["mean_dL"] = mean_dL.item()

        sample["std_dx"] = std_dx.item()
        sample["std_dy"] = std_dy.item()
        sample["std_dL"] = std_dL.item()

        samples.append(sample)
    
    df_samples_mean_std = pd.DataFrame(samples)    
    
    return df_samples_mean_std

###########################################END TO END######################################################

# =============================================================================
# def load_model_end_to_end(dataset_name: str):
#     #trained neural network that is used for 3 NN for 3 sensors
#     statedict_path = f"statedicts/End_to_end_state_dict_model_db_{dataset_name}_seed_{seed}.pt"
# 
#     model = NetNiet_E2E().to(device)
#     model.load_state_dict(torch.load(statedict_path))
#     model.eval()
# 
#     return model
# 
# def predict_sensor_measurements_end_to_end(u_hat: [np.array, torch.Tensor]) -> torch.Tensor:
#  
#     model_end_to_end = load_model_end_to_end(dataset_name)
# 
#     if type(u_hat) is not torch.Tensor:
#         u_hat = torch.tensor(u_hat, dtype=dtype, device=device)
#     
#     u_hat=torch.unsqueeze(u_hat,dim=0)
#     q_hat = model_end_to_end(u_hat) 
#     
#     return q_hat
#     
# def plot_neural_network_predictions_end_to_end():
#     samples = []
#     
#     for time_idx in df["time_idx"].unique():
#         
#         df_at_t = df[df["time_idx"] == time_idx]
#         q_gt = df_at_t[["q_dx", "q_dy", "q_dL"]].iloc[0].to_numpy()
#         
#         assert q_gt.shape[0] == num_sensors
#         
#         u_gt = df_at_t["u"].to_numpy()
# 
#         q_prediction = predict_sensor_measurements_end_to_end(u_gt)
#         
#         q_hat=q_prediction.detach().squeeze()
#         
#         error_q = q_hat - q_gt
#         rmse = torch.sqrt(torch.mean(error_q**2))
#         
#         sample = {}
#         sample["q_gt_dx"] = q_gt[0].item()
#         sample["q_gt_dy"] = q_gt[1].item()
#         sample["q_gt_dL"] = q_gt[2].item()
# 
#         sample["time_idx"] = time_idx
# 
#         sample["q_hat_dx"] = q_hat[0].item()
#         sample["q_hat_dy"] = q_hat[1].item()
#         sample["q_hat_dL"] = q_hat[2].item()
# 
#         sample["error_q_dx"] = error_q[0].item()
#         sample["error_q_dy"] = error_q[1].item()
#         sample["error_q_dL"] = error_q[2].item()
# 
#         sample["RMSE_u"] = rmse.item()
# 
#         samples.append(sample)
# 
#     df_samples_end_to_end = pd.DataFrame(samples)
#     df_samples_end_to_end.to_csv(f'end_to_end/plot/{dataset_name}_samples_end_to_end_plot.csv')  
# 
#     return df_samples_end_to_end
# =============================================================================

def plot_q_di_gt_and_hat(df_samples_gt,  df_samples_mean_std, d_i_list):
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                           gridspec_kw={
                           'width_ratios': [4, 1],
                           'wspace': 0.25,
                           'hspace': 0.8}) 
    fig.set_figheight(4)
    fig.set_figwidth(20)
    sample_rate = 40
    
    for d_i in d_i_list:
                
        if d_i == "dx":
            gt_line_color = 'royalblue'
            line_color = 'cornflowerblue'
            band_color = 'lightskyblue'
         
        if d_i == "dy":
            gt_line_color = 'lightcoral'
            line_color = 'tomato'
            band_color = 'coral'              
            
        ax[0].plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"q_gt_{d_i}"],gt_line_color, linewidth=2, linestyle='-',  label=f"q_gt_{d_i}")   
        ax[0].plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std[f"mean_{d_i}"],line_color,linewidth=2,  linestyle='--', label=f"q_hat_{d_i}_GD", dashes=(5, 4))
        
        ax[1].plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"q_gt_{d_i}"],gt_line_color, linewidth=2, linestyle='-',  label=f"q_gt_{d_i}")   
        ax[1].plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std[f"mean_{d_i}"],line_color,linewidth=2,  linestyle='--', label=f"q_hat_{d_i}_GD", dashes=(5, 4))
        
        ax[0].fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std[f"mean_{d_i}"]-df_samples_mean_std[f"std_{d_i}"], df_samples_mean_std[f"mean_{d_i}"]+df_samples_mean_std[f"std_{d_i}"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
        ax[1].fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std[f"mean_{d_i}"]-df_samples_mean_std[f"std_{d_i}"], df_samples_mean_std[f"mean_{d_i}"]+df_samples_mean_std[f"std_{d_i}"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
    
    ax[1].set_xticks(np.linspace(18,22,3))

    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))  
    
    ax[0].set_xlabel('Time\:[s]', fontsize=fontsize_label_time)
    ax[0].set_ylabel('$q$\:[m]', fontsize=fontsize_label_di)
    ax[1].set_xlabel('Time\:[s]', fontsize=fontsize_label_time)
    ax[1].set_ylabel('$q$\:[m]', fontsize=fontsize_label_di) 

    ax[0].grid()
    ax[1].grid()
    ax[0].legend(('$\Delta_x$','$\hat{\Delta}_x$','$\Delta_y$','$\hat{\Delta}_y$'), fontsize=15, loc='upper right')

    ax[0].set_xlim(0,len(df_samples_gt["time_idx"])/sample_rate)
    ax[1].set_xlim(18,22)
        
    return fig

if __name__ == "__main__":
    plt.close("all")
    print("loading...")
    
    #if evaluated_db == True:   
    #    df_samples_end_to_end = plot_neural_network_predictions_end_to_end()
  
    df_samples_mean_std = mean_std_band()
    df_samples_gt = pd.read_csv(f'inference_data/{dataset_name}_inference_optim_q_dx_q_dy_seed_0.csv')
    
    #df_samples_end_to_end = pd.read_csv(f'end_to_end/plot/{dataset_name}_samples_end_to_end_plot.csv')

    fig_dx =  plot_q_di_gt_and_hat(df_samples_gt, df_samples_mean_std, ['dx','dy'])
    #fig_dy =  plot_q_di_gt_and_hat(df_samples_end_to_end, df_samples_mean_std, 'dy')
    #fig_dL =  plot_q_di_gt_and_hat(df_samples_end_to_end, df_samples_mean_std, 'dL')
    
    plt.subplots_adjust(left=None, bottom=0.18, right=None, top=None)
     
    plt.show()
    plt.savefig(f"plots/{dataset_name}_size_{figsize}.pdf")
    print("done")
    