import pandas as pd
import pathlib
import torch
from matplotlib.ticker import FormatStrFormatter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
fontsize_axis = 20
fontsize_label_di = 20
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

dataset_name = 'analytical_db_n_b-3_n_s-9_n_m-3_T0_n_t-120000_rand_phi_off_rand_psi_s_rand_d_s_r' \
               '_to_' \
               'analytical_db_n_b-3_n_s-12_n_m-3_T3_n_t-400_inference_sensor_failure'

num_segments = 3
# init random seed
seed = 2

def mean_std_band():
    
    df_samples_seed_0 = pd.read_csv(f'inference_data/{dataset_name}_seed_0.csv')
    df_samples_seed_1 = pd.read_csv(f'inference_data/{dataset_name}_seed_1.csv')
    df_samples_seed_2 = pd.read_csv(f'inference_data/{dataset_name}_seed_2.csv')
    
  
    for segm in num_segments:
        samples = []

        for time_idx in df_samples_seed_0["time_idx"].unique():
           
            df_at_t_s0 = 1000*df_samples_seed_0[df_samples_seed_0["time_idx"] == time_idx]
            df_at_t_s1 = 1000*df_samples_seed_1[df_samples_seed_0["time_idx"] == time_idx]
            df_at_t_s2 = 1000*df_samples_seed_2[df_samples_seed_0["time_idx"] == time_idx]
            
            q_hat_s0 = df_at_t_s0[[f"q_hat_dx_{segm}", f"q_hat_dy_{segm}", f"q_hat_dL_{segm}"]].iloc[0].to_numpy()
            q_hat_s1 = df_at_t_s1[[f"q_hat_dx_{segm}", f"q_hat_dy_{segm}", f"q_hat_dL_{segm}"]].iloc[0].to_numpy()
            q_hat_s2 = df_at_t_s2[[f"q_hat_dx_{segm}", f"q_hat_dy_{segm}", f"q_hat_dL_{segm}"]].iloc[0].to_numpy()

            mean_dx = np.mean(np.array([[q_hat_s0[0]],[q_hat_s1[0]],[q_hat_s2[0]]]))
            mean_dy = np.mean(np.array([[q_hat_s0[1]],[q_hat_s1[1]],[q_hat_s2[1]]]))
            mean_dL = np.mean(np.array([[q_hat_s0[2]],[q_hat_s1[2]],[q_hat_s2[2]]]))      
                   
            std_dx = np.std(np.array([[q_hat_s0[0]],[q_hat_s1[0]],[q_hat_s2[0]]]))
            std_dy = np.std(np.array([[q_hat_s0[1]],[q_hat_s1[1]],[q_hat_s2[1]]]))
            std_dL = np.std(np.array([[q_hat_s0[2]],[q_hat_s1[2]],[q_hat_s2[2]]]))      
            
            sample = {}
            #sample["time_idx"] = time_idx
    
            sample[f"mean_dx_{segm}"] = mean_dx.item()
            sample[f"mean_dy_{segm}"] = mean_dy.item()
            sample[f"mean_dL_{segm}"] = mean_dL.item()
    
            sample[f"std_dx_{segm}"] = std_dx.item()
            sample[f"std_dy_{segm}"] = std_dy.item()
            sample[f"std_dL_{segm}"] = std_dL.item()
    
            samples.append(sample)
            
            if segm == 0:
               df_samples_mean_std_0 = pd.DataFrame(samples)           
            if segm == 1:
               df_samples_mean_std_1 = pd.DataFrame(samples)             
            if segm == 2:
               df_samples_mean_std_2 = pd.DataFrame(samples)  

    return df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2


def plot_q_di_gt_and_hat(df_samples_gt,  df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2, d_i_list):
    
    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(16)                      
    sample_rate = 40
    for segm in num_segments:
        for d_i in d_i_list:

            if d_i == "dx" and segm == 0:
                gt_line_color = 'dodgerblue'
                line_color = 'deepskyblue'
                band_color = 'cyan'
                             
            elif d_i == "dy" and segm == 0:
                gt_line_color = 'lightcoral'
                line_color = 'tomato'
                band_color = 'coral'              
            
            elif d_i == "dL" and segm == 0 :
                gt_line_color = 'forestgreen'
                line_color = 'limegreen'
                band_color = 'palegreen'              

            elif d_i == "dx" and segm == 1:
                gt_line_color = 'darkblue'
                line_color = 'mediumblue'
                band_color = 'steelblue'
             
            elif d_i == "dy" and segm == 1:
                gt_line_color = 'sienna'
                line_color = 'chocolate'
                band_color = 'sandybrown'               
            
            elif d_i == "dL" and segm == 1 :
                gt_line_color = 'seagreen'
                line_color = 'mediumseagreen'
                band_color = 'springgreen'  
                
            elif d_i == "dx" and segm == 2:
                gt_line_color = 'royalblue'
                line_color = 'cornflowerblue'
                band_color = 'lightskyblue'
           
                         
            elif d_i == "dy" and segm == 2:
                gt_line_color = 'maroon'
                line_color = 'firebrick'
                band_color = 'indianred'               
            
            elif d_i == "dL" and segm == 2: 
                gt_line_color = 'darkolivegreen'
                line_color = 'olivedrab'
                band_color = 'yellowgreen'                   
            
            
            if segm == 0 :
                if d_i == 'dx':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{x,0}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{x,1}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{y,0}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{y,1}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\delta_{L,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\delta}L_1$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                          
            if segm == 1 :
                if d_i == 'dx':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{x,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{x,2}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{y,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{y,2}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\delta_{L,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\delta}L_2$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
            
            if segm == 2 :
                if d_i == 'dx':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{x,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{x,3}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\Delta_{y,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\Delta}_{y,3}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, 1000*df_samples_gt[f"q_gt_{d_i}_{segm}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$\delta_{L,1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{\delta}L_3$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
   
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.f'))  
    
    ax.set_xlabel(r'Time\:[s]', fontsize=fontsize_label_time)
    ax.set_ylabel(r'$q$\:[mm]', fontsize=fontsize_label_di)

    ax.grid()    
    #ax.legend(('$\Delta_{x,1}$','$\hat{\Delta}_{x,1}$','$\Delta_{y,1}$','$\hat{\Delta}_{y,1}$','$\delta_{L,1}$','$\hat{\delta}_{L,1}$','$\Delta_{x,2}$','$\hat{\Delta}_{x,2}$','$\Delta_{y,2}$','$\hat{\Delta}_{y,2}$','$\delta_{L,2}$','$\hat{\delta}_{L,2}$','$\Delta_{x,3}$','$\hat{\Delta}_{x,3}$','$\Delta_{y,3}$','$\hat{\Delta}_{y,3}$','$\delta_{L,3}$','$\hat{\delta}_{L,3}$')
    #          , ncol=3, fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))    
    
    ax.legend(ncol=1, fontsize=12.55, loc='center left', bbox_to_anchor=(1, 0.5))  
    ax.set_xlim(0, len(df_samples_gt["time_idx"])/sample_rate)
    
    return fig

if __name__ == "__main__":
    plt.close("all")
    print("loading...")
    
    #if evaluated_db == True:   
    #    df_samples_end_to_end = plot_neural_network_predictions_end_to_end()
  
    num_segments = [i for i in range(num_segments)]
    df_samples_gt = pd.read_csv(f'inference_data/{dataset_name}_seed_0.csv')
    
    df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2 = mean_std_band()   
    
    fig_dx  =  plot_q_di_gt_and_hat(df_samples_gt, df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2, ['dx','dy','dL'])

    plt.tight_layout()
    plt.show()
    pathlib.Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/{dataset_name}_size_{figsize}_q.pdf")
    print("done")
    