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
    
    df_samples_seed_0 = pd.read_csv(f'datasets/inference/{dataset_name}_seed_0.csv')
    df_samples_seed_1 = pd.read_csv(f'datasets/inference/{dataset_name}_seed_1.csv')
    df_samples_seed_2 = pd.read_csv(f'datasets/inference/{dataset_name}_seed_2.csv')
    
  
    for segm in num_segments:
        samples = []
        Sens1=4*segm
        Sens2=4*segm+1
        Sens3=4*segm+2
        Sens4=4*segm+3
        
        for time_idx in df_samples_seed_0["time_idx"].unique():
           
            df_at_t_s0 = df_samples_seed_0[df_samples_seed_0["time_idx"] == time_idx]
            df_at_t_s1 = df_samples_seed_1[df_samples_seed_0["time_idx"] == time_idx]
            df_at_t_s2 = df_samples_seed_2[df_samples_seed_0["time_idx"] == time_idx]
            
            #values of different seeds
            u_hat_s0 = df_at_t_s0[[f"u_hat_{Sens1}", f"u_hat_{Sens2}", f"u_hat_{Sens3}", f"u_hat_{Sens4}"]].iloc[0].to_numpy()
            u_hat_s1 = df_at_t_s1[[f"u_hat_{Sens1}", f"u_hat_{Sens2}", f"u_hat_{Sens3}", f"u_hat_{Sens4}"]].iloc[0].to_numpy()
            u_hat_s2 = df_at_t_s2[[f"u_hat_{Sens1}", f"u_hat_{Sens2}", f"u_hat_{Sens3}", f"u_hat_{Sens4}"]].iloc[0].to_numpy()
            
            mean_u1 = np.mean(np.array([[u_hat_s0[0]],[u_hat_s1[0]],[u_hat_s2[0]]]))
            mean_u2 = np.mean(np.array([[u_hat_s0[1]],[u_hat_s1[1]],[u_hat_s2[1]]]))
            mean_u3 = np.mean(np.array([[u_hat_s0[2]],[u_hat_s1[2]],[u_hat_s2[2]]]))      
            mean_u4 = np.mean(np.array([[u_hat_s0[3]],[u_hat_s1[3]],[u_hat_s2[3]]]))      
                   
            std_u1 = np.std(np.array([[u_hat_s0[0]],[u_hat_s1[0]],[u_hat_s2[0]]]))
            std_u2 = np.std(np.array([[u_hat_s0[1]],[u_hat_s1[1]],[u_hat_s2[1]]]))
            std_u3 = np.std(np.array([[u_hat_s0[2]],[u_hat_s1[2]],[u_hat_s2[2]]]))      
            std_u4 = np.std(np.array([[u_hat_s0[3]],[u_hat_s1[3]],[u_hat_s2[3]]]))      
            
            sample = {}
            #sample["time_idx"] = time_idx
    
            sample[f"mean_dx_{segm}"] = mean_u1.item()
            sample[f"mean_dy_{segm}"] = mean_u2.item()
            sample[f"mean_dL_{segm}"] = mean_u3.item()
            sample[f"mean_dM_{segm}"] = mean_u4.item()
    
            sample[f"std_dx_{segm}"] = std_u1.item()
            sample[f"std_dy_{segm}"] = std_u2.item()
            sample[f"std_dL_{segm}"] = std_u3.item()
            sample[f"std_dM_{segm}"] = std_u4.item()
    
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
        
        
        Sens1=4*segm
        Sens2=4*segm+1
        Sens3=4*segm+2
        Sens4=4*segm+3
       
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
            
            elif d_i == "dM" and segm == 0: 
                gt_line_color = 'magenta'
                line_color = 'orchid'
                band_color = 'violet'   

            elif d_i == "dM" and segm == 1: 
                gt_line_color = 'purple'
                line_color = 'mediumorchid'
                band_color = 'plum'                   
 
            elif d_i == "dM" and segm == 2: 
                gt_line_color = 'indigo'
                line_color = 'rebeccapurple'
                band_color = 'mediumpurple'   
            
            if segm == 0 :
                if d_i == 'dx':
                    print(Sens1)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens1}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{1}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{1}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    print(Sens2)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens2}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{2}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{2}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    print(Sens3)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens3}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{3}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{3}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dM':
                    print(Sens4)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens4}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{4}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_0[f"mean_{d_i}_0"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{4}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_0[f"mean_{d_i}_0"]-df_samples_mean_std_0[f"std_{d_i}_0"], df_samples_mean_std_0[f"mean_{d_i}_0"]+df_samples_mean_std_0[f"std_{d_i}_0"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                                        
            if segm == 1 :
            
                if d_i == 'dx':
                    print(Sens1)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens1}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{5}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{5}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    print(Sens2)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens2}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{6}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{6}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    print(Sens3)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens3}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{7}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{7}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dM':
                    print(Sens4)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens4}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{8}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_1[f"mean_{d_i}_1"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{8}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_1[f"mean_{d_i}_1"]-df_samples_mean_std_1[f"std_{d_i}_1"], df_samples_mean_std_1[f"mean_{d_i}_1"]+df_samples_mean_std_1[f"std_{d_i}_1"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
           
            if segm == 2 :
                if d_i == 'dx':
                    print(Sens1)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens1}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{9}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{9}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dy':
                    print(Sens2)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens2}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{10}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{10}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dL':
                    print(Sens3)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens3}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{11}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{11}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
                if d_i == 'dM':
                    print(Sens4)
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_gt[f"u_gt_{Sens4}"], color= gt_line_color, linewidth=2, linestyle='-')#,  label='$u_{12}$')   
                    ax.plot(df_samples_gt["time_idx"]/sample_rate, df_samples_mean_std_2[f"mean_{d_i}_2"],color= line_color,linewidth=2,  linestyle='--', label='$\hat{u}_{12}$', dashes=(5, 4))
                    ax.fill_between(df_samples_gt["time_idx"]/sample_rate,df_samples_mean_std_2[f"mean_{d_i}_2"]-df_samples_mean_std_2[f"std_{d_i}_2"], df_samples_mean_std_2[f"mean_{d_i}_2"]+df_samples_mean_std_2[f"std_{d_i}_2"], alpha=0.4, edgecolor=band_color, facecolor=band_color)    
        
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  
    
    ax.set_xlabel('Time\:[s]', fontsize=fontsize_label_time)
    ax.set_ylabel("$u$ [mT]", fontsize=fontsize_label_di)

    ax.grid()    
    
    ax.legend(ncol=1, fontsize=11.5, loc='center left', bbox_to_anchor=(1, 0.5))  
    ax.set_xlim(0,len(df_samples_gt["time_idx"])/sample_rate)
    
    return fig

if __name__ == "__main__":
    plt.close("all")
    print("loading...")
    
    #if evaluated_db == True:   
    #    df_samples_end_to_end = plot_neural_network_predictions_end_to_end()
  
    num_segments = [i for i in range(num_segments)]
    df_samples_gt = pd.read_csv(f'datasets/inference/{dataset_name}_seed_0.csv')
    
    df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2 = mean_std_band()   
    
    fig_dx  =  plot_q_di_gt_and_hat(df_samples_gt, df_samples_mean_std_0,df_samples_mean_std_1,df_samples_mean_std_2, ['dx','dy','dL','dM'])

    plt.tight_layout()
    plt.show()
    pathlib.Path("plots").mkdir(exist_ok=True)
    plt.savefig(f"plots/{dataset_name}_size_{figsize}_u.pdf")
    print("done")
    