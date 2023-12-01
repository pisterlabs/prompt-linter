'''
@Author: Hitesh Kishore Das 
@Date: 2022-08-30 13:54:29 
'''

#%%

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import numpy as np
import cmasher as cr 
import sys
import os
import gc
import pickle as pk

style_lib  = '../plot/style_lib/' 
# pallette   = style_lib + 'cup_pallette.mplstyle'
pallette   = style_lib + 'dark_pallette.mplstyle'
# pallette   = style_lib + 'bright_pallette.mplstyle'
plot_style = style_lib + 'plot_style.mplstyle'
text_style = style_lib + 'text.mplstyle'

plt.style.use([pallette, plot_style, text_style])

cwd = os.path.dirname(__file__)
package_abs_path = cwd[:-len(cwd.split('/')[-1])]

sys.path.insert(0, f'{package_abs_path}data_analysis/')
import clump_analysis as ca
import coherence as ch

sys.path.insert(0, f'{package_abs_path}plot/')
import plot_3d as pt
import video as vid 

sys.path.insert(0, f'{package_abs_path}athena/')
import data_read as dr

sys.path.insert(0, f'{package_abs_path}utils/')
from timer import timer 
from units import KELVIN, mu

sys.path.insert(0, f'{package_abs_path}plot/')
import plot_3d as pt

#%%
N_procs_default  = 1
file_ind_default = 1

n_arg = len(sys.argv)
if n_arg<=2:
    print(f"Need two arguments. Only {n_arg} were provided.")
    print(f"N_procs set to default: {N_procs_default} processors...")
    print(f"file_int set to default: {file_ind_default}...")
    N_procs  = N_procs_default
    file_ind = file_ind_default
elif n_arg==3:
    N_procs  = int(sys.argv[1])
    file_ind = int(sys.argv[2])
    print(f"N_procs set  : {N_procs} processors...")
    print(f"file_int set : {file_ind}...")
else:
    print(f"Too many arguments provided...")
    print(f"N_procs set to default: {N_procs_default} processors..")
    print(f"file_int set to default: {file_ind_default}...")
    N_procs = N_procs_default
    file_ind = file_ind_default

# file_loc += 'para_scan_Rlsh5_1000_res0_256_rseed_1_M_0.5_chi_100_beta_100/'
# file_loc += 'para_scan_Rlsh5_1000_res0_256_rseed_1_M_0.5_chi_100_hydro/'
# file_loc += 'para_scan_Rlsh4_2500_res0_128_rseed_1_M_0.5_chi_100_hydro/'
# file_loc += 'Turb.out2.00600.athdf'

# data_dir = '/afs/mpa/home/hitesh/remote/freya/data/'
data_dir = '/ptmp/mpa/hitesh/data/'

# save_dir = './save_arr/anisotropy'
save_dir = '/ptmp/mpa/hitesh/MHD_multiphase_turbulence/analysis/save_arr/anisotropy'

sim_list  = []

sim_list += ['Rlsh_1000_res_256_M_0.5_hydro/']
sim_list += ['Rlsh_1000_res_256_M_0.5_beta_100/']

sim_list += ['Rlsh_250_res_256_M_0.5_hydro/']
sim_list += ['Rlsh_250_res_256_M_0.5_beta_100/']

sim_list += ['Rlsh_250_res_256_M_0.25_hydro/']
sim_list += ['Rlsh_250_res_256_M_0.25_beta_100/']

N_sim = len(sim_list)
file_loc_list = N_sim*[data_dir]

for i in range(N_sim):
    file_loc_list[i] += sim_list[i]


N_snap_start = 501
N_snap_end   = 680
# N_snap_end   = 520

frac_aniso_list = []
time_list = []

for N_snap in range(N_snap_start, N_snap_end+1):

    file_name =  file_loc_list[file_ind] + f'Turb.out2.{str(N_snap).zfill(5)}.athdf'

    # MHD_flag = True
    MHD_flag = False 

    out_dict = dr.get_array_athena(file_name, fields=["T"],MHD_flag=MHD_flag)


    T = out_dict['T']
    # T_cut = 4e5
    T_min = 4e4
    T_cut = 2*T_min

    time = out_dict['time']
    # prs = out_dict['P']
    # T = (prs/rho) * KELVIN * mu

    wnd = 2.5*3 #int((np.shape(T))[0]/10)+0.5
    sigma_wnd_mul = 3.0

    devices = N_procs*['cpu']
    parallel_flag = True

    frac_aniso_arr = ch.fractional_anisotropy(inp_arr=T, sigma=wnd/sigma_wnd_mul,window=wnd, parallel_flag=parallel_flag, devices=devices)
    frac_aniso_t = np.average(frac_aniso_arr[T<T_cut])

    frac_aniso_list.append(frac_aniso_t)
    time_list.append(time)


save_dict={}
save_dict['coherence'] = frac_aniso_list
save_dict['time' ]     = time_list

save_loc = f'{save_dir}/{sim_list[file_ind]}'

with open(f'{save_loc}/anisotropy_time_Tcut_2Tfloor_filamentariness.pkl', 'wb') as f:
    pk.dump(save_dict, f)

del(save_dict)
gc.collect()
