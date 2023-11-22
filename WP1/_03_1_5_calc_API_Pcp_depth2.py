'''
@author: Abbas-Uni-Stuttgart

September, 2023

1:37:50 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import tqdm
import seaborn as sn
from depth_funcs_new import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

from depth_funcs_new import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

# from depth_funcs_new import depth_ftn_mp

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

def nse(predictions, targets):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))

def main():
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    out_save_dir = data_path / r"Results\07_pcp_API"
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    
    n_vecs = int(1e4)
    n_cpus = 7
    
    alpha = 0.9
    time_steps_shift = [1, 2, 3, 4]
    
    path_ids_abv_150 = pd.read_csv(
        r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\IDs_of_Catch_A_abv_150km2.csv", sep=',', index_col=0)
    
    
    path_to_data_q = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'discharge_spec')
    path_to_data_pcp = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'precipitation')

    #===========================================================================
    
    #===========================================================================
    data_hdf5 = HDF5(infile=path_to_data_q)
    catch_ids = data_hdf5.get_all_names()
    
    data_hdf5_pcp = HDF5(infile=path_to_data_pcp)
    
    catch_coords = data_hdf5.get_coordinates(ids=catch_ids)

    df_coords = pd.DataFrame(index=catch_ids,
        data=catch_coords['easting'], columns=['X'])
    y_dwd_coords = catch_coords['northing']
    df_coords.loc[:, 'Y'] = y_dwd_coords
    
    path_LSTM = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM/LSTM_discharge.csv",
                            index_col=0, parse_dates=True)
    
    path_LSTM_shm = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM_SHM/LSTM_SHM_discharge.csv",
                            index_col=0, parse_dates=True)
    
    usph_vecs = gen_usph_vecs_mp(n_vecs, 1, n_cpus)
    
    usph_vecs_dQ = gen_usph_vecs_mp(n_vecs, 4, n_cpus)
    
    catch_ids_to_use = path_ids_abv_150.index.astype(str).intersection(catch_ids)
    for catch_id in tqdm.tqdm(path_LSTM.columns):
        # if catch_id == '14002':
        # print(catch_id)
        # break
        
        if not os.path.exists(os.path.join(out_save_dir, r"tstep_api_%s_heatmap.png" % (catch_id))):
                
            df_stn = data_hdf5.get_pandas_dataframe(catch_id)
            df_stn = df_stn.dropna(how='all')
            
            df_stn_pcp = data_hdf5_pcp.get_pandas_dataframe(catch_id)
            df_stn_pcp = df_stn_pcp.dropna(how='all')
            
            df_lstm = path_LSTM.loc[:, catch_id]
            df_lstm_shm = path_LSTM_shm.loc[:, catch_id]
            
            cmn_idx_all = df_lstm.index.intersection(df_lstm_shm.index.intersection(
                    df_stn.index))
                
            df_lstm = path_LSTM.loc[cmn_idx_all, catch_id]
            df_lstm_shm = path_LSTM_shm.loc[cmn_idx_all, catch_id]
            df_stn = df_stn.loc[cmn_idx_all, :]
            df_stn_pcp = df_stn_pcp.loc[cmn_idx_all,:]
            
            df_API = pd.DataFrame(index=df_stn_pcp.index, columns=['API'])
    
            for ii, _t in tqdm.tqdm(enumerate(df_API.index)):
                if ii == 0:
                    # ii=1
                    api_t = df_stn_pcp.iloc[ii]
                    df_API.iloc[ii] = api_t.values
                elif ii > 0:
                    api_t = df_API.iloc[ii-1].values
                    api_t1 = api_t*alpha + df_stn_pcp.iloc[ii].values
                    df_API.iloc[ii] = api_t1
                    
            # df_API.plot()
            cmn_idx_q_pcp = df_stn_pcp.index.intersection(df_stn.index)
            df_orig_shift = pd.DataFrame(index=df_API.index, columns=time_steps_shift)
            
            
            
            # df_stn_t = df_stn.loc[df_stn.index[::2],:]
            # df_stn_tp1 = df_stn.loc[df_stn.index.difference(df_stn_t.index),:]
            
            # cmn_idx_length = min(df_stn_t.index.shape[0], df_stn_tp1.index.shape[0])
            
            # df_in_vals_t = df_stn_t.iloc[:cmn_idx_length].values.astype(float).copy('c')
            # df_in_vals_tp1 = df_stn_tp1.iloc[:cmn_idx_length].values.astype(float).copy('c')
            # depths_da_da = depth_ftn_mp(df_in_vals_t, df_in_vals_tp1, usph_vecs, n_cpus)
            
            # depths_da_da2 = depth_ftn_mp(df_in_vals_tp1, df_in_vals_t, usph_vecs, n_cpus)
            #=======================================================================
            # 
            #=======================================================================
            
    
            for tshift in time_steps_shift:
                df_stn_norm_orig_shifted = df_API.shift(tshift)
                df_orig_shift.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
    
            df_orig_shift_nonan = df_orig_shift.dropna(how='any', axis=0)
            
            df_orig_shift_nonan_d = df_orig_shift_nonan.values.astype(float).copy('c')
            depths_da_Q = depth_ftn_mp(df_orig_shift_nonan_d, df_orig_shift_nonan_d, usph_vecs_dQ, n_cpus)
            
            df_orig_shift_nonan.loc[:, 'd'] = depths_da_Q
            depths_da_da_low = df_orig_shift_nonan.d[(1 <= df_orig_shift_nonan.d) & (df_orig_shift_nonan.d <=4)]
            
            dq_m = depths_da_da_low.index - pd.Timedelta(days=2)
            dq_p = depths_da_da_low.index + pd.Timedelta(days=2)
            
            datetime_idx = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m, dq_p)]
            merged_all_tdx = list(itertools.chain(*datetime_idx))
            
            # df_stn_pcp_low_q.plot()
            # plt.show()
            
            #=======================================================================
            # 
            #=======================================================================
            
            nse_lstm = nse(df_lstm.values.ravel(), df_stn.values.ravel())
            nse_lstm_shm = nse(df_lstm_shm.values.ravel(), df_stn.values.ravel())
            
                
            cmn_idx_q_pcp = df_stn_pcp.index.intersection(df_stn.index)
            df_orig_shift_q = pd.DataFrame(index=df_stn.index, columns=time_steps_shift)
            df_lstm_shift = pd.DataFrame(index=df_lstm.index, columns=time_steps_shift)
            df_lstm_shm_shift = pd.DataFrame(index=df_lstm_shm.index, columns=time_steps_shift)
    
            for tshift in time_steps_shift:
                df_stn_norm_orig_shifted = df_stn.shift(tshift)
                df_lstm_norm_shifted = df_lstm.shift(tshift)
                df_lstm_shm_norm_shift = df_lstm_shm.shift(tshift)
                
                df_orig_shift_q.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
                df_lstm_shift.loc[df_lstm_norm_shifted.index, tshift] = df_lstm_norm_shifted.values.ravel()
                df_lstm_shm_shift.loc[df_lstm_shm_norm_shift.index, tshift] = df_lstm_shm_norm_shift.values.ravel()
            
            df_orig_shift_q_nonan = df_orig_shift_q.dropna(axis=0)          
            df_lstm_shift_nonan = df_lstm_shift.dropna(axis=0)
            df_lstm_shm_shift_nonan = df_lstm_shm_shift.dropna(axis=0)
            
            df_orig_shift_nonan_d_q = df_orig_shift_q_nonan.values.astype(float).copy('c')
            df_lstm_shift_nonan_d = df_lstm_shift_nonan.values.astype(float).copy('c')
            df_lstm_shm_shift_nonan_d = df_lstm_shm_shift_nonan.values.astype(float).copy('c')
            
            depths_da_Q = depth_ftn_mp(df_orig_shift_nonan_d_q, df_orig_shift_nonan_d_q, usph_vecs_dQ, n_cpus)
            depths_da_Q_lstm = depth_ftn_mp(df_lstm_shift_nonan_d, df_lstm_shift_nonan_d, usph_vecs_dQ, n_cpus)
            
            depths_da_Q_lstm_shm = depth_ftn_mp(df_lstm_shm_shift_nonan_d, df_lstm_shm_shift_nonan_d, usph_vecs_dQ, n_cpus)
            
            #=======================================================================
            # 
            #=======================================================================
            df_orig_shift_q_nonan.loc[:, 'd'] = depths_da_Q
            # depths_da_da_low = df_orig_shift_q_nonan.d[(1 <= df_orig_shift_q_nonan.d) & (df_orig_shift_q_nonan.d <=4)]
            
            
            df_lstm_shift_nonan.loc[:, 'd'] = depths_da_Q_lstm
            # depths_da_da_low_lstm = df_lstm_shift_nonan.d[(1 <= df_lstm_shift_nonan.d) & (df_lstm_shift_nonan.d <=4)]
            
            df_lstm_shm_shift_nonan.loc[:, 'd'] = depths_da_Q_lstm_shm
            # depths_da_da_low_lstm_shm = df_lstm_shm_shift_nonan.d[(1 <= df_lstm_shm_shift_nonan.d) & (df_lstm_shm_shift_nonan.d <=4)]
            
            
            # dq_m = depths_da_da_low.index - pd.Timedelta(days=2)
            # dq_p = depths_da_da_low.index + pd.Timedelta(days=2)
            
            # dq_m_lstm = depths_da_da_low_lstm.index - pd.Timedelta(days=2)
            # dq_p_lstm = depths_da_da_low_lstm.index + pd.Timedelta(days=2)
            #
            # dq_m_lstm_shm = depths_da_da_low_lstm_shm.index - pd.Timedelta(days=4)
            # dq_p_lstm_shm = depths_da_da_low_lstm_shm.index + pd.Timedelta(days=4)
            #
    
            # datetime_idx_q = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m, dq_p)]
            # merged_ix_q = list(itertools.chain(*datetime_idx_q))
            # # df_stn_low_d = df_stn.loc[merged_ix_q]
            #
            # datetime_idx_lstm = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m_lstm, dq_p_lstm)]
            # merged_ix_q_lstm = list(itertools.chain(*datetime_idx_lstm))
            # # df_lstm_low_d = df_lstm.loc[merged_ix_q_lstm]
            #
            # datetime_idx_lstm_shm = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m_lstm_shm, dq_p_lstm_shm)]
            # merged_ix_q_lstm_shm = list(itertools.chain(*datetime_idx_lstm_shm))
            # df_lstm_shm_low_d = df_lstm_shm.loc[merged_ix_q_lstm_shm]
            
            df_stn_pcp_low_d = df_stn_pcp.loc[df_stn_pcp.index.intersection(
                merged_all_tdx), :]
            
            # q data 
            df_q_obsv_low_d = df_stn.loc[df_stn_pcp_low_d.index,:]
            df_q_lstm_low_d = df_lstm.loc[df_stn_pcp_low_d.index]
            df_q_lstm_shm_low_d = df_lstm_shm.loc[df_stn_pcp_low_d.index]
            
            # q d for low d p
            df_api_obsv_low_d = df_orig_shift_nonan.loc[df_orig_shift_nonan.index.intersection(
                merged_all_tdx),'d']
            df_d_obsv_low_d = df_orig_shift_q_nonan.loc[df_api_obsv_low_d.index,'d']
            df_d_lstm_low_d = df_lstm_shift_nonan.loc[df_api_obsv_low_d.index, 'd']
            df_d_lstm_shm_low_d = df_lstm_shm_shift_nonan.loc[df_api_obsv_low_d.index, 'd']
            
            # df_stn_pcp_low_d
            
            df_stn_api_low_d = df_API.loc[depths_da_da_low.index, :]
            # df_stn_pcp_low_d = df_stn_pcp.loc[merged_all_tdx, :]
            
            plt.ioff()
            plt.figure(figsize=(8, 6), dpi=300)
            plt.plot(df_API.index,df_API.values, label='API', c='gray') 
            plt.plot(df_stn_pcp.index, df_stn_pcp.values.ravel(), linewidth=1., label='Pcp')
    
            plt.scatter(df_stn_api_low_d.index, df_stn_api_low_d.values, facecolor='r', edgecolor='darkred', label='1<=D<=4', marker='X', s=50)
            
            
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.legend()
            plt.ylabel('mm/d')
            plt.savefig(os.path.join(out_save_dir, r"P_api_%s.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            #merged_all_tdx = list(itertools.chain(*datetime_idx))
            plt.ioff()
            plt.figure(figsize=(8, 6), dpi=300)
            plt.bar(df_stn_pcp.loc[df_stn_pcp.index.intersection(
                merged_all_tdx), :].index, height=df_stn_pcp.loc[df_stn_pcp.index.intersection(
                merged_all_tdx), :].values.ravel(), width=20, facecolor='r', edgecolor='r', label='Pcp-D')
    
            plt.plot(df_stn_pcp.index, df_stn_pcp.values.ravel(), linewidth=1., label='Pcp', alpha=0.5)
            plt.grid(alpha=0.5)
            plt.tight_layout()
            plt.legend()
            plt.ylabel('mm/d')
            plt.savefig(os.path.join(out_save_dir, r"P_api_%s_low_d.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            
            #=======================================================================
            # 
            #=======================================================================
            plt.ioff()
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
            ax1.scatter(df_stn_pcp_low_d.values, df_q_obsv_low_d.values, label='Obsv Q', facecolor='r', edgecolor='darkred', marker='D', alpha=0.75)
            ax1.scatter(df_stn_pcp_low_d.values, df_q_lstm_low_d.values, label='LSTM Q', facecolor='b', edgecolor='darkblue', marker='X', alpha=0.25)
            ax1.scatter(df_stn_pcp_low_d.values, df_q_lstm_shm_low_d.values, label='LSTM-SHM Q', marker='o', facecolor='g', edgecolor='darkgreen', alpha=0.25)
            ax1.grid(alpha=0.75)
            
            ax1.set_xlabel('P [mm/d]')
            ax1.set_ylabel('Q [m/s]')
            plt.legend(loc=0)
            plt.savefig(os.path.join(out_save_dir, r"tstep_api_%s_Q_P.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            
            #=======================================================================
            # 
            #=======================================================================
            plt.ioff()
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
            ax1.scatter(df_api_obsv_low_d.values, df_api_obsv_low_d.values, label='Obsv API', facecolor='k', edgecolor='gray', marker=',', alpha=0.75)
            
            ax1.scatter(df_api_obsv_low_d.values, df_d_obsv_low_d.values, label='Obsv Q', facecolor='r', edgecolor='darkred', marker='D', alpha=0.75)
            ax1.scatter(df_api_obsv_low_d.values, df_d_lstm_low_d.values, label='LSTM Q', facecolor='b', edgecolor='darkblue', marker='X', alpha=0.25)
            ax1.scatter(df_api_obsv_low_d.values, df_d_lstm_shm_low_d.values, label='LSTM-SHM Q', marker='o', facecolor='g', edgecolor='darkgreen', alpha=0.25)
            ax1.grid(alpha=0.75)
            
            ax1.set_xlabel('Depth API (+- 2 days)')
            ax1.set_ylabel('Depth Q')
            plt.xlim([0, 15])
            plt.ylim([0, 15])
            plt.legend(loc=0)
            plt.savefig(os.path.join(out_save_dir, r"tstep_api_%s_Q_P_D.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            # '4 < D <=10',
            df_rlt = pd.DataFrame(index=['1 <= D <=4',  '10 < D'],
                                  columns=['Obsv', 'LSTM', 'LSTM-SHM'], data=1.)
            api_b4, api_4_10, api_10 = [], [], []
            o_b4, o_4_10, o_10 = [], [], []
            ls_b4, ls_4_10, ls_10 = [], [], []
            ls_sh_b4, ls_sh_4_10, ls_sh_10 = [], [], []
            for ii, _d in enumerate(depths_da_da_low.values):
                df_api_obsv_low_d_ix = pd.Series(df_api_obsv_low_d).index[ii]
                # break
                ix_str = df_api_obsv_low_d_ix - pd.Timedelta(days=2)
                ix_end = df_api_obsv_low_d_ix + pd.Timedelta(days=2)
                
                obsv_d = df_orig_shift_q_nonan.loc[ix_str:ix_end, 'd'].min()
                lstm_d = df_lstm_shift_nonan.loc[ix_str:ix_end, 'd'].min()
                lstm_shm_d = df_lstm_shm_shift_nonan.loc[ix_str:ix_end, 'd'].min()
                
                if _d <=4:
                    print(_d, obsv_d, lstm_d, lstm_shm_d)
                    # break
                    api_b4.append(_d)
                    if obsv_d <=12:
                        o_b4.append(obsv_d)
                    # elif 4 <obsv_d and obsv_d <= 10:
                        # o_4_10.append(obsv_d)
                    else:
                        o_10.append(obsv_d)
                    ###
                    if lstm_d <=12:
                        # break
                        ls_b4.append(lstm_d)
                    # elif 4 < lstm_d  and lstm_d <= 10:
                        # ls_4_10.append(lstm_d)
                    else:
                        ls_10.append(lstm_d)    
                    
                    if lstm_shm_d <=12:
                        ls_sh_b4.append(lstm_shm_d)
                    # elif 4 < lstm_shm_d  and lstm_shm_d <= 10:
                        # ls_sh_4_10.append(lstm_shm_d)
                    else:
                        ls_sh_10.append(lstm_shm_d) 
            df_rlt.loc['1 <= D <=10',:] = [len(o_b4)/len(api_b4),
                   len(ls_b4)/len(api_b4), len(ls_sh_b4)/len(api_b4)]
            
            # df_rlt.loc['4 < D <=10',:] = [len(o_4_10)/len(api_b4),
                   # len(ls_4_10)/len(api_b4), len(ls_sh_4_10)/len(api_b4)]
            
            df_rlt.loc['10 < D',:] = [len(o_10)/len(api_b4),
                   len(ls_10)/len(api_b4), len(ls_sh_10)/len(api_b4)]
            
            # plt.matshow()
            # plt.colorbar(shrink=0.7)
            plt.ioff()
            plt.figure(figsize=(4,3), dpi=300)
            plt.title('n=%d - NSE LSTM=%0.2f - LSTM-SHM=%0.2f' % (len(api_b4), nse_lstm, nse_lstm_shm))
            sn.heatmap(100*df_rlt, annot=True, cmap=plt.get_cmap('jet'))
            plt.savefig(os.path.join(out_save_dir, r"tstep_api_%s_heatmap.png" % (catch_id)), bbox_inches='tight')
            plt.close()
        # df_rlt.sum(axis=0)
                
                # api_b4.append(_d)
        # df_datetime_idx = pd.DatetimeIndex(merged_all_tdx)
        # df_qpm = pd.DataFrame(index=)
        
        #=======================================================================
        # 
        #=======================================================================
        
    #     plt.ioff()
    #     fig, ax1 = plt.subplots(1, 1, figsize=(12, 4), dpi=300)
    #
    #     ax2 = ax1.twinx()
    #     ax1.plot(df_stn.index, df_stn.values, c='gray') 
    #     ax2.plot(df_stn_pcp.loc[cmn_idx_q_pcp].index, -df_stn_pcp.loc[cmn_idx_q_pcp].values, c='b', alpha=0.2) 
    #     for td_idx in datetime_idx:  
    #         td_idx_cmn = df_stn.index.intersection(td_idx)         
    #         ax2.plot(df_stn_pcp.loc[td_idx_cmn, :].index, -df_stn_pcp.loc[td_idx_cmn,:].values, c='b', alpha=0.25)
    #         ax1.plot(df_stn.loc[td_idx_cmn, :].index, df_stn.loc[td_idx_cmn,:].values, c='r', alpha=0.5)
    #
    #         # break
    #
    #     # ax2.set_ylim([-60, 0])
    #     ax1.grid(alpha=0.5)
    #
    #     ax2.set_ylabel('P [mm/d]', c='b')
    #     ax1.set_ylabel('Q [m3/s]')
    #     ax1.set_xlabel('Time index')
    # #
    #     # plt.legend(loc=0)
    #     plt.savefig(os.path.join(out_save_dir, r"tstep_api_%s.png" % (catch_id)), bbox_inches='tight')
    #     plt.close()
    #
    #     pass
            
        #     df_t_p1_d = pd.DataFrame(index=range(cmn_idx_length))
        #     df_t_p1_d['Qt'] = df_in_vals_t
        #     df_t_p1_d['Qtp1'] = df_in_vals_tp1
        #     df_t_p1_d['d'] = depths_da_da
        #     df_t_p1_d['d2'] = depths_da_da2
        #     df_t_p1_d_low = df_t_p1_d['d'][(0 <= df_t_p1_d['d']) & (df_t_p1_d['d'] <=4)]
        #     df_t_p1_d_low2 = df_t_p1_d['d2'][(0 <= df_t_p1_d['d2']) & (df_t_p1_d['d2'] <=4)]
        #
        #     # df_t_p1_d_low_idx = df_t_p1_d.iloc[df_t_p1_d_low.index,:]
        #     plt.ioff()
        #     plt.figure(figsize=(4, 4), dpi=300)
        #
        #     plt.scatter(np.log(df_stn_t.iloc[:cmn_idx_length]),
        #                  np.log(df_stn_tp1.iloc[:cmn_idx_length]), alpha=0.71, facecolor='k',edgecolor='gray', s=15)
        #     plt.scatter(np.log(df_stn_t.iloc[:cmn_idx_length].iloc[df_t_p1_d_low.index]),
        #                  np.log(df_stn_tp1.iloc[:cmn_idx_length].iloc[df_t_p1_d_low.index]), alpha=0.51, facecolor='r',edgecolor='darkred', s=15)
        #     plt.scatter(np.log(df_stn_t.iloc[:cmn_idx_length].iloc[df_t_p1_d_low2.index]),
        #                  np.log(df_stn_tp1.iloc[:cmn_idx_length].iloc[df_t_p1_d_low2.index]), alpha=0.51, facecolor='b',edgecolor='darkblue', s=15)
        #
        #     plt.xlim([-1, 6])
        #     plt.ylim([-1, 6])
        #     plt.grid(alpha=0.5)
        #
        #
        #     plt.ylabel('ln(Q(t+1))')
        #
        #     plt.xlabel('ln(Q(t))')
        # #
        #     # plt.legend(loc=0)
        #     plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\00_Data_quality\scatter_q_%s.png" % (catch_id), bbox_inches='tight')
        #     plt.close()
        # normalize by median
        
        # df_stn_norm_orig = df_stn# / df_stn.median()
        # df_stn[df_stn > 90] = np.nan

 




    #===========================================================================
    # In this study, we normalized the data
    # depth between 0 and 1 by dividing the depth by half the total
    # number of points in the convex hull.
    #===========================================================================
    
    
    pass


# print('done')

if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    