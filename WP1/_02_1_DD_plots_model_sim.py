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
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import tqdm
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)
import scipy.stats as st

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

def nse(predictions, targets):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))


def depth_ftn(x, y, ei):

    mins = x.shape[0] * np.ones((y.shape[0],))  # initial value

    for i in ei:  # iterate over unit vectors
        d = np.dot(i, x.T)  # scalar product

        dy = np.dot(i, y.T)  # scalar product

        # argsort gives the sorting indices then we used it to sort d
        d = d[np.argsort(d)]

        dy_med = np.median(dy)
        dy = ((dy - dy_med) * (1 - (1e-10))) + dy_med

        # find the index of each project y in x to preserve order
        numl = np.searchsorted(d, dy)
        # numl is number of points less then projected y
        numg = d.shape[0] - numl

        # find new min
        mins = np.min(
            np.vstack([mins, np.min(np.vstack([numl, numg]), axis=0)]), 0)

    return mins

# ndims = 4
# ei = -1 + (2 * np.random.randn(1000, ndims))
#
# rand_pts_2 = np.random.normal(size=(100, 2))

# depth_val = depth_ftn(rand_pts_2, rand_pts_2, ei)

def main():
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    
    
    var_to_test = 'discharge_spec'
    
    path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)
    
    path_LSTM = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM/LSTM_discharge.csv",
                            index_col=0, parse_dates=True)
    
    path_LSTM_shm = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM_SHM/LSTM_SHM_discharge.csv",
                            index_col=0, parse_dates=True)
    #===========================================================================
    # Depth func parameters
    n_vecs = int(1e4)
    n_cpus = 3
    
    
    beg_date = '1970-01-01'
    end_date = '2015-09-30'
    
    
    time_steps_shift = [1, 2, 3, 4]
    
    nds = len(time_steps_shift)
    date_range = pd.date_range(start=beg_date, end=end_date, freq='D')
    
    out_save_dir = data_path / (r"Results\08_DD_plots_model_sim\nds_%d" % nds)
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
        
    #===========================================================================
    data_hdf5 = HDF5(infile=path_to_data)
    catch_ids = data_hdf5.get_all_names()
    
    catch_coords = data_hdf5.get_coordinates(ids=catch_ids)

    df_coords = pd.DataFrame(index=catch_ids,
        data=catch_coords['easting'], columns=['X'])
    y_dwd_coords = catch_coords['northing']
    df_coords.loc[:, 'Y'] = y_dwd_coords
    
    df = pd.DataFrame(index=path_LSTM.columns,
                      columns=['NSE LSTM','NSE LSTM-SHM','OBS not in LSTM', 'LSTM not in OBSV',  'OBSV not in LSTM_SHM', 'LSTM_SHM not in OBSV' ])
    
    
            
    for catch_id in tqdm.tqdm(path_LSTM.columns):
        
        if True:#not os.path.exists(out_save_dir / (r'%s_1d_2.png' % (catch_id))):
            print(catch_id)
            # break
            catch_id = '12001'
            df_stn = data_hdf5.get_pandas_dataframe_between_dates(
                catch_id, event_start=beg_date, event_end=end_date)
            df_stn = df_stn.dropna(how='all')
            
            df_lstm = path_LSTM.loc[:, catch_id]
            df_lstm_shm = path_LSTM_shm.loc[:, catch_id]
            
            cmn_idx_all = df_lstm.index.intersection(df_lstm_shm.index.intersection(
                df_stn.index))
            
            df_lstm = path_LSTM.loc[cmn_idx_all, catch_id]
            df_lstm_shm = path_LSTM_shm.loc[cmn_idx_all, catch_id]
            df_stn = df_stn.loc[cmn_idx_all, :]
            
            nse_lstm = nse(df_lstm.values.ravel(), df_stn.values.ravel())
            nse_lstm_shm = nse(df_lstm_shm.values.ravel(), df_stn.values.ravel())
            
            # normalize by median
            df_stn_norm_orig = df_stn / df_stn.median()
            df_lstm_norm = df_lstm / df_lstm.median()
            df_lstm_shm_norm = df_lstm_shm / df_lstm_shm.median()
            # df_stn[df_stn > 90] = np.nan
    
            df_orig_shift = pd.DataFrame(index=df_stn_norm_orig.index, columns=time_steps_shift)
            df_lstm_shift = pd.DataFrame(index=df_lstm_norm.index, columns=time_steps_shift)
            df_lstm_shm_shift = pd.DataFrame(index=df_lstm_shm_norm.index, columns=time_steps_shift)
    
            # nbr_idx = 0
            
            # plt.scatter(df_stn_norm_orig.values.ravel(), df_lstm_norm.values.ravel())
            # plt.show()
            usph_vecs = gen_usph_vecs_mp(n_vecs, df_stn_norm_orig.columns.size, n_cpus)
    
            for tshift in time_steps_shift:
                df_stn_norm_orig_shifted = df_stn_norm_orig.shift(tshift)
                df_lstm_norm_shifted = df_lstm_norm.shift(tshift)
                df_lstm_shm_norm_shift = df_lstm_shm_norm.shift(tshift)
                
                df_orig_shift.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
                df_lstm_shift.loc[df_lstm_norm_shifted.index, tshift] = df_lstm_norm_shifted.values.ravel()
                df_lstm_shm_shift.loc[df_lstm_shm_norm_shift.index, tshift] = df_lstm_shm_norm_shift.values.ravel()
                      
    
            df_orig_shift_nonan = df_orig_shift.dropna(axis=0)
            df_lstm_shift_nonan = df_lstm_shift.dropna(axis=0)
            df_lstm_shm_shift_nonan = df_lstm_shm_shift.dropna(axis=0)
            
            df_orig_shift_nonan_max = df_orig_shift_nonan.loc[(df_orig_shift_nonan.max(axis=1) > 5).index,:]
            df_lstm_shift_nonan = df_lstm_shift.loc[df_orig_shift_nonan_max.index,:].dropna(axis=0)

            
            cmn_idx = df_orig_shift_nonan.index.intersection(df_lstm_shm_shift_nonan.index)
            
            norm_fact_4d = cmn_idx.size / 2
                # break
            usph_vecs1d = gen_usph_vecs_mp(n_vecs, 1, n_cpus)
            
            usph_vecs = gen_usph_vecs_mp(n_vecs, df_lstm_shm_shift_nonan.columns.size, n_cpus)
            
            var_arr_orig = df_stn_norm_orig.values.astype(float).copy('c')
            var_arr_lstm = pd.DataFrame(df_lstm_norm).values.astype(float).copy('c')
            var_arr_lstm_shm =  pd.DataFrame(df_lstm_shm_norm).values.astype(float).copy('c')
            
            shifted_var_arr_orig = df_orig_shift_nonan.loc[cmn_idx].values.astype(float).copy('c')
            shifted_var_arr_lstm = df_lstm_shift_nonan.loc[cmn_idx].values.astype(float).copy('c')
            shifted_var_arr_lstm_shm = df_lstm_shm_shift_nonan.loc[cmn_idx].values.astype(float).copy('c')
            
            ranked_df_orig = df_orig_shift_nonan.rank(method='average', ascending=True, pct=True).values
            ranked_df_lstm = df_lstm_shift_nonan.rank(method='average', ascending=True, pct=True).values

            # ranked_df_orig- 0.5
            df_orig_shift_nonan / df_orig_shift_nonan.iloc[:,0].median()
            
            norm_fact_1d = var_arr_orig.size / 2
            
            # depths_da_da = depth_ftn_mp(var_arr_orig, var_arr_orig,
                                        # usph_vecs1d, n_cpus, 1)
            
            depths_da_osv_osv = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_orig,
                                        usph_vecs, n_cpus, 1)
            
            depths_da_osv_lstm = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_lstm,
                                        usph_vecs, n_cpus, 1)
            
            # plt.show()
            depths_da_lstm_obs = depth_ftn_mp(shifted_var_arr_lstm, shifted_var_arr_orig,
                                        usph_vecs, n_cpus, 1)
            
            
            # depths_da_lstm_obs2 = depth_ftn(shifted_var_arr_lstm, shifted_var_arr_orig, ei)
            
            # depths_da_lstm_obs = depth_ftn_mp(shifted_var_arr_lstm, shifted_var_arr_orig,
                                        # usph_vecs, n_cpus, 1)
            
            depths_da_lstm_shm_obsv = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_lstm_shm,
                                        usph_vecs, n_cpus, 1)
            
            depths_da_obsv_lstm_shm = depth_ftn_mp(shifted_var_arr_lstm_shm, shifted_var_arr_orig,
                                        usph_vecs, n_cpus, 1)
            
            # depths_da_lstm_lstm = depth_ftn_mp(var_arr_lstm, var_arr_lstm,
            #                             usph_vecs1d, n_cpus, 1)
            #
            # depths_da_lstm_shm = depth_ftn_mp(var_arr_lstm_shm, var_arr_lstm_shm,
            #                             usph_vecs1d, n_cpus, 1)
            
            
            
            low_d_orig = np.where([depths_da_osv_lstm == 0])[1]
            low_d_lstm = np.where([depths_da_lstm_obs == 0])[1]
            low_d_obsv_shm = np.where([depths_da_lstm_shm_obsv == 0])[1]
            low_d_shm_obsv = np.where([depths_da_obsv_lstm_shm == 0])[1]
            
            # low_d_orig_vals = depths_da_osv_lstm[low_d_lstm]
            # low_d_lstmm_vals = depths_da_lstm_obs[low_d_lstm]
            # low_d_orig_shm_vals = depths_da_lstm_shm_obsv[low_d_obsv_shm]
            # low_d_shm_orig_vals = depths_da_obsv_lstm_shm[low_d_lstm]
            
            low_Q_orig_vals = shifted_var_arr_orig[low_d_orig]
            low_Q_lstmm_vals = shifted_var_arr_orig[low_d_lstm]
            low_Q_orig_shm_vals = shifted_var_arr_orig[low_d_obsv_shm]
            low_Q_shm_orig_vals = shifted_var_arr_orig[low_d_shm_obsv]
            
            depths_da_osv_lstm / norm_fact_1d
            depths_da_osv_lstm.shape
            
            plt.ioff()
            plt.figure(figsize=(4, 4), dpi=300)
            plt.scatter(shifted_var_arr_orig[:,0], shifted_var_arr_orig[:,1], c ='gray', label='n=%d vectors' % shifted_var_arr_orig[:,1].shape[0])
            plt.scatter(low_Q_lstmm_vals[:,0], low_Q_lstmm_vals[:,1], c ='r', label='D=0 OBSV not in LSTM n=%d' % low_Q_lstmm_vals[:,1].shape[0])
            plt.scatter(low_Q_orig_vals[:,0], low_Q_orig_vals[:,1], c ='b', label='D=0 LSTM not in OBSV n=%d'% low_Q_orig_vals[:,1].shape[0]) 
            plt.grid(alpha=0.6)
            plt.legend(loc=0, ncols=1, fontsize=8)
            plt.axis('equal')
            plt.xlabel('Observed Q(t)')
            plt.ylabel('Observed Q(t+1)')
            plt.tight_layout()
            plt.savefig(out_save_dir / (r'Q_obsv_lstm_plot_%s_%dD.png' % (catch_id, nds)), bbox_inches='tight')
            plt.close()
            
            #===================================================================
            # 
            #===================================================================
            plt.ioff()
            plt.figure(figsize=(4, 4), dpi=300)
            plt.scatter(shifted_var_arr_orig[:,0], shifted_var_arr_orig[:,1], c ='gray', label='n=%d vectors' % shifted_var_arr_orig[:,1].shape[0])
            plt.scatter(low_Q_shm_orig_vals[:,0], low_Q_shm_orig_vals[:,1], c ='r', label='D=0 OBSV not in LSTM-SHM  n=%d'% low_Q_shm_orig_vals[:,1].shape[0]) 
            plt.scatter(low_Q_orig_shm_vals[:,0], low_Q_orig_shm_vals[:,1], c ='b', label='D=0 LSTM-SHM not in OBSV n=%d' % low_Q_orig_shm_vals[:,1].shape[0])
            plt.grid(alpha=0.6)
            plt.legend(loc=0, ncols=1, fontsize=8)
            plt.axis('equal')
            plt.xlabel('Observed Q(t)')
            plt.ylabel('Observed Q(t+1)')
            plt.tight_layout()
            plt.savefig(out_save_dir / (r'Q_obsv_lstm_plot_%s_%dD.png' % (catch_id, nds)), bbox_inches='tight')
            plt.close()
            
            #===================================================================
            # 
            #===================================================================
            plt.ioff()
            plt.figure(figsize=(8, 4), dpi=300)
            plt.plot(df_orig_shift_nonan.loc[cmn_idx].index, df_orig_shift_nonan.loc[cmn_idx].values, c ='gray', label='NSE=%0.2f' % nse_lstm)
            plt.plot(df_lstm_shift_nonan.loc[cmn_idx].index, df_lstm_shift_nonan.loc[cmn_idx].values, c ='g', label='Q model')

            #, label='n=%d vectors' % shifted_var_arr_orig[:,1].shape[0])
            if len(low_d_lstm) > 0:
                plt.scatter(df_orig_shift_nonan.iloc[low_d_lstm].index, df_orig_shift_nonan.iloc[low_d_lstm, 0].values, c ='r', label='D=0 OBSV not in LSTM n=%d' % low_d_lstm.shape[0])
            if len(low_d_orig) > 0:
                plt.scatter(df_orig_shift_nonan.iloc[low_d_orig].index, df_orig_shift_nonan.iloc[low_d_orig, 0].values, c ='b', label='D=0 LSTM not in OBSV n=%d'% low_d_orig.shape[0]) 

            plt.grid(alpha=0.6)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys() , loc=0, fontsize=8, ncols=2)

            # plt.legend()
            # plt.axis('equal')
            plt.xlabel('Time')
            plt.ylabel('Q(t) m/s')
            # plt.tight_layout()
            plt.savefig(out_save_dir / (r'Time_Q_obsv_lstm_plot_%s_%sD.png' % (catch_id, nds)), bbox_inches='tight')
            plt.close()
            #===================================================================
            # 
            #===================================================================
            plt.ioff()
            plt.figure(figsize=(8, 4), dpi=300)
            plt.plot(df_orig_shift_nonan.loc[cmn_idx].index, df_orig_shift_nonan.loc[cmn_idx].values, c ='gray', label='NSE=%0.2f' % nse_lstm_shm)
            plt.plot(df_lstm_shm_shift_nonan.loc[cmn_idx].index, df_lstm_shm_shift_nonan.loc[cmn_idx].values, c ='g', label='Q model')

            #, label='n=%d vectors' % shifted_var_arr_orig[:,1].shape[0])
            if len(low_d_shm_obsv) > 0:
                
                plt.scatter(df_orig_shift_nonan.iloc[low_d_shm_obsv].index, df_orig_shift_nonan.iloc[low_d_shm_obsv, 0].values, c ='r', label='D=0 OBSV not in LSTM-SHM n=%d' % low_Q_shm_orig_vals[:,1].shape[0])
            if len(low_d_obsv_shm) > 0:
                plt.scatter(df_orig_shift_nonan.iloc[low_d_obsv_shm].index, df_orig_shift_nonan.iloc[low_d_obsv_shm, 0].values, c ='b', label='D=0 LSTM-SHM not in OBSV n=%d'% low_Q_orig_shm_vals[:,1].shape[0]) 

            plt.grid(alpha=0.6)
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys() , loc=0, fontsize=8, ncols=2)

            # plt.legend()
            # plt.axis('equal')
            plt.xlabel('Time')
            plt.ylabel('Q(t) m/s')
            # plt.tight_layout()
            plt.savefig(out_save_dir / (r'Time_Q_obsv_lstmshm_plot_%s_%sD.png' % (catch_id, nds)), bbox_inches='tight')
            plt.close()
            
            
            #===================================================================
            # 
            #===================================================================
            
            # depths_da_osv_lstm[depths_da_osv_lstm == 0] = -1
            # depths_da_lstm_obs[depths_da_lstm_obs == 0] = -1
            
            df.loc[catch_id, :] = [nse_lstm, nse_lstm_shm, len(low_d_lstm), len(low_d_orig), len(low_d_shm_obsv), len(low_d_obsv_shm)]
    df.to_csv(out_save_dir / (r'%dD_depth_out.csv' % nds))

            # plt.scatter(depths_da_osv_lstm, depths_da_lstm_obs)
            # plt.scatter(low_d_orig_vals, low_d_lstmm_vals, s=df_orig_shift_nonan.iloc[low_d_lstm,:].max(axis=1).values.ravel())
            #
            #
            # low_d_orig_vals.shape
            # plt.ioff()                
            # fig, (ax) = plt.subplots(1,1, dpi=300, figsize=(12, 4))
            #
            # ax.scatter(df_orig_shift_nonan.index, df_orig_shift_nonan.values, c='r', label='Obsv')
            # ax.scatter(df_lstm_shift.index, df_lstm_shift.values, c='r', label='Obsv')
            # plt.grid(alpha=0.5)
            # plt.show()
            
            
            # df.columns
            #
    plt.ioff()                
    fig, (ax) = plt.subplots(1,1, dpi=300, figsize=(12, 4))
    for _col in df.columns:
        if 'OBS' in _col:
            ax.plot(df.index, df.loc[:,_col], label='%s' % _col, alpha=0.95)
    
    
    ax.set_xticklabels(labels=df.index, rotation=45, fontsize=10)
    ax.legend(loc=0, ncols=3, fontsize=8)
    
    plt.xlabel('Catchment ID')
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_save_dir / (r'%dD_all.png' % nds), bbox_inches='tight')
    plt.close()
            #
            # df_nse_d0 = df.iloc[:,[0, 2]]
            # df_nse_d0sorted = df_nse_d0.sort_values(by='NSE LSTM')
            # fig, (ax2) = plt.subplots(1,1, dpi=300, figsize=(4, 4))    
            # ax2.scatter(df_nse_d0sorted.iloc[:,0], df_nse_d0sorted.iloc[:,1]/df_nse_d0sorted.iloc[:,1].max(),  c='r')
            # # ax2.legend(loc='upper left')
            # plt.xlabel('NSE-LSTM')
            # plt.ylabel('OBSV not in LSTM')
            # plt.grid(alpha=0.5)
            # plt.tight_layout()
            # plt.savefig(out_save_dir / (r'nse_d0_lstm.png' ), bbox_inches='tight')
            # plt.close()
            # df.to_csv(out_save_dir / (r'all_4d.csv'))
            # plt.ioff()                
            # fig, (ax) = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            # ax.scatter(depths_da_osv_lstm , depths_da_lstm_obs, s=30,
            #             marker='o', edgecolor='k', facecolor='k', alpha=0.74, label='D')
            #

            
            # df_orig_shift_nonan.iloc[low_d_lstm,:].plot()
            
            # df_lstm_shift_nonan.iloc[low_d_lstm,:].plot()
            # plt.show()
            # plt.ioff()                
            # fig, (ax) = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # # ax.plot([0, 1000], [0, 1000], c='b', linestyle='--', alpha=0.75)
            # ax.scatter(ranked_df_orig[:,0] , ranked_df_orig[:,1], s=depths_da_osv_lstm/10,
            #            marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='LSTM in Obsv')
            # ax.scatter(ranked_df_lstm[:,0] , ranked_df_lstm[:,1], s=depths_da_lstm_obs/10,
            #             marker='o', edgecolor='b', facecolor='darkblue', alpha=0.4, label='Obsv in LSTM')
            #
            #
            #
            # # ax.scatter(ranked_df_lstm[:,0][low_d_lstm] , ranked_df_lstm[:,1][low_d_lstm], s=depths_da_lstm_obs[low_d_lstm]*10,
            #             # marker='D', edgecolor='lime', facecolor='g', alpha=0.4)
            # ax.scatter(ranked_df_orig[:,0][low_d_orig] , ranked_df_orig[:,1][low_d_orig], s=depths_da_osv_lstm[low_d_orig]*10,
            #            marker='X', edgecolor='red', facecolor='r', alpha=0.4)
            #
            #
            # ax.set_xlim([-.1, 1.1])
            # ax.set_ylim([-.1, 1.1])
            
            
            
            
            # ax.scatter(depths_da_da , depths_da_lstm_shm, s=10,
                       # marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv-LSTM SHM' )
            
            # # ax.legend(loc=0, ncols=2)
            # ax.set_xlabel(r'Observation - $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM- $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            #
            # # ax2.plot([0, 10], [0, 10], c='b', linestyle='--', alpha=0.75)
            #
            # ax2.scatter(depths_da_da[low_d_orig] , depths_da_lstm[low_d_orig], s=30,
            #            marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv')
            # ax2.scatter(depths_da_da[low_d_lstm] , depths_da_lstm_lstm[low_d_lstm], s=30,
            #            marker='X', edgecolor='b', facecolor='c', alpha=0.4, label='LSTM')
            # ax2.scatter(depths_da_da[low_d_lstm_shm] , depths_da_lstm_shm[low_d_lstm_shm], s=20,
                       # marker='D', edgecolor='g', facecolor='lime', alpha=0.4   , label='LSTM')
            # ax2.set_xlim([0, 10])
            # ax2.set_ylim([0, 10])
            # ax.grid(alpha=0.7)
            # ax.set_xticklabels(labels=df.index, rotation=45, fontsize=10)
            # ax.legend(loc=0, ncols=3, fontsize=8)
            # plt.tight_layout()
            # plt.savefig(out_save_dir / (r'4d_all.png' ), bbox_inches='tight')
            # plt.close()
            #

            
            # plt.ioff()                
            # fig, (ax, ax2) = plt.subplots(1,2, dpi=300, figsize=(6, 4))
            #
            # ax.plot([0, 1000], [0, 1000], c='b', linestyle='--', alpha=0.75)
            # ax.scatter(depths_da_da , depths_da_lstm, s=10,
            #            marker='o', edgecolor='k', facecolor='gray', alpha=0.4 )
            # # ax.scatter(depths_da_da , depths_da_lstm_shm, s=10,
            #            # marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv-LSTM SHM' )
            #
            # # ax.legend(loc=0, ncols=2)
            # ax.set_xlabel(r'Observation - $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM- $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            #
            # # ax2.plot([0, 10], [0, 10], c='b', linestyle='--', alpha=0.75)
            #
            # ax2.scatter(depths_da_da[low_d_orig] , depths_da_lstm[low_d_orig], s=30,
            #            marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv')
            # ax2.scatter(depths_da_da[low_d_lstm] , depths_da_lstm_lstm[low_d_lstm], s=30,
            #            marker='X', edgecolor='b', facecolor='c', alpha=0.4, label='LSTM')
            # # ax2.scatter(depths_da_da[low_d_lstm_shm] , depths_da_lstm_shm[low_d_lstm_shm], s=20,
            #            # marker='D', edgecolor='g', facecolor='lime', alpha=0.4   , label='LSTM')
            # # ax2.set_xlim([0, 10])
            # # ax2.set_ylim([0, 10])
            # ax2.legend(loc=0, ncols=2, fontsize=8)
            # plt.tight_layout()
            # plt.savefig(out_save_dir / (r'%s_1d.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # plt.ioff()   
            #

            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.plot([0, 1000], [0, 1000], c='b', linestyle='--', alpha=0.75)
            # # ax.scatter(depths_da_da , depths_da_lstm, s=10,
            #            # marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='Obsv-LSTM' )
            # ax.scatter(depths_da_da , depths_da_lstm_shm, s=10,
            #             marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv-LSTM SHM' )
            #
            # ax.legend(loc=0, ncols=2)
            # ax.set_xlabel(r'Observation - $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM SHM- $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_1d_shm.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # plt.ioff()   
            #
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.plot([0, 1000], [0, 1000], c='b', linestyle='--', alpha=0.75)
            # # ax.scatter(depths_da_da , depths_da_lstm, s=10,
            #            # marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='Obsv-LSTM' )
            # ax.scatter(depths_da_da , depths_da_lstm, s=10,
            #             marker='o', edgecolor='r', facecolor='darkred', alpha=0.4, label='Obsv-LSTM SHM' )
            #
            # ax.legend(loc=0, ncols=2)
            # ax.set_xlabel(r'Observation - $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM- $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_1d_lstm.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # plt.ioff()                
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.plot([0, 1000], [0, 1000], c='b', linestyle='--', alpha=0.75)
            # # ax.scatter(depths_da_da , depths_da_lstm, s=10,
            #            # marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='Obsv-LSTM' )
            # ax.scatter(depths_da_lstm , depths_da_lstm_shm, s=10,
            #             marker='o', edgecolor='r', facecolor='darkred', alpha=0.4 )
            #
            # ax.set_xlabel(r'Model LSTM - $D_{\beta}(x)$')
            # ax.set_ylabel(r'Model LSTM-SHM - $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_1d_ls_sh.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # #=======================================================================
            # # 
            # #=======================================================================
            #
            #
            # depths_da_da_4d = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_orig,
            #                             usph_vecs, n_cpus, 1)
            #
            # depths_da_lstm_4d = depth_ftn_mp(shifted_var_arr_lstm, shifted_var_arr_orig,
            #                             usph_vecs, n_cpus, 1)
            #
            # depths_da_lstm_shm_4d = depth_ftn_mp(shifted_var_arr_lstm_shm, shifted_var_arr_orig,
            #                             usph_vecs, n_cpus, 1)
            #
            # plt.ioff()                
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.scatter(depths_da_da_4d, depths_da_da_4d, c='r', marker='.', alpha=0.75, s=1)
            # ax.scatter(depths_da_da_4d , depths_da_lstm_4d, s=10,
            #            marker='o', edgecolor='k', facecolor='gray', alpha=0.4)
            # # ax.scatter(depths_da_da_4d / norm_fact_4d , depths_da_lstm_shm_4d / norm_fact_4d, s=10,
            #            # marker='o', edgecolor='orange', facecolor='darkred', alpha=0.4, label='Obsv-LSTM SHM' )
            #
            # # ax.legend(loc=0, ncols=2, fontsize=7)
            # ax.set_xlabel(r'Obsv. $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_4d_lstm.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            # #=======================================================================
            # # 
            # #=======================================================================
            # plt.ioff()                
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.scatter(depths_da_da_4d, depths_da_da_4d, c='r', marker='.', alpha=0.75, s=1)
            # # ax.scatter(depths_da_da_4d / norm_fact_4d , depths_da_lstm_4d / norm_fact_4d, s=10,
            #            # marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='Obsv-LSTM' )
            # ax.scatter(depths_da_da_4d , depths_da_lstm_shm_4d, s=10,
            #            marker='o', edgecolor='orange', facecolor='darkred', alpha=0.4)
            #
            # # ax.legend(loc=0, ncols=2, fontsize=7)
            # ax.set_xlabel(r'Obsv. $D_{\alpha}(x)$')
            # ax.set_ylabel(r'Model LSTM-SHM $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_4d_lstm_shm.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # # depths2 = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_ngbr, usph_vecs, n_cpus, 1)
            # print('done calculating depth')
            #
            #
            # #=======================================================================
            # # 
            # #=======================================================================
            #
            # depths_da_da = depth_ftn_mp(var_arr_orig, var_arr_orig,
            #                             usph_vecs1d, n_cpus, 1)
            #
            # depths_da_lstm = depth_ftn_mp(var_arr_lstm, var_arr_lstm,
            #                             usph_vecs1d, n_cpus, 1)
            #
            # depths_da_lstm_shm = depth_ftn_mp(var_arr_lstm_shm, var_arr_lstm_shm,
            #                             usph_vecs1d, n_cpus, 1)
            #
            #
            # plt.ioff()                
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            #
            # ax.scatter(var_arr_orig, depths_da_da, c='r', marker='.', alpha=0.5, s=10, label='Obsv')
            # ax.scatter(var_arr_lstm, depths_da_lstm, c='b', marker='.', alpha=0.5, s=10, label='LSTM')
            # ax.scatter(var_arr_lstm_shm, depths_da_lstm_shm, c='g', marker='.', alpha=0.5, s=10, label='LSTM-SHM')
            #
            # ax.set_xticks(np.arange(0, np.max(var_arr_orig), 2))
            #
            # ax.legend(loc=0, ncols=1, fontsize=7)
            # ax.set_xlabel(r'Specific Q $m/s$')
            # ax.set_ylabel(r'Depth $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_1d_sp.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # # depths2 = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_ngbr, usph_vecs, n_cpus, 1)
            # plt.ioff()                
            # fig, ax = plt.subplots(1,1, dpi=300, figsize=(6, 4))
            #
            # # ax.scatter(df_stn_norm_orig.index, df_stn_norm_orig.values, s=depths_da_da/1000,
            # #            c='r', marker='x', alpha=0.95)
            # # ax.scatter(df_lstm_norm.index, df_lstm_norm.values, s=depths_da_lstm/1000,
            # #            c='b', marker='x', alpha=0.95)
            # # ax.scatter(df_lstm_shm_norm.index, df_lstm_shm_norm.values, c='g',
            # #             marker='x', alpha=0.95, s=depths_da_lstm_shm/1000)
            # #
            #
            # ax.plot(df_stn_norm_orig.index, df_stn_norm_orig.values,
            #            c='r', alpha=0.75, label='Obsv')
            # ax.plot(df_lstm_norm.index, df_lstm_norm.values, 
            #            c='b', alpha=0.75, label='LSTM')
            # ax.plot(df_lstm_shm_norm.index, df_lstm_shm_norm.values, c='g',
            #             alpha=0.75, label='LSTM-SHM')
            #
            #
            #
            # # ax.set_xticks(np.arange(0, np.max(var_arr_orig), 2))
            #
            # ax.legend(loc=0, ncols=1, fontsize=7)
            # ax.set_ylabel(r'Specific Q $m/s$')
            # # ax.set_ylabel(r'Depth $D_{\beta}(x)$')
            # ax.set_title('Catch. %s - NSE-LSTM: %0.2f - NSE-LSTM-SHM: %0.2f'
            #              % (catch_id, nse_lstm, nse_lstm_shm))
            # ax.grid(alpha=0.5)
            # plt.savefig(out_save_dir / (r'%s_1d_time.png' % (catch_id)), bbox_inches='tight')
            # plt.close()
            #
            # # depths2 = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_ngbr, usph_vecs, n_cpus, 1)
            # print('done calculating depth')
    #===========================================================================
    # In this study, we normalized the data
    # depth between 0 and 1 by dividing the depth by half the total
    # number of points in the convex hull.
    #===========================================================================
    
    
    
    # """
    # Hence, if the DDplot
    # of two catchments is close to the diagonal line, we assume
    # the catchments have similar response behaviour. A Q(t) vs Q
    # (t − 1) (d = 2) plot for two sample catchments α and β is shown
    # in Figure 3(a), while the corresponding Q(t), Q(t − 1) and Q
    # (t − 2) (d = 3) plots for the same catchments are shown in Figure
    # 3(b). A DD-plot where d = 4 is given in Figure 3(c). Figures 3(a)
    # and (b) summarize the flow dynamics of the catchment in two
    # and three dimensions, where we can see that catchments α and β
    # have quite similar behaviour. In Figure 3(c), the DD-plot summarizes
    # the four-dimensional relationship between flow
    # dynamics of catchments α and β.
    # """
    

if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    