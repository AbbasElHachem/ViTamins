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
    out_save_dir = data_path / r"Results\06_disch_Q_4D"
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    
    n_vecs = int(1e4)
    n_cpus = 7
    
    variables = ['']
    
    time_steps_shift = [1, 2, 3, 4]
    
    path_ids_abv_150 = pd.read_csv(
        r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\IDs_of_Catch_A_abv_150km2.csv", sep=',', index_col=0)
    
    
    path_to_data_q = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'discharge_spec')
    path_to_data_pcp = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'precipitation')
    
    path_LSTM = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM/LSTM_discharge.csv",
                            index_col=0, parse_dates=True)
    
    path_LSTM_shm = pd.read_csv(r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM_SHM/LSTM_SHM_discharge.csv",
                            index_col=0, parse_dates=True)
    
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
    df_coords.loc[:, 'lon'] =catch_coords['lon']
    df_coords.loc[:, 'lat'] =catch_coords['lat']
    
    usph_vecs = gen_usph_vecs_mp(n_vecs, 1, n_cpus)
    
    usph_vecs_dQ = gen_usph_vecs_mp(n_vecs, 4, n_cpus)
    
    catch_ids_to_use = path_LSTM_shm.columns
    
    df_results = pd.DataFrame(index=catch_ids_to_use,
                              columns=['n_obsv', 'n_lstm', 'n_lstm_shm', 'n_obsv_lstm', 'n_obsv_lstm_shm', 'n_lstm_lstm_shm', 'nse_lstm', 'nse_lstm_shm'])
    
    # df_coords_to_use = df_coords.loc[catch_ids_to_use, :]
    # df_coords_to_use.to_csv(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\coords_60_EZG_Roberto.csv")
    for catch_id in tqdm.tqdm(catch_ids_to_use):
        # if catch_id == '14002':
        print(catch_id)
        # catch_id = '27003'
        # catch_id = '12001'
        # break
        df_stn = data_hdf5.get_pandas_dataframe(catch_id)
        df_stn = df_stn.dropna(how='all')
        
        df_lstm = path_LSTM.loc[:, catch_id]
        df_lstm_shm = path_LSTM_shm.loc[:, catch_id]
            
        df_stn_pcp = data_hdf5_pcp.get_pandas_dataframe(catch_id)
        df_stn_pcp = df_stn_pcp.dropna(how='all')
        
        cmn_idx_all = df_lstm.index.intersection(df_lstm_shm.index.intersection(
                df_stn.index))
            
        df_lstm = path_LSTM.loc[cmn_idx_all, catch_id]
        df_lstm_shm = path_LSTM_shm.loc[cmn_idx_all, catch_id]
        df_stn = df_stn.loc[cmn_idx_all, :]
        
        nse_lstm = nse(df_lstm.values.ravel(), df_stn.values.ravel())
        nse_lstm_shm = nse(df_lstm_shm.values.ravel(), df_stn.values.ravel())
        
            
        cmn_idx_q_pcp = df_stn_pcp.index.intersection(df_stn.index)
        df_orig_shift = pd.DataFrame(index=df_stn.index, columns=time_steps_shift)
        df_lstm_shift = pd.DataFrame(index=df_lstm.index, columns=time_steps_shift)
        df_lstm_shm_shift = pd.DataFrame(index=df_lstm_shm.index, columns=time_steps_shift)

        # nbr_idx = 0
        
        usph_vecs = gen_usph_vecs_mp(n_vecs, df_stn.columns.size, n_cpus)

        for tshift in time_steps_shift:
            df_stn_norm_orig_shifted = df_stn.shift(tshift)
            df_lstm_norm_shifted = df_lstm.shift(tshift)
            df_lstm_shm_norm_shift = df_lstm_shm.shift(tshift)
            
            df_orig_shift.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
            df_lstm_shift.loc[df_lstm_norm_shifted.index, tshift] = df_lstm_norm_shifted.values.ravel()
            df_lstm_shm_shift.loc[df_lstm_shm_norm_shift.index, tshift] = df_lstm_shm_norm_shift.values.ravel()
                  
        df_orig_shift_nonan = df_orig_shift.dropna(axis=0)
        
        df_lstm_shift_nonan = df_lstm_shift.dropna(axis=0)
        df_lstm_shm_shift_nonan = df_lstm_shm_shift.dropna(axis=0)
        
        df_orig_shift_nonan_d = df_orig_shift_nonan.values.astype(float).copy('c')
        df_lstm_shift_nonan_d = df_lstm_shift_nonan.values.astype(float).copy('c')
        df_lstm_shm_shift_nonan_d = df_lstm_shm_shift_nonan.values.astype(float).copy('c')
        
        depths_da_Q = depth_ftn_mp(df_orig_shift_nonan_d, df_orig_shift_nonan_d, usph_vecs_dQ, n_cpus)
        depths_da_Q_lstm = depth_ftn_mp(df_lstm_shift_nonan_d, df_lstm_shift_nonan_d, usph_vecs_dQ, n_cpus)
        
        depths_da_Q_lstm_shm = depth_ftn_mp(df_lstm_shm_shift_nonan_d, df_lstm_shm_shift_nonan_d, usph_vecs_dQ, n_cpus)
        
        df_orig_shift_nonan.loc[:, 'd'] = depths_da_Q
        depths_da_da_low = df_orig_shift_nonan.d[(1 <= df_orig_shift_nonan.d) & (df_orig_shift_nonan.d <=4)]
        
        
        df_lstm_shift_nonan.loc[:, 'd'] = depths_da_Q_lstm
        depths_da_da_low_lstm = df_lstm_shift_nonan.d[(1 <= df_lstm_shift_nonan.d) & (df_lstm_shift_nonan.d <=4)]
        
        df_lstm_shm_shift_nonan.loc[:, 'd'] = depths_da_Q_lstm_shm
        depths_da_da_low_lstm_shm = df_lstm_shm_shift_nonan.d[(1 <= df_lstm_shm_shift_nonan.d) & (df_lstm_shm_shift_nonan.d <=4)]
        
        
        dq_m = depths_da_da_low.index - pd.Timedelta(days=2)
        dq_p = depths_da_da_low.index + pd.Timedelta(days=2)
        
        dq_m_lstm = depths_da_da_low_lstm.index - pd.Timedelta(days=4)
        dq_p_lstm = depths_da_da_low_lstm.index + pd.Timedelta(days=4)
        
        dq_m_lstm_shm = depths_da_da_low_lstm_shm.index - pd.Timedelta(days=4)
        dq_p_lstm_shm = depths_da_da_low_lstm_shm.index + pd.Timedelta(days=4)
        
        datetime_idx = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m, dq_p)]
        datetime_idx_lstm = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m_lstm, dq_p_lstm)]
        datetime_idx_lstm_shm = [pd.date_range(start=st, end=et, freq='D') for st, et in zip(dq_m_lstm_shm, dq_p_lstm_shm)]
        
        
        cmn_low_lstm = depths_da_da_low_lstm.index.intersection(depths_da_da_low.index)
        cmn_low_lstm_shm = depths_da_da_low_lstm_shm.index.intersection(depths_da_da_low.index)
        cmn_low_lstm_lstm_shm = depths_da_da_low_lstm_shm.index.intersection(depths_da_da_low_lstm.index)
        
        diff_low_lstm = depths_da_da_low_lstm.index.difference(depths_da_da_low.index)
        diff_low_lstm_shm = depths_da_da_low.index.difference(depths_da_da_low_lstm_shm.index)
        diff_low_lstm_lstm_shm = depths_da_da_low_lstm_shm.index.difference(depths_da_da_low_lstm.index)
        
        
        print(cmn_low_lstm.shape, cmn_low_lstm_shm.shape)
        
        df_results.loc[catch_id, :] = [depths_da_da_low.index.shape[0], depths_da_da_low_lstm.index.shape[0], depths_da_da_low_lstm_shm.index.shape[0],
                                       cmn_low_lstm.shape[0], cmn_low_lstm_shm.shape[0], cmn_low_lstm_lstm_shm.shape[0], nse_lstm, nse_lstm_shm]
        
        
        merged_all_tdx = list(itertools.chain(*datetime_idx))
        
        
        # plt.ioff()
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharey=False, sharex=False)
        #
        # ax1.scatter(depths_da_Q, depths_da_Q_lstm, facecolor='gray', edgecolor='k', marker='o', label='LSTM')
        # # ax1.scatter(depths_da_da_low.values.ravel(), df_lstm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='r', edgecolor='darkred', marker='X', label='low-D')
        #
        # ax2.scatter(depths_da_Q, depths_da_Q_lstm_shm, edgecolor='darkblue', facecolor='b', marker='X', label='LSTM-SHM')
        # # ax2.scatter(depths_da_da_low.values.ravel(), df_lstm_shm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='darkred', edgecolor='r', marker='D', label='Low-D')
        # ax3.scatter(depths_da_Q_lstm, depths_da_Q_lstm_shm, edgecolor='darkgreen', facecolor='g', marker='d', label='Models')
        #
        # ax1.set_ylabel('Depth model LSTM')
        # ax1.set_xlabel('Depth observed')
        #
        # ax2.set_ylabel('Depth model LSTM-SHM')
        # ax2.set_xlabel('Depth observed')
        #
        # ax3.set_ylabel('Depth model LSTM-SHM')
        # ax3.set_xlabel('Depth model LSTM')
        #
        # ax1.plot([0, max(df_orig_shift_nonan.d.max(), df_lstm_shift_nonan.d.max())], 
        #          [0, max(df_orig_shift_nonan.d.max(), df_lstm_shift_nonan.d.max())], 'r-.')
        # ax2.plot([0, max(df_orig_shift_nonan.d.max(), df_lstm_shm_shift_nonan.d.max())], 
        #          [0, max(df_orig_shift_nonan.d.max(), df_lstm_shm_shift_nonan.d.max())], 'r-.')
        #
        # ax3.plot([0, max(df_lstm_shm_shift_nonan.d.max(), df_lstm_shift_nonan.d.max())], 
        #          [0, max(df_lstm_shm_shift_nonan.d.max(), df_lstm_shift_nonan.d.max())], 'r-.')
        #
        #
        # ax1.grid(alpha=0.5)
        # ax2.grid(alpha=0.5)
        # ax3.grid(alpha=0.5)
        # ax1.legend(loc=0)
        # ax2.legend(loc=0)
        # ax3.legend(loc=0)
        # plt.tight_layout()
        # plt.savefig(os.path.join(out_save_dir, r"obsv_model_4q_%s.png" % (catch_id)), bbox_inches='tight')
        # plt.close()
        

        
    #     plt.ioff()
    #     fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=300, sharey=False, sharex=False)
    #
    #     ax1.scatter(df_lstm.loc[diff_low_lstm].index, df_lstm.loc[diff_low_lstm].values, c='r', marker='X', label='D not in Obsv')
    #
    #     ax1.plot(df_stn.index, df_stn.values, c='gray', label='obsv')
    #     ax1.plot(df_lstm.index, df_lstm.values, c='g', label='LSTM')
    #
    #     ax1.grid(alpha=0.5)
    #
    #     ax1.set_ylabel('Q [m/s]')
    #     ax1.set_xlabel('Time index')
    # #    
    #     plt.legend(loc=0)
    #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_obsv_models.png" % (catch_id)), bbox_inches='tight')
    #     plt.close()
    #
    #
    #     plt.ioff()
    #     fig, (ax1) = plt.subplots(1, 1, figsize=(8, 4), dpi=300, sharey=False, sharex=False)
    #
    #     ax1.scatter(df_lstm.loc[diff_low_lstm].loc['2010'].index, df_lstm.loc[diff_low_lstm].loc['2010'].values, c='r', marker='X', label='D')
    #
    #     ax1.plot(df_stn.loc['2010'].index, df_stn.loc['2010'].values, c='gray', label='obsv')
    #     ax1.plot(df_lstm.loc['2010'].index, df_lstm.loc['2010'].values, c='g', label='LSTM')
    #
    #     ax1.grid(alpha=0.5)
    #
    #     ax1.set_ylabel('Q [m/s]')
    #     ax1.set_xlabel('Time index')
    # #    
    #     plt.legend(loc=0)
    #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_obsv_models2.png" % (catch_id)), bbox_inches='tight')
    #     plt.close()
        
        
        
        
        # df_datetime_idx = pd.DatetimeIndex(merged_all_tdx)
        # df_qpm = pd.DataFrame(index=)
    
        
        # try:
        #     plt.ioff()
        #     fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        #
        #     # ax2 = ax1.twinx()
        #     ax1.plot(df_stn.index, df_stn.values, c='gray') 
        #     # ax2.plot(df_stn_pcp.loc[cmn_idx_q_pcp].index, -df_stn_pcp.loc[cmn_idx_q_pcp].values, c='b', alpha=0.2) 
        #
        #     for td_idx in datetime_idx:  
        #         td_idx_cmn = df_stn.index.intersection(td_idx)         
        #         # ax2.plot(df_stn_pcp.loc[td_idx_cmn, :].index, -df_stn_pcp.loc[td_idx_cmn,:].values, c='b', alpha=0.25)
        #         ax1.plot(df_stn.loc[td_idx_cmn, :].index, df_stn.loc[td_idx_cmn,:].values, c='r', alpha=0.5)
        #
        #     ax1.grid(alpha=0.5)
        #
        #     ax2.set_ylabel('P [mm/d]', c='b')
        #     ax1.set_ylabel('Q [m/s]')
        #     ax1.set_xlabel('Time index')
        # #
        #     # plt.legend(loc=0)
        #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_obsv.png" % (catch_id)), bbox_inches='tight')
        #     plt.close()
        #
        #     plt.ioff()
        #     fig, ax2 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        #     ax2.plot(df_lstm.index, df_lstm.values, c='gray')
        #     for td_idx_lstm in datetime_idx_lstm: 
        #         cmns = df_lstm.index.intersection(td_idx_lstm)
        #         ax2.plot(df_lstm.loc[cmns].index, df_lstm.loc[cmns].values, c='g', alpha=0.5)
        #
        #
        #         # break
        #
        #     # ax2.set_ylim([-60, 0])
        #     ax2.grid(alpha=0.5)
        #
        #     ax2.set_ylabel('P [mm/d]', c='b')
        #     ax2.set_ylabel('Q [m/s]')
        #     ax2.set_xlabel('Time index')
        # #
        #     # plt.legend(loc=0)
        #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_lstm.png" % (catch_id)), bbox_inches='tight')
        #     plt.close()
        #
        #
        #     plt.ioff()
        #     fig, ax2 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        #     ax2.plot(df_lstm_shm.index, df_lstm_shm.values, c='gray')
        #     for td_idx_lstm_shm in datetime_idx_lstm_shm:
        #         cmns = df_lstm_shm.index.intersection(td_idx_lstm_shm) 
        #         ax2.plot(df_lstm_shm.loc[cmns].index, df_lstm_shm.loc[cmns].values, c='b', alpha=0.5)
        #
        #
        #     ax2.grid(alpha=0.5)
        #
        #     ax2.set_ylabel('P [mm/d]', c='b')
        #     ax2.set_ylabel('Q [m/s]')
        #     ax2.set_xlabel('Time index')
        # #
        #     # plt.legend(loc=0)
        #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_lstm_shm.png" % (catch_id)), bbox_inches='tight')
        #     plt.close()
        #
        #     plt.ioff()
        #     fig, ax1 = plt.subplots(1, 1, figsize=(8, 4), dpi=300)
        #
        #     # ax2 = ax1.twinx()
        #
        #
        #     # ax1.scatter(df_stn.values, df_lstm.values, label='LSTM', c='g', alpha=0.5, marker='x')
        #     # ax1.scatter(df_stn.values, df_lstm_shm.values, label='LSTM-SHM', c='b', alpha=0.5, marker='d')
        #     # ax1.scatter(df_stn.values, df_stn.values, label='Obsv', c='r', alpha=0.5, marker='.')
        #     ax1.plot(df_stn.index, df_stn.values, c='r', label='obsv', alpha=0.75)
        #     ax1.plot(df_lstm.index, df_lstm.values, c='g', label='LSTM', alpha=0.75)
        #     # ax1.plot(df_lstm_shm.index, df_lstm_shm.values, c='b', label='LSTM-SHM')
        #     ax1.grid(alpha=0.5)
        #
        #     ax1.set_ylabel('Model Q [m/s]')
        #     ax1.set_xlabel('Observation Q [m/s]')
        # #    
        #     plt.legend(loc=0)
        #     plt.savefig(os.path.join(out_save_dir, r"tstep_q_%s_obsv_models.png" % (catch_id)), bbox_inches='tight')
        #     plt.close()
            # pass
        # except Exception as msg:
            # print(msg)
            # continue 
    df_results.to_csv(os.path.join(out_save_dir, r"tstep_4q_lstm_shm.csv"))
    #===========================================================================
    # In this study, we normalized the data
    # depth between 0 and 1 by dividing the depth by half the total
    # number of points in the convex hull.
    #===========================================================================
    
    df_results.head()
    
    plt.ioff()
    
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
    for _col in df_results.columns[:3]:
        if _col == 'n_obsv':
            
            ax2.plot(df_results.index, df_results.loc[:, _col], label=_col, marker='x', c='r')
        
        else:
            ax2.plot(df_results.index, df_results.loc[:, _col], label=_col)
        
        
    ax2.grid(alpha=0.5)
    ax2.legend(loc=0, ncols=3)
    ax2.set_xticks(df_results.index)
    ax2.set_xticklabels(df_results.index, rotation=90, fontsize=10)
    ax2.set_ylabel('Number of unusual events 1<= d <=4')
    
    plt.savefig(os.path.join(out_save_dir, r"tstep_all.png"), bbox_inches='tight')
    plt.close()
    
    plt.ioff()
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 6), dpi=300)
    for _col in df_results.columns[3:5]:
        # if _col == 'n_obsv':
            
            # ax2.plot(df_results.index, df_results.loc[:, _col], label=_col, marker='x', c='r')
        
        # else:
        ax2.plot(df_results.index, 100*df_results.loc[:, _col]/df_results.iloc[:, 0], label=_col)
        
        
    ax2.grid(alpha=0.5)
    ax2.legend(loc=0, ncols=3)
    ax2.set_xticks(df_results.index)
    ax2.set_xticklabels(df_results.index, rotation=90, fontsize=10)
    ax2.set_ylabel('Number of COMMON unusual events 1<= d <=4')
    
    plt.savefig(os.path.join(out_save_dir, r"tstep_all_cmn.png"), bbox_inches='tight')
    plt.close()
    
            
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
    