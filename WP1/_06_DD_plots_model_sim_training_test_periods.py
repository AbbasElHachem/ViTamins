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

from appdis import (AppearDisappearAnalysis, AppearDisappearPlot,
    cnvt_ser_to_mult_dims_df)


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
            
            df_stn = data_hdf5.get_pandas_dataframe_between_dates(
                catch_id, event_start=beg_date, event_end=end_date)
            df_stn = df_stn.dropna(how='all')
            
            # path_disch_data = pd.read_csv(
            #     r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\data_Roberto\CAMELS-GB\timeseries_v2\CAMELS_GB_hydromet_timeseries_%s.csv" % catch_id,
            #     index_col=0, parse_dates=True, sep=',')
            #
            # df_q = path_disch_data.loc[:, 'discharge_spec']
            
            # cmn_idx = df_q.index.intersection(df_stn.index)
            
            # plt.ioff()
            # plt.scatter(df_q.loc[cmn_idx].values.ravel(),
            # df_stn.loc[cmn_idx].values.ravel())
            # plt.show()
            
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
            
            month_vals =df_lstm_shift_nonan.resample('M').sum().index
            usph_vecs = gen_usph_vecs_mp(n_vecs, df_lstm_shm_shift_nonan.columns.size, n_cpus)
            
            def calc_depth_of_array_in_data(df_, month_vals):
                df_reslt = pd.DataFrame(index=month_vals, columns=range(31))
                for _m in month_vals:
                    print(_m)
                    start_tv = _m - pd.Timedelta(days=30)
                    test_vals = df_.loc[start_tv:_m, :]
                    test_vals_vals = test_vals.values.astype(float).copy('c')
                    ref_vals = df_.loc[df_.index.difference(test_vals.index),:]
                    ref_vals_vals = ref_vals.values.astype(float).copy('c')
                    
                    depths_da_osv_osv = depth_ftn_mp(ref_vals_vals, test_vals_vals,
                                            usph_vecs, n_cpus, 1)
                    
                    df_reslt.loc[_m, :len(depths_da_osv_osv)-1] = depths_da_osv_osv
                    # depths_da_osv_osv = depth_ftn_mp(test_vals_vals, ref_vals_vals,
                                            # usph_vecs, n_cpus, 1)
                print('done')
                return df_reslt
            
            df_reslt_obsv = calc_depth_of_array_in_data(df_=df_orig_shift_nonan, month_vals=month_vals)
            df_reslt_lstm = calc_depth_of_array_in_data(df_=df_lstm_shift_nonan, month_vals=month_vals)
            df_reslt_lstm_shm = calc_depth_of_array_in_data(df_=df_lstm_shm_shift_nonan, month_vals=month_vals)
            
            
            plt.ioff()
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8), dpi=300, sharey=False, sharex=False)
            

            # ax1.scatter(depths_da_da_low.values.ravel(), df_lstm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='r', edgecolor='darkred', marker='X', label='low-D')
            ax1.imshow(np.array(df_reslt_obsv.T.values.tolist()).astype('float'), vmin=0, vmax=10, cmap=plt.get_cmap('viridis_r'))
    
            ax1.grid(alpha=0.5)
            # ax3.legend(loc=0)
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir, r"month_q_obsv_%s.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            
            plt.ioff()
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8), dpi=300, sharey=False, sharex=False)
            
            # ax1.scatter(depths_da_da_low.values.ravel(), df_lstm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='r', edgecolor='darkred', marker='X', label='low-D')
            ax1.imshow(np.array(df_reslt_lstm.T.values.tolist()).astype('float'), vmin=0, vmax=10, cmap=plt.get_cmap('viridis_r'))

            ax1.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir, r"month_q_lstm_%s.png" % (catch_id)), bbox_inches='tight')
            plt.close()


            plt.ioff()
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8), dpi=300, sharey=False, sharex=False)

            # ax1.scatter(depths_da_da_low.values.ravel(), df_lstm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='r', edgecolor='darkred', marker='X', label='low-D')
            ax1.imshow(np.array(df_reslt_lstm_shm.T.values.tolist()).astype('float'), vmin=0, vmax=10, cmap=plt.get_cmap('viridis_r'))

            ax1.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir, r"month_q_lstm_shm_%s.png" % (catch_id)), bbox_inches='tight')
            plt.close()

            plt.ioff()
            fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8), dpi=300, sharey=False, sharex=False)

            # ax1.scatter(depths_da_da_low.values.ravel(), df_lstm_shift_nonan.loc[depths_da_da_low.index, 'd'], facecolor='r', edgecolor='darkred', marker='X', label='low-D')
            ax1.imshow(np.array((df_reslt_obsv-df_reslt_lstm_shm).T.values.tolist()).astype('float'), vmin=-10, vmax=10, cmap=plt.get_cmap('viridis_r'))

            ax1.grid(alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir, r"month_q_obsv_lstm_shm_%s.png" % (catch_id)), bbox_inches='tight')
            plt.close()
            
            #===================================================================


if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    