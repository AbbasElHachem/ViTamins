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
import seaborn as sn
import matplotlib.dates as mdates

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)
import scipy.stats as st

cmap = plt.get_cmap('viridis_r')
cmap.set_bad("gray")


modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

def nse(predictions, targets):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))


# def depth_ftn(x, y, ei):
#
#     mins = x.shape[0] * np.ones((y.shape[0],))  # initial value
#
#     for i in ei:  # iterate over unit vectors
#         d = np.dot(i, x.T)  # scalar product
#
#         dy = np.dot(i, y.T)  # scalar product
#
#         # argsort gives the sorting indices then we used it to sort d
#         d = d[np.argsort(d)]
#
#         dy_med = np.median(dy)
#         dy = ((dy - dy_med) * (1 - (1e-10))) + dy_med
#
#         # find the index of each project y in x to preserve order
#         numl = np.searchsorted(d, dy)
#         # numl is number of points less then projected y
#         numg = d.shape[0] - numl
#
#         # find new min
#         mins = np.min(
#             np.vstack([mins, np.min(np.vstack([numl, numg]), axis=0)]), 0)
#
#     return mins
#
# ndims = 4
# ei = -1 + (2 * np.random.randn(10000, ndims))
#
# rand_pts_2 = np.random.normal(size=(100, 2))

def get_d_with_time(nyears, df, usph_vecs, n_cpus):
    df_results = pd.DataFrame(index=nyears, columns=['n_d'])
    start_year = str(nyears[0])
    for ii, _y in enumerate(nyears):
        print(_y)
        
        if ii==0:
            df_ref_set_start = df.loc[str(_y),:]
            test_vals_vals = df_ref_set_start.values.astype(float).copy('c')
            d_vals = depth_ftn_mp(test_vals_vals, test_vals_vals,
                                usph_vecs, n_cpus, 1)
            df_results.loc[_y, 'n_d'] = len(d_vals[d_vals <= 4])
            # df_results.loc[_y, 'n_d_model'] = len(d_vals[d_vals <= 4])

        if ii > 0:
            # _y = 2007
            
            df_ref_set_start = df.loc[start_year:str(_y),:]
            new_test_d = df.loc[str(_y),:]
            ref_vals_vals = df_ref_set_start.values.astype(float).copy('c')
            test_vals_vals = new_test_d.values.astype(float).copy('c')
            d_vals = depth_ftn_mp(ref_vals_vals, test_vals_vals,
                                usph_vecs, n_cpus, 1)
            print(len(ref_vals_vals), len(test_vals_vals))
            df_results.loc[_y, 'n_d'] = len(d_vals[d_vals <= 4])
        # df_results.loc[_y, 'n_d_model'] = len(d_vals[d_vals <= 4])


    return df_results
            
def main():
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    
    
    var_to_test = 'discharge_spec'
    
    path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)
    
    path_LSTM = pd.read_csv(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\data_Roberto\LSTM_discharge.csv",
                            #r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM/LSTM_discharge.csv",
                            index_col=0, parse_dates=True)
    
    path_LSTM_shm = pd.read_csv(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\data_Roberto\LSTM_SHM_discharge.csv",
        #r"https://raw.githubusercontent.com/KIT-HYD/Hy2DL/main/results/models/LSTM_SHM/LSTM_SHM_discharge.csv",
                            index_col=0, parse_dates=True)
    #===========================================================================
    # Depth func parameters
    n_vecs = int(1e4)
    n_cpus = 6
    
    
    beg_date = '1987-01-01'
    end_date = '2012-09-30'
    
    # time periods
    #
    # training_period = ['1987-10-01','1999-09-30']
    #
    # validation_period = ['1999-10-01','2004-09-30']
    #
    # testing_period = ['2004-10-01','2012-09-30']

    
    time_steps_shift = [1, 2, 3, 4]
    
    nds = len(time_steps_shift)
    date_range = pd.date_range(start=beg_date, end=end_date, freq='D')
    
    out_save_dir = r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\09_depth_month"
    
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
    
    
    usph_vecs = gen_usph_vecs_mp(n_vecs, nds, n_cpus)
    
    df_results_obsv_all = pd.DataFrame(index=range(2007, 2013, 1), columns=path_LSTM.columns)
    df_results_lstm_all = pd.DataFrame(index=range(2007, 2013, 1), columns=path_LSTM.columns)
    df_results_lstm_shm_all = pd.DataFrame(index=range(2007, 2013, 1), columns=path_LSTM.columns)
    
    df_results = pd.DataFrame(index=path_LSTM.columns, columns=['NS_LSTM_Train', 'NS_LSTM_Train_Vald', 'NS_LSTM_Test', 
                    'NS_LSTM_SHM_Train', 'NS_LSTM_SHM_Train_Vald', 'NS_LSTM_SHM_Test', 'PU_Obsv', 'PU_Lstm', 'PU_Lstm_shm'])
    for catch_id in tqdm.tqdm(path_LSTM.columns):
        
        if True:#not os.path.exists(out_save_dir / (r'%s_1d_2.png' % (catch_id))):
            print(catch_id)
            # break
            # catch_id = '27003'
            # catch_id = '12001'
            df_stn = data_hdf5.get_pandas_dataframe_between_dates(
                catch_id, event_start=beg_date, event_end=end_date)
            
            df_stn = df_stn.dropna(how='all')
            
            
            df_stn_orig = df_stn.copy(deep=True)
            df_stn_orig = df_stn_orig.loc[beg_date:end_date,:]
            
            # df_stn = df_stn.loc['2007':,:]
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
            
            nse_lstm_train = nse(df_lstm.loc['1989':'1999'].values.ravel(), df_stn.loc['1989':'1999',:].values.ravel())
            nse_lstm_shm_train = nse(df_lstm_shm.loc['1989':'1999':].values.ravel(), df_stn.loc['1989':'1999',:].values.ravel())
            
            nse_lstm_valid = nse(df_lstm.loc['1989':'2004'].values.ravel(), df_stn.loc['1989':'2004',:].values.ravel())
            nse_lstm_shm_valid = nse(df_lstm_shm.loc['1989':'2004':].values.ravel(), df_stn.loc['1989':'2004',:].values.ravel())
            
            
            
            nse_lstm_test = nse(df_lstm.loc['2005':].values.ravel(), df_stn.loc['2005':,:].values.ravel())
            nse_lstm_shm_test = nse(df_lstm_shm.loc['2005':].values.ravel(), df_stn.loc['2005':,:].values.ravel())
            
            
                        # normalize by median
            df_stn_norm_orig = df_stn / df_stn.median()
            df_lstm_norm = df_lstm / df_lstm.median()
            df_lstm_shm_norm = df_lstm_shm / df_lstm_shm.median()
            # df_stn[df_stn > 90] = np.nan
            
            df_orig_shift_all_y = pd.DataFrame(index=df_stn_orig.index, columns=time_steps_shift)
            df_orig_shift = pd.DataFrame(index=df_stn_norm_orig.index, columns=time_steps_shift)
            df_lstm_shift = pd.DataFrame(index=df_lstm_norm.index, columns=time_steps_shift)
            df_lstm_shm_shift = pd.DataFrame(index=df_lstm_shm_norm.index, columns=time_steps_shift)
            
            # nbr_idx = 0
            
            # plt.scatter(df_stn_norm_orig.values.ravel(), df_lstm_norm.values.ravel())
            # plt.show()
    
            for tshift in time_steps_shift:
                
                df_stn_norm_orig_shifted_all_data = df_stn_orig.shift(tshift)
                df_stn_norm_orig_shifted = df_stn_norm_orig.shift(tshift)
                df_lstm_norm_shifted = df_lstm_norm.shift(tshift)
                df_lstm_shm_norm_shift = df_lstm_shm_norm.shift(tshift)
                
                df_orig_shift_all_y.loc[df_stn_norm_orig_shifted_all_data.index, tshift] = df_stn_norm_orig_shifted_all_data.values.ravel()
                df_orig_shift.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
                df_lstm_shift.loc[df_lstm_norm_shifted.index, tshift] = df_lstm_norm_shifted.values.ravel()
                df_lstm_shm_shift.loc[df_lstm_shm_norm_shift.index, tshift] = df_lstm_shm_norm_shift.values.ravel()
                      
            
            df_orig_shift_all_y = df_orig_shift_all_y.dropna(axis=0)
            df_orig_shift_nonan = df_orig_shift.dropna(axis=0)
            df_lstm_shift_nonan = df_lstm_shift.dropna(axis=0)
            df_lstm_shm_shift_nonan = df_lstm_shm_shift.dropna(axis=0)
            
            
            # df_test = df_lstm_shift_nonan.copy(deep=True)
            
            # nyears_all = np.unique(df_orig_shift_all_y.index.year)
#

            nyears = np.unique(df_lstm_shift_nonan.index.year)
            
            nyears_12 = np.unique(df_lstm_shift_nonan.loc['1989':'2004'].index.year)
            nyears_3 = np.unique(df_lstm_shift_nonan.loc['2005':].index.year)

            # df_results_obsv_all_years = get_d_with_time(nyears_all, df_orig_shift_all_y, usph_vecs, n_cpus)
            
            df_orig_shift_nonan_p12 = df_orig_shift_nonan.loc['1989':'2004']
            df_lstm_shift_nonan_p12 = df_lstm_shift_nonan.loc['1989':'2004']
            df_lstm_shm_shift_nonan_p12 = df_lstm_shm_shift_nonan.loc['1989':'2004']
            #
            df_results_obsv_p12 = get_d_with_time(nyears_12, df_orig_shift_nonan_p12, usph_vecs, n_cpus)
            df_results_lstm_p12 = get_d_with_time(nyears_12, df_lstm_shift_nonan_p12, usph_vecs, n_cpus)
            df_results_lstm_shm_p12 = get_d_with_time(nyears_12, df_lstm_shm_shift_nonan_p12, usph_vecs, n_cpus)
            #
            
            df_orig_shift_nonan_p3 = df_orig_shift_nonan.loc['2005':]
            df_lstm_shift_nonan_p3 = df_lstm_shift_nonan.loc['2005':]
            df_lstm_shm_shift_nonan_p3 = df_lstm_shm_shift_nonan.loc['2005':]
            #
            
            df_results_obsv_p3 = get_d_with_time(nyears_3, df_orig_shift_nonan_p3, usph_vecs, n_cpus)
            df_results_lstm_p3 = get_d_with_time(nyears_3, df_lstm_shift_nonan_p3, usph_vecs, n_cpus)
            df_results_lstm_shm_p3 = get_d_with_time(nyears_3, df_lstm_shm_shift_nonan_p3, usph_vecs, n_cpus)
            #
            
            r3_123_obsv = sum(df_results_obsv_p3.values)/(sum(df_results_obsv_p3.values)+sum(df_results_obsv_p12.values))
            r3_123_lstm = sum(df_results_lstm_p3.values)/(sum(df_results_lstm_p3.values)+sum(df_results_lstm_p12.values))
            r3_123_lstm_shm = sum(df_results_lstm_shm_p3.values)/(sum(df_results_lstm_shm_p3.values)+sum(df_results_lstm_shm_p12.values))

            df_results.loc[catch_id, :] = [nse_lstm_train, nse_lstm_valid, nse_lstm_test, 
               nse_lstm_shm_train, nse_lstm_shm_valid, nse_lstm_shm_test, r3_123_obsv[0], r3_123_lstm[0], r3_123_lstm_shm[0]]
            
            
            # df_results_obsv_all.loc[:, catch_id] = df_results_obsv.values.ravel()
            # df_results_lstm_all.loc[:, catch_id] = df_results_lstm.values.ravel()
            # df_results_lstm_shm_all.loc[:, catch_id] = df_results_lstm_shm.values.ravel()
            #

            #
            # cols = [str(_s).split(' ')[0] for _s in df_reslt_lstm.T.columns]
            # # len(cols)
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            # # plt.ioff()
            # text_str = ('NSE Training LSTM=%0.2f - LSTM-SHM=%0.2f \n'
            # 'NSE Validation LSTM=%0.2f - LSTM-SHM=%0.2f \n'
            # 'NSE Testing LSTM=%0.2f - LSTM-SHM=%0.2f' % (nse_lstm_train, nse_lstm_shm_train,
            #                                            nse_lstm_valid, nse_lstm_shm_valid,
            #                                            nse_lstm_test, nse_lstm_shm_test))
            # plt.ioff()
            # fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=300)
            # # plt.text(0.053, 0.92, '%s' % text_str, fontsize=10,
            #           # fontweight='bold', bbox={'facecolor': 'lightgreen',  'alpha': 0.5})
            # # plt.title('- NSE LSTM=%0.2f - LSTM-SHM=%0.2f' % (nse_lstm, nse_lstm_shm))
            # ax.plot(df_results_obsv.index, df_results_lstm.cumsum(), c='b', label='Lstm', marker='x')
            # ax.plot(df_results_obsv.index, df_results_lstm_shm.cumsum(), c='g', label='Lstm-Shm', marker='o')
            # ax.plot(df_results_obsv.index, df_results_obsv.cumsum(), c='r', label='Obsv', marker='+')
            #
            # # ax.vlines(x=1989)
            # # place a text box in upper left in axes coords
            # ax.text(0.35, 0.25, text_str, transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', bbox=props)
            # plt.grid(alpha=0.5)
            # plt.legend(loc=0)
            # plt.ylabel('Number of unusual events 1<=D<=4')
            # plt.xlabel('Time')
            # plt.tight_layout()
            # plt.savefig(os.path.join(out_save_dir, "n_d_time_%s.png" % (catch_id)), bbox_inches='tight')
            # plt.close()
            #

            # plt.ioff()
            # fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=300)
            # ax.plot(df_results_obsv_all_years.index, df_results_obsv_all_years.cumsum(), c='r', label='Obsv', marker='+')
            # plt.grid(alpha=0.5)
            # plt.legend(loc=0)
            # plt.ylabel('Number of unusual events 1<=D<=4')
            # plt.xlabel('Time')
            # plt.tight_layout()
            # plt.savefig(os.path.join(out_save_dir, "n_d_time_%s_orig_data.png" % (catch_id)), bbox_inches='tight')
            # plt.close()


    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(6,4), dpi=300)
    
    # df_results.columns
    
    ax.scatter(df_results.loc[:,'NS_LSTM_Test']-df_results.loc[:,'NS_LSTM_Train_Vald'], df_results.loc[:,'PU_Lstm'])
    ax.scatter(df_results.loc[:,'NS_LSTM_SHM_Test']-df_results.loc[:,'NS_LSTM_SHM_Train_Vald'], df_results.loc[:,'PU_Lstm_shm'])

    pass
    # for _col in df_results_obsv_all.columns:
    #     max_sum = max( df_results_obsv_all.loc[:,_col].cumsum().max(), df_results_lstm_all.loc[:,_col].cumsum().max(), df_results_lstm_shm_all.loc[:,_col].cumsum().max())
    #     # break
    #     ax.plot(df_results_obsv_all.index, df_results_obsv_all.loc[:,_col].cumsum()/max_sum, c='r', alpha=0.5)
    #     ax.plot(df_results_obsv_all.index, df_results_lstm_all.loc[:,_col].cumsum()/max_sum, c='b', alpha=0.5)
    #     ax.plot(df_results_obsv_all.index, df_results_lstm_shm_all.loc[:,_col].cumsum()/max_sum, c='g', alpha=0.5)
    # ax.plot(df_results_obsv.index, df_results_obsv_all.loc[:,_col].cumsum()/max_sum, c='r', label='Obsv', marker='+')
    # ax.plot(df_results_obsv.index,  df_results_lstm_all.loc[:,_col].cumsum()/max_sum, c='b', label='Lstm', marker='x')
    # ax.plot(df_results_obsv.index, df_results_lstm_shm_all.loc[:,_col].cumsum()/max_sum, c='g', label='Lstm-Shm', marker='o')
    #
    #
    plt.grid(alpha=0.5)
    plt.legend(loc=0)
    plt.ylabel('Ratio unusual events: Testing / (Training + Vaidation)')
    plt.xlabel('Difference NSE Testing - NSE Training + Validation')
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, "n_d_ratio.png"), bbox_inches='tight')
    plt.close()
            
if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    