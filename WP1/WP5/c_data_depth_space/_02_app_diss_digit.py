'''
@author: Faizan-Uni-Stuttgart
@author: Abbas El Hachem

Perform the appearing and disappearing events analysis
Using the Tukey's depth function and dividing the input data into
windows (N events per window), this class computes ratios of events
that have appeared or disappeared for any two given time windows (with
respect to the test window).

The time window can be a set of consecutive years or months or steps.
Events in test window are checked for containment inside the
reference window. Points that have a depth of zero in the reference
window are considered disappearing if the reference window is ahead
of the test window in steps and appearing if vice versa.

For example, consider a dataset of 200 time steps (rows) and 2
stations (columns). First 100 time steps are set as reference and the
others as the test window. Using the Tukey's (or any) depth function,
depth for each point of the test window in the reference window is
computed. Tukey's depth funtion returns a zero for any point that is
outside the convex hull (created by the points in the reference
dataset). It returns a one if a point lies on the convex
hull. Let's say 10 points' depth are zero. So for this specific case,
we have 10 appearing situations which is ten percent of the test
window. This is the main output of this analysis. Based on the
specified parameters, other outputs are also computed.
Read the entire documentation for more information.
   

'''
import sys
import os
import timeit
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from appdis import (
    AppearDisappearAnalysis,
    AppearDisappearPlot,
    cnvt_ser_to_mult_dims_df)

from _01_1_transf_pcp_to_digit import tranform_num_to_digit

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

np.set_printoptions(
    precision=3,
    threshold=2000,
    linewidth=200000,
    formatter={'float': '{:+0.3f}'.format})

pd.options.display.precision = 3
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 250

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)


from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf


def main():

    radar_loc = 'Hannover'  # Hannover  Tuerkheim  Feldberg

    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        % radar_loc)

    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        # r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        r"\%s_dwd_stns_1440min_1880_2019.h5"
        % radar_loc)

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    n_vecs = int(1e7)

    data_freq = '1440min'

    # in_refr_var_file = (
    #     r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
    #     r"\1440min_data\P02290_1440min_data_1781_2019.csv")

    # in_refr_var_file = (
    #     r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD\60min_data\P00020_60min_data_2004_2019.csv")
    # in_test_var_file = main_dir / r'valid_kfold_01__cats_outflow.csv'
    thr = 3
    n_uvecs = int(1e5)
    n_cpus = 5  # 'auto'
    n_dims = 6
    ws = 20  # int(20 * 365 // n_dims)  # window size
    analysis_style = 'un_peel'
    time_win_type = 'year'  # 'month'
    n_ticks = 10
    cmap = 'jet'  # 'viridis'
    time_unit_step_size = int(365 // n_dims)

    sep = ';'
    time_fmt = '%Y-%m-%d'
    beg_date = '1900-01-01'
    end_date = '2019-12-31'

    peel_depth = 1  # greater than this are kept
    n_boots = 0
    nv_boots = 0
    hdf_flush_flag = 1
    vol_data_lev = 1  # 1
    loo_flag = True
    max_allowed_corr = 0.5
    app_dis_cb_max = 1  # 10

    sel_idxs_flag = False
    take_rest_flag = False
    ann_flag = False
    plot_flag = False

    # sel_idxs_flag = False
    # take_rest_flag = False
    ann_flag = True
    plot_flag = True

    if sel_idxs_flag:
        sel_idxs_lab = '_sel_idxs'

    else:
        sel_idxs_lab = ''

    if take_rest_flag:
        rest_lab = '_rest'

    else:
        rest_lab = ''

    if analysis_style == 'un_peel':
        peel_depth = 0

    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\app_diss_stns_%s_daily_transf"
        % analysis_style)

    # out_save_dir = (
    # r"X:\exchange\ElHachem\2021_09_13_Steinbruch\Depth_func20")
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)

    # wanted_ids = ['P00069', 'P00277', 'P00587', 'P03269']
    for _dwd_id in dwd_ids[:10]:
        try:
            print(_dwd_id)
        # break
            df_stn = dwd_hdf5.get_pandas_dataframe_between_dates(
                _dwd_id, event_start=beg_date,
                event_end=end_date)
            df_low_freq = resampleDf(df_stn, data_freq)

            df_transf = tranform_num_to_digit(df=df_low_freq,
                                              pcp_thr=thr,
                                              nbr_days=6)
            df_transf.columns = [_dwd_id]
            if np.unique(df_transf.index.year).size >= 20:
                out_dir = os.path.join(out_save_dir, '%s_2' % _dwd_id)

                print('out_dir:', out_dir)

                hdf5_path = Path(out_dir) / 'app_dis_ds.hdf5'

                if ann_flag:
                    #         with open(in_var_file, 'rb') as _hdl:
                    #             in_var_dict = pickle.load(_hdl)
                    #             in_anom_df = in_var_dict['anomaly_var_df'].iloc[:steps]
                    #
                    #             if sel_idxs_flag:
                    #                 tot_in_var_arr = in_anom_df.values.copy('c')
                    #
                    #             else:
                    #                 tot_in_var_arr = in_var_dict['pcs_arr'][:steps, :].copy('c')
                    #
                    #                 if take_rest_flag:
                    #                     rest_arr = tot_in_var_arr[:, n_dims - 1:]
                    #                     rest_arr = (rest_arr ** 2).sum(axis=1) ** 0.5
                    #                     rest_arr = rest_arr.reshape(-1, 1)
                    #
                    #                     tot_in_var_arr = np.hstack(
                    #                         (tot_in_var_arr[:, :n_dims - 1], rest_arr))
                    #
                    #                     assert tot_in_var_arr.shape[1] == n_dims
                    #
                    #             time_idx = in_anom_df.index
                    #             del in_var_dict, in_anom_df

                    # in_refr_df = pd.read_csv(in_refr_var_file, index_col=0, sep=sep,
                    #                          engine='c', low_memory=True)
                    # in_refr_df.index = pd.to_datetime(in_refr_df.index, format=time_fmt)

                    in_refr_df = df_transf.astype(float)
                    in_refr_ser = in_refr_df.loc[beg_date:end_date, str(
                        _dwd_id)]

                    res_refr_df = cnvt_ser_to_mult_dims_df(in_refr_ser, n_dims)

                    tot_refr_var_arr = res_refr_df.values.copy('c')
                    time_idx = res_refr_df.index
            #         time_idx = np.arange(0, tot_refr_var_arr.shape[0]).astype(np.int64)

                    # in_test_df = pd.read_csv(in_test_var_file, index_col=0, sep=sep)
                    in_test_df = in_refr_df.copy()
                    in_test_df.index = pd.to_datetime(
                        in_test_df.index, format=time_fmt)
                    in_test_ser = in_test_df.loc[beg_date:end_date, str(
                        _dwd_id)]

                    res_test_df = cnvt_ser_to_mult_dims_df(in_test_ser, n_dims)
                    tot_test_var_arr = res_test_df.values.copy('c')
                    # n_dims = tot_test_var_arr.shape[1]
                    #
                    # usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)
                    #
                    # depths1 = depth_ftn_mp(
                    #     tot_test_var_arr,
                    #     tot_test_var_arr,
                    #     usph_vecs, n_cpus, 1)

                    # depths1.shape
                    #
                    # tot_test_var_arr.shape
                    #
                    # import matplotlib.pyplot as plt
                    # plt.ioff()
                    # plt.scatter(res_test_df.sum(axis=1),
                    #             depths1)
                    # plt.show()
                    ad_ans = AppearDisappearAnalysis()
                    ad_ans.set_data_arrays(
                        tot_refr_var_arr, tot_test_var_arr, False)
                    ad_ans.set_time_index(time_idx)
                    ad_ans.generate_and_set_unit_vectors(
                        n_dims, n_uvecs, n_cpus)

                    ad_ans.set_analysis_parameters(
                        time_win_type,
                        ws,
                        analysis_style,
                        peel_depth,
                        n_cpus,
                        time_unit_step_size)

                    if sel_idxs_flag:
                        ad_ans.set_optimization_parameters(
                            0.85,
                            0.95,
                            150,
                            20000,
                            5000,
                            max_allowed_corr)

                    ad_ans.set_boot_strap_on_off(n_boots)
                    ad_ans.set_volume_boot_strap_on_off(nv_boots)
                    ad_ans.set_outputs_directory(out_dir)
                    ad_ans.save_outputs_to_hdf5_on_off(True, hdf_flush_flag)

                    ad_ans.save_volume_data_level(vol_data_lev, loo_flag)

                    ad_ans.verify()

            #         ad_ans.resume_from_hdf5(hdf5_path)

                    ad_ans.cmpt_appear_disappear()
                    ad_ans.terminate_analysis()

                if plot_flag:
                    ad_plot = AppearDisappearPlot()
                    ad_plot.set_hdf5(hdf5_path)

                    # import h5py
                    #
                    #
                    # with h5py.File(hdf5_path, "r") as f:
                    #     # List all groups
                    #     print("Keys: %s" % f.keys())
                    #     a_group_key = list(f.keys())[0]
                    #
                    #     # Get the data
                    #     data = list(f[a_group_key])

                    ad_plot.set_outputs_directory(out_dir)
                    ad_plot.set_fig_props(n_ticks, cmap, app_dis_cb_max)
                    ad_plot.verify()

                    ad_plot.plot_app_dis()

                    ad_plot.plot_ans_dims()

                    print('done plotting')
                    if sel_idxs_flag:
                        ad_plot.plot_sim_anneal_opt()

                    if vol_data_lev:
                        ad_plot.plot_volumes()
        except Exception:
            print('Error')
            continue
            # ad_plot.plot_ecops()
    return


if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
