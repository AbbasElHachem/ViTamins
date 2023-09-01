'''
@author: Faizan-Uni-Stuttgart

Mar 3, 2021

1:37:50 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path
import tqdm
import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf

DEBUG_FLAG = True


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


def main():

    # =============================================================

    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\05_space")

    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5"
        # r"\dwd_comb_1440min_gk3.h5"
    )
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_1min_data_agg_60min_2020_utm32.h5")

    beg_time = '1995-01-01 00:00:00'
    end_time = '2019-12-31 23:55:00'
    freq = '60min'
    min_nbr_stns = 50
    min_thr_val = 3
    sum_over_stns = 80
    path_to_intense_events = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis\00_events_select"
        r"\select_events_thr_based\Hannover\rem_timesteps_60min_Hannover_thr_6.00_.csv")

    time_fmt = '%Y-%m-%d %H:%M:%S'
    sep = ';'

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()
    df_coords = dwd_hdf5.get_coordinates(dwd_ids)
    in_df_coords_utm32 = pd.DataFrame(
        index=dwd_ids,
        data=df_coords['easting'], columns=['X'])
    y_dwd_coords = df_coords['northing']
    in_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords
    in_df_coords_utm32.drop_duplicates(keep='first', inplace=True)
    in_df_coords_utm32.loc[:, 'lon'] = df_coords['lon']
    in_df_coords_utm32.loc[:, 'lat'] = df_coords['lat']

    # dd = dwd_hdf5.get_pandas_dataframe(dwd_ids[0]).dropna()
    # np.percentile(dd.values, 94)
    # stuff for depth ftn
    ndims = 6
    # ei = -1 + (2 * np.random.randn(100000000, ndims))
    # normc = np.sqrt((ei ** 2).sum(axis=1))
    # norm_idxs = normc < 1.0
    # normc = normc[norm_idxs]
    # print('final ei shape:', normc.shape)
    # ei = ei[norm_idxs] / normc[:, None]
    # dwd_ids = dwd_ids[:1000]

    df_intense_events = pd.read_csv(
        path_to_intense_events, sep=sep, index_col=0,
        engine='c')
    df_intense_events.index = pd.to_datetime(
        df_intense_events.index, format=time_fmt)
    dwd_ids_hannover = df_intense_events.columns.to_list()

    pcp_df_all = dwd_hdf5.get_pandas_dataframe(
        dwd_ids_hannover)

    # get nbr of stns vs sum over stns
    # idx_abv_thr = np.where(pcp_df_all > 0)

    # pcp_thrs = [0.1, 1, 3, 5, 10]
    # pcp_df_all_2000 = pcp_df_all.loc['2000':, :]
    # for _thr in pcp_thrs:
    #     dates_sum_stns = pcp_df_all_2000.iloc[
    #         np.where(pcp_df_all_2000.sum(axis=1) >
    #                  _thr)[0], :].index
    #
    #     df_nbr_stns = pd.DataFrame(index=dates_sum_stns)
    #     for ii, _ix in enumerate(dates_sum_stns):
    #         print(ii, len(dates_sum_stns))
    #         df_nbr_stns.loc[_ix, 'Nbr stns'] = (np.where(
    #             pcp_df_all_2000.loc[_ix, :].dropna().values >= _thr)[0].size)
    #     # df_nbr_stns.plot()
    #     # plt.show()
    #     df_nbr_stns = df_nbr_stns[df_nbr_stns.values > 0]
    #     sum_stns = pcp_df_all_2000.loc[df_nbr_stns.index, :].sum(axis=1)
    #     # plot nbr of stns vs sum
    #     plt.ioff()
    #     plt.figure(figsize=(12, 8), dpi=200)
    #     plt.scatter(df_nbr_stns.values.ravel(),
    #                 sum_stns.values.ravel(),
    #                 c='b',
    #                 marker='o',
    #                 alpha=.75)
    #     plt.xlabel('Number of stations with pcp > %d mm/%s'
    #                % (_thr, freq), fontsize=16)
    #     plt.ylabel('Sum over radar area of Hannover',
    #                fontsize=16)
    #     plt.xlim([0, max(df_nbr_stns.values) + 1])
    #     plt.xticks(np.unique(df_nbr_stns.values))
    #     plt.grid(alpha=.25)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(
    #         out_save_dir, 'sum_vs_nbr_stns_%d_%s.png'
    #         % (_thr, freq)))
    #     plt.close()
    pcp_df_all_binary = pcp_df_all.copy()

    idx_abv_thr = np.where(pcp_df_all_binary >= min_thr_val)
    # pcp_df_all_binary.iloc[idx_abv_thr[0],:] = 1
    # pcp_df_all_binary[pcp_df_all_binary >= min_thr_val] = 1
    # pcp_df_all_binary[pcp_df_all_binary < min_thr_val] = 0
    # dates_many_stns = df_intense_events.iloc[
    #     np.where(df_intense_events.sum(axis=1) >
    #              min_nbr_stns)[0], :].index

    dates_many_stns = pcp_df_all.iloc[
        np.where(pcp_df_all.sum(axis=1) >
                 sum_over_stns)[0], :].index

    pcp_over_stns = pcp_df_all.loc[dates_many_stns, :].sum(axis=1)
    df_events = pd.DataFrame(index=dates_many_stns,
                             columns=dwd_ids_hannover,
                             data=pcp_df_all.loc[dates_many_stns, :].values)

    df_events = df_events.loc['2000':, :]
    df_events.plot(legend=False)
    # for _idx in df_events.index:
    #     # pcp_event = dwd_hdf5.get_pandas_dataframe_for_date(
    #         # dwd_ids_hannover, _idx).dropna()
    #     pcp_event = pcp_df_all.loc[_idx, :].dropna()
    #
    #     df_events.loc[_idx, pcp_event.index  # columns
    #                   ] = pcp_event.values.ravel()

    df_events_10 = df_events.dropna(how='any', axis=1)
    # df_events_10 = df_events_10**0.097
    df_events_10_sum = df_events_10.sum(axis=1)
    # df_events_10_sum_col = df_events_10.sum(axis=0)

    # pcp_df_all_nonan = pcp_df_all.fillna(0).loc['2000':, :]

    # df_events_all = pcp_df_all_nonan.T

    # corr_mat_all = np.corrcoef(df_events_all.values)

    df_events_nan = df_events_10.T

    corr_mat = np.corrcoef(df_events_nan.values)

    # corr_mat.shape

    '''
    x : array_like
        A 1-D or 2-D array containing multiple variables and observations.
        Each row of `x` represents a variable, and each column a single
        observation of all those variables. Also see `rowvar` below.
    
    '''
    eig_val, eig_mat = np.linalg.eig(corr_mat)
    sort_idxs = np.argsort(eig_val)
    eig_val_sort = eig_val[sort_idxs]
    eig_val_sum = eig_val_sort.sum()
    eig_val_cum_sum = np.cumsum(eig_val_sort[::-1]) / eig_val_sum
    print(eig_val_cum_sum)
    df_events_nan.values.shape, eig_mat.shape
    b_j_s = np.dot(df_events_nan.T.values, eig_mat)

    xs = b_j_s[:, :ndims].copy(order='c')
    xs.shape
    # ys = b_j_s[:, :ndims][curr_ppt_arr_thresh_idxs, :]

    # df_events.iloc[0,:].dropna()

    n_vecs = int(1e6)
    n_cpus = 8

    n_dims = xs.shape[1]
    #
    # n_dims = 1
    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)
    depths_all = depth_ftn_mp(
        xs, xs, usph_vecs, n_cpus, 1)

    print('donce calculating depth')
    # df_events_10_sum.plot()
    # df_events_10_sum.iloc[9]

    plt.ioff()

    plt.figure(figsize=(12, 8))
    # plt.plot(b_j_s[:, 0], alpha=0.7)
    plt.scatter(xs[:, 0],
                xs[:, 1], c=depths_all,
                cmap=plt.get_cmap('jet_r'),
                alpha=0.83, label='all')
    plt.axis('equal')
    plt.show()

    plt.ioff()

    plt.figure(figsize=(12, 8))
    # plt.plot(b_j_s[:, 0], alpha=0.7)
    plt.scatter(df_events_10_sum.values,
                depths_all, c=depths_all,
                cmap=plt.get_cmap('jet_r'),
                alpha=0.83, label='all')
    # plt.axis('equal')
    plt.show()

    df_nbr_stns = pd.DataFrame(index=df_events_10.index)
    for _ix in df_events_10_sum.index:

        df_nbr_stns.loc[_ix, 'Nbr stns'] = (np.where(
            df_events_10.loc[_ix, :].values > 0)[0].size)
        # break
    plt.ioff()

    plt.figure(figsize=(12, 8))
    # plt.plot(b_j_s[:, 0], alpha=0.7)
    plt.scatter(df_nbr_stns.values,
                depths_all, c=depths_all,
                cmap=plt.get_cmap('jet_r'),
                alpha=0.83, label='all')
    # plt.axis('equal')
    plt.show()

    print(depths_all.shape)

    pca = PCA(n_components=0.9, svd_solver='full')
    X = df_events_nan  # .iloc[0,:]
    X_r = pca.fit(X).transform(X)

    # from scipy.stats import pearsonr
    #
    # pearsonr(pcp_data.iloc[:, 0], pcp_data.iloc[:, 1])
    # #corr_mtx = np.empty()
    X_r.shape

    # pca.explained_variance_ratio_
    print("Components = ", pca.n_components_,
          ";\nTotal explained variance = ",
          round(pca.explained_variance_ratio_.sum(), 5))

    pass
    #

    # bound_ppt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # interval_ppt = np.linspace(0.1, 0.95)
    # colors_ppt = plt.get_cmap('YlGnBu')(interval_ppt)
    # cmap_ppt = LinearSegmentedColormap.from_list('name', colors_ppt)
    # cmap_ppt.set_over('m')
    # cmap_ppt.set_under('lightgrey')  # lightgrey
    # norm_ppt = mcolors.BoundaryNorm(bound_ppt, cmap_ppt.N)

    #
    #
    # plt.ioff()
    # plt.figure(figsize=(12, 8), dpi=200)
    # for _evt in dates_many_stns:
    #
    #     plt.plot(range(len(data_df_nonan.columns)),
    #              data_df_nonan.loc[_evt, :],
    #              alpha=0.75)
    #
    # plt.xlabel('Stn ID', fontsize=14)
    # plt.ylabel('Pcp [mm/%s]' % freq, fontsize=14)
    # plt.grid(alpha=0.25)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_save_dir, 'events_%s.png' % freq))
    # plt.close()
    #
    # plt.ioff()
    # plt.figure(figsize=(12, 8), dpi=200)
    #
    # plt.plot(sum_events.index, sum_events.values, c='b')
    # plt.scatter(sum_events.index, sum_events.values, marker='o', c='b')
    # plt.xlabel('Event ID', fontsize=14)
    # plt.ylabel('Pcp [mm/%s]' % freq, fontsize=14)
    # plt.grid(alpha=0.75)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_save_dir, 'sum_events_%s.png' % freq))
    # plt.close()
    #
    # assert np.all(np.isfinite(data_df_nonan.values))

    # tst_times_dict = {}  # _id:[] for _id in dwd_ids}
    # for _stn in dwd_ids:
    #     if _stn in df_intense_events.columns:
    #         stns_events = df_intense_events.loc[
    #             :, _stn].dropna(how='all')
    #         if stns_events.index.size > 0:
    #             tst_times_dict[_stn] = stns_events.index.intersection(
    #                 data_df_nonan.index)
    #
    # _stns_with_evts = data_df_nonan.columns.intersection(
    #     list(tst_times_dict.keys()))
    # data_df_nonan = data_df_nonan.loc[:, _stns_with_evts]
    # print(data_df_nonan)
    # print(tst_times_dict)

    # assert data_df.shape[1] == len(tst_times_dict)

    #
    # ref_pts = data_df_nonan.values.copy(order='c')
    # ref_pts.shape
    # usph_vecs.shape
    # tst_depths_dict = {}

    # for tst_stn in tst_times_dict:

    # tst_stn in data_df.columns

    # test_pts = data_df_nonan.loc[
    # tst_times_dict[tst_stn], :].values.copy(order='c')
    # test_pts = data_df_nonan.loc[dates_many_stns, :].values.copy(order='c')

    # test_pts.shape
    # ref_pts_recent = data_df_nonan.loc['2010':, :].values.copy(order='c')

    # print(depths_all.shape)
    #
    # data_ser_depth_all = pd.DataFrame(index=data_df_nonan.index,
    #                                   data=depths_all)
    # depths_all_high_idx = np.where(depths_all <= 10)[0]
    # depths_all_high = depths_all[depths_all <= 10]
    #
    # plt.ioff()
    # plt.figure(figsize=(16, 8), dpi=200)
    # plt.scatter(range(len(depths_all)), depths_all, label='d>10', c='b')
    # plt.scatter(depths_all_high_idx, depths_all_high, c='r', label='d<=10')
    # plt.xlabel('Hourly time steps 1995-2019', fontsize=16)
    # plt.ylabel('Depth d', fontsize=16)
    # plt.legend(loc=0)
    # plt.grid(alpha=0.25)
    # xticks = int(len(data_df_nonan.index) / (24 * 365))
    # plt.xticks(np.linspace(0, len(data_df_nonan.index), 25),
    #            labels=np.unique(data_df_nonan.index.year))
    # plt.ylim([0, 100])
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_save_dir, r'hourly_events_depth.png'))
    # plt.close()
    #
    # plt.ioff()
    # plt.figure(figsize=(16, 8), dpi=200)
    # for dval in range(1, 11):
    #     idx_unusual_events = data_ser_depth_all[
    #         data_ser_depth_all <= dval].dropna()
    #     idx_unusual_events_daily = idx_unusual_events.groupby(
    #         idx_unusual_events.index.year).sum()
    #
    #     plt.plot(
    #         idx_unusual_events_daily.index,
    #         idx_unusual_events_daily.values,
    #         label='d<=%d' % dval)
    #     plt.scatter(
    #         idx_unusual_events_daily.index,
    #         idx_unusual_events_daily.values,
    #         marker='d')
    #
    # plt.xlabel('Year', fontsize=16)
    # plt.ylabel('Sum of unusual hours per year', fontsize=16)
    # plt.legend(loc=0)
    # plt.grid(alpha=0.25)
    #
    # plt.xticks(idx_unusual_events_daily.index,
    #            labels=np.unique(data_df_nonan.index.year))
    # plt.ylim([0, max(idx_unusual_events_daily.values) + 5])
    # plt.yticks([0, 10, 20, 50, 100, 200, 300, 400, 500])
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_save_dir, r'sum_hourly_events_depth5.png'))
    # plt.close()
    #
    # timesteps_event = data_df_nonan.iloc[depths_all_high_idx, :]
    #
    # # timesteps_event.T.describe()
    # for _ii, _dd in zip(timesteps_event.index, depths_all_high):
    #
    #     df_nonan = timesteps_event.loc[_ii, :].dropna(how='all')
    #
    #     xvals = in_df_coords_utm32.loc[df_nonan.index, 'X'].values.ravel()
    #     yvals = in_df_coords_utm32.loc[df_nonan.index, 'Y'].values.ravel()
    #
    #     fig, (ax) = plt.subplots(
    #         1, 1, figsize=(12, 8), dpi=100)
    #     im0 = ax.scatter(xvals, yvals, c=df_nonan.values,
    #                      marker='x',
    #                      cmap=cmap_ppt)
    #
    #     ax.set(xlabel='X [m]', ylabel='Y [m]',
    #            title='%s - Depth D=%d' % (_ii, _dd))
    #
    #     # IMPORTANT ANIMATION CODE HERE
    #     # Used to keep the limits constant
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     ax.set(xticks=[], yticks=[])
    #     ax.grid(alpha=0.25)
    #     for i, txt in enumerate(df_nonan.values):
    #         if txt > 0.2:
    #             ax.annotate(round(txt, 1), (xvals[i],
    #                                         yvals[i]),
    #                         color='b')
    #
    #         elif txt < 0.2:
    #             ax.annotate(round(txt, 1), (xvals[i],
    #                                         yvals[i]),
    #                         color='grey')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_save_dir, r'hourly_%s.png'
    #                              % str(_ii).replace(':', '_')))
    #     plt.close()
    #     break
    #
    # for dval in range(1, 11):
    #
    #     plt.ioff()
    #     fig, (ax1, ax2) = plt.subplots(
    #         1, 2, figsize=(12, 8), dpi=100)
    #
    #     df_depth_val = data_ser_depth_all[
    #         data_ser_depth_all.values == dval]
    #     # max_sum_ = max(timesteps_event.loc[df_depth_val.index, :].sum(axis=1))
    #
    #     for time_idx in df_depth_val.index:
    #         df_nonan = timesteps_event.loc[time_idx, :].dropna(how='all')
    #
    #         cum_pcp_event = np.cumsum(df_nonan.values)
    #         sum_pcp_event = df_nonan.values.sum()
    #         x_lorenz = cum_pcp_event / sum_pcp_event
    #         x_lorenz = np.insert(x_lorenz, 0, 0)
    #
    #         # df_nonan_pos = df_nonan[df_nonan > 0]
    #
    #         # from statsmodels import distributions
    #         # ax.scatter(np.arange(x_lorenz.size) / (
    #         #     x_lorenz.size - 1), x_lorenz,
    #         #     marker='x')
    #         ax1.plot(np.arange(x_lorenz.size) / (
    #             x_lorenz.size - 1), x_lorenz, alpha=0.5)
    #
    #         # break
    #     # line plot of equality
    #     ax1.plot([0, 1], [0, 1], color='k')
    #     ax1.set(xlabel='Station ID', ylabel='Pcp contribution',
    #             title='Depth D=%d' % (dval))
    #
    #     # ax.set_ylim(0, 1.01)
    #     # ax.set(xticks=[], yticks=np.unique([cum_pcp_event]))
    #     ax1.grid(alpha=0.25)
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(out_save_dir, r'_lorenzhourly_%d.png'
    #                              % dval))
    #     plt.close()
    #     break
    #
    # depths_all_high.shape
    #
    # # plt.scatter(range(len(depths_all)), depths_all)
    # depths = depth_ftn_mp(ref_pts, test_pts, usph_vecs, n_cpus, 1)
    #
    # depths.shape
    # print('done calculating depth values')
    #
    # depths_ser = pd.Series(
    #     # index=tst_times_dict[tst_stn],
    #     index=dates_many_stns,
    #     data=depths)
    #
    # depths_all_high_idx = np.where(depths_ser <= 10)[0]
    # depths_all_high = depths_ser[depths_ser <= 10]
    # print('\n')
    # # break
    #
    # plt.ioff()
    # plt.figure(figsize=(12, 8), dpi=200)
    #
    # plt.plot(depths_ser.index, depths_ser.values, c='r')
    # plt.scatter(depths_ser.index, depths_ser.values, marker='o', c='r')
    # plt.xlabel('Timestep', fontsize=14)
    # plt.ylabel('Depth d', fontsize=14)
    # plt.grid(alpha=0.75)
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_save_dir, 'depth_events_%s.png' % freq))
    # plt.close()
    #
    # plt.ioff()
    # plt.figure(figsize=(12, 8), dpi=200)
    #
    # plt.scatter(sum_events.values, depths_ser.values, marker='X', c='g', s=30)
    # plt.xlabel('Sum event [mm/%s]' % freq, fontsize=14)
    # plt.ylabel('Depth d', fontsize=14)
    # plt.grid(alpha=0.75)
    # plt.tight_layout()
    # plt.savefig(os.path.join(
    #     out_save_dir, 'scatter_depth_events_%s.png' % freq))
    # plt.close()
    #
    # # TODO: comtine here
    # # depths_ser.loc['2019']
    #
    # depths_ser_d3 = depths_ser[depths_ser == 1]
    # depths_ser_d10 = depths_ser[depths_ser == 8]
    #
    # vals_d3 = data_df_nonan.loc[depths_ser_d3.index, :]
    #
    # # from pykrige import OrdinaryKriging as OK
    #
    # for event_idx in depths_ser_d10.index:
    #     depth_val = depths_ser.loc[event_idx]
    #     event_start = event_idx - pd.Timedelta(minutes=int(freq.split('m')[0]))
    #
    #     df = dwd_hdf5.get_pandas_dataframe_between_dates(
    #         dwd_ids,
    #         event_start=event_start,
    #         event_end=event_idx)
    #     df_nonan = df.dropna(how='all', axis=1)
    #
    #     xvals = in_df_coords_utm32.loc[df_nonan.columns, 'X'].values.ravel()
    #     yvals = in_df_coords_utm32.loc[df_nonan.columns, 'Y'].values.ravel()
    #     all_images = []
    #
    #     for ii, _evt in enumerate(df_nonan.index):
    #         pcp_vals1 = df_nonan.iloc[ii, :].values
    #         plt.ioff()
    #         fig, (ax) = plt.subplots(
    #             1, 1, figsize=(12, 8), dpi=100)
    #         im0 = ax.scatter(xvals, yvals, c=pcp_vals1,
    #                          marker='x',
    #                          cmap=cmap_ppt)
    #
    #         ax.set(xlabel='X [m]', ylabel='Y [m]',
    #                title='%s - Depth D=%d' % (_evt, depth_val))
    #
    #         # IMPORTANT ANIMATION CODE HERE
    #         # Used to keep the limits constant
    #         ax.set_xlim(xmin, xmax)
    #         ax.set_ylim(ymin, ymax)
    #         ax.set(xticks=[], yticks=[])
    #         ax.grid(alpha=0.25)
    #         for i, txt in enumerate(df_nonan.iloc[ii, :].values):
    #             if txt > 0.2:
    #                 ax.annotate(round(txt, 1), (xvals[i],
    #                                             yvals[i]),
    #                             color='b')
    #
    #             elif txt < 0.2:
    #                 ax.annotate(round(txt, 1), (xvals[i],
    #                                             yvals[i]),
    #                             color='grey')
    #         # plt.colorbar(im0, ax=ax,
    #         #              ticks=bound_ppt,
    #         #              orientation="vertical",  shrink=0.85,
    #         #              pad=0.04,  label='[mm/5min]')
    #
    #         plt.grid(alpha=0.25)
    #         # Used to return the plot as an image rray
    #         fig.canvas.draw()  # draw the canvas, cache the renderer
    #         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #         # , vmax=max(max_stn_pcp,
    #         all_images.append(image)
    #         plt.close('all')
    #
    #     print(' saving image')
    #     #         kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
    #     imageio.mimsave(
    #         os.path.join(out_save_dir, r'radar_event_date_%s_%s.gif'
    #                      % (str(event_start).replace(' ', '_').replace(':', '_'),
    #                         '5min')),
    #         all_images, fps=1)
    #     print(' done saving image')
    #     break
    #
    # #===================================================================
    # # calculate copulas
    # #===================================================================
    # # if you want to specify an outputfile -> os.path.join(savefolder,
    # import scipy.stats as stats
    #
    # modulepath = r'C:\Users\hachem\Desktop\WPy64-3880\python-3.8.8.amd64\Lib\site-packages\rmwspy'
    # sys.path.append(modulepath)
    #
    # from random_mixing_whittaker_shannon import RMWS
    # import gcopula_sparaest as sparest
    # import covariancefunction as cov_func
    # xs = in_df_coords_utm32.lon.min()
    # # xinc = 0.014798
    # xinc, yinc = 0.02, 0.02
    # xsize = int((in_df_coords_utm32.lon.max() -
    #              in_df_coords_utm32.lon.min()) / xinc) + 1
    #
    # ys = in_df_coords_utm32.lat.min()
    # ysize = int((in_df_coords_utm32.lat.max() -
    #              in_df_coords_utm32.lat.min()) / yinc) + 1
    # # data_df_nonan.loc[dates_many_stns, :].values.copy(order='c')
    #
    # neigbhrs_radius_dwd_list = np.arange(0, 53000, 2000)
    # nlag = len(neigbhrs_radius_dwd_list)
    # from scipy.spatial import distance
    # # nor_fact = 10
    # plt.ioff()
    # fig, (ax) = plt.subplots(
    #     1, 1, figsize=(12, 8), dpi=100)
    # for _dd, cc in zip(range(1, 10),
    #                    ['grey', 'b', 'g', 'r', 'c', 'm', 'orange', 'k', 'darkred']):
    #     depths_ser_d10 = data_ser_depth_all[data_ser_depth_all == _dd].dropna()
    #
    #     for _ii, _dd in zip(depths_ser_d10.index, depths_ser_d10.values):
    #         print(_ii)
    #         df_nonan = data_df_nonan.loc[_ii, :].dropna(how='all')
    #         # break
    #         xvals = in_df_coords_utm32.loc[df_nonan.index, 'X'].values.ravel()
    #         yvals = in_df_coords_utm32.loc[df_nonan.index, 'Y'].values.ravel()
    #
    #         dwd_xy_ngbrs = np.c_[xvals, yvals]
    #         pre = df_nonan.values
    #         pva = np.var(pre)
    #         vg = np.zeros(nlag)
    #         ng = np.zeros(nlag)
    #         nact = dwd_xy_ngbrs.shape[0]
    #         ds = distance.cdist(
    #             dwd_xy_ngbrs, dwd_xy_ngbrs, metric='euclidean')
    #         dx1 = np.empty((nact, nact))
    #         dx1[:, :] = dwd_xy_ngbrs[:, 0]
    #         dx2 = dx1.transpose()
    #         dxa = dx1 - dx2
    #         dy1 = np.empty((nact, nact))
    #         dy1[:, :] = dwd_xy_ngbrs[:, 1]
    #         dy2 = dy1.transpose()
    #         dya = dy1 - dy2
    #         lag = 5000.0
    #         v1 = np.empty((nact, nact))
    #         v1[:, :] = pre
    #         # print(v1)
    #         v2 = v1.transpose()
    #         v3 = (v1 - v2)**2
    #         # print(v3)
    #         ds = ds.flatten()
    #         v3 = v3.flatten()
    #         v3 = v3[ds < nlag * lag]
    #         ds = ds[ds < nlag * lag]
    #         for i in range(nlag):
    #             va = v3[ds < (i + 0.5) * lag]
    #             da = ds[ds < (i + 0.5) * lag]
    #             va = va[da > (i - 0.5 * lag)]
    #             # print(va)
    #             vg[i] = np.mean(va)
    #         vg_normed = vg / vg[-1]
    #         ax.plot(neigbhrs_radius_dwd_list,
    #                 vg_normed, c=cc, alpha=0.25)
    #
    #     ax.plot(neigbhrs_radius_dwd_list,
    #             vg_normed, c=cc, label='%d' % _dd)
    #     # dwd_lat_lon = np.c_[xvals, yvals]
    #     # p_xy = np.copy(dwd_lat_lon)
    #     #
    #     # p_xy[:, 0] = (p_xy[:, 0] - xs) / xinc
    #     # p_xy[:, 1] = (p_xy[:, 1] - ys) / yinc
    #     # p_xy = p_xy.astype(int)
    #     #
    #     # u = (stats.rankdata(df_nonan) - 0.5) / df_nonan.shape[0]
    #     #
    #     # # observations in copula (rank) space
    #     # covmods = ['Exp']
    #     # #['Mat', 'Exp', 'Sph',]
    #     # # covariance function that will be tried for the fitting
    #     # ntries = 6
    #     # # number of tries per covariance function with random subsets
    #     # cmods = sparest.paraest_multiple_tries(
    #     #     np.copy(p_xy),
    #     #     u,
    #     #     ntries=[ntries, ntries],
    #     #     n_in_subset=5,               # number of values in subsets
    #     #     neighbourhood='nearest',     # subset search algorithm
    #     #     covmods=covmods,             # covariance functions
    #     #     outputfile=None)       # store all fitted models in an output file
    #     #
    #     # # take the copula model with the highest likelihood
    #     # # reconstruct from parameter array
    #     # likelihood = -555
    #     # for model in range(len(cmods)):
    #     #     for tries in range(ntries):
    #     #         if cmods[model][tries][1] * -1. > likelihood:
    #     #             likelihood = cmods[model][tries][1] * -1.
    #     #             cmod = '0.01 Nug(0.0) + 0.99 %s(%1.3f)' % (
    #     #                 covmods[model], cmods[model][tries][0][0])
    #     #             if covmods[model] == 'Mat':
    #     #                 cmod += '^%1.3f' % (cmods[model][tries][0][1])
    #     #
    #     # cov_mod_sill = float(cmod.split('+')[1].split(' ')[1])
    #     # cov_mod_range = float(cmod.split(
    #     #     '+')[1].split(' ')[2].split('(')[1].strip(')'))
    #     # h = np.arange(0, cov_mod_range + 100)
    #     # exp_cov = cov_func.type_exp(h, Range=cov_mod_range, Sill=cov_mod_sill)
    #
    #     # plt.show()
    #     # ax.set_xlim(0, 250)
    #     # ax.set_ylim(0, 1.01)
    # ax.set_xlabel('Range [m]')
    # ax.set_ylabel('Gamma (h)')
    # ax.legend(loc='lower left', ncol=2)
    # ax.grid(alpha=0.25)
    # plt.tight_layout()
    # plt.savefig(os.path.join(
    #     out_save_dir, 'vg_events_all_normed10.png'))
    # plt.close()

    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
