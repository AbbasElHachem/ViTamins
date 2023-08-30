# !/usr/bin/env python.
# -*- coding: utf-8 -*-

#=======================================================================
import sys
import os

import numpy as np
import pandas as pd

import copy
import h5py
from shapely.geometry import Polygon
import netCDF4 as nc
import matplotlib.pyplot as plt
import multiprocessing as mp
#from scipy.ndimage import map_coordinates
#from pysteps import motion

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelsize': 14})

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

#=======================================================================


path_coords = (
    r"X:\staff\elhachem\ClimXtreme\04_analysis\07_wind_displacement"
    r"\Locations_Hannover\significant_timesteps\xy_radar_grid_gk3.csv")

path_dwd_idx = (
    r"X:\staff\elhachem\ClimXtreme\04_analysis"
    r"\00_events_select\DWD_stns_radolan_coords_idx_Hannover_shifted2.csv")


# read coords
xy_coords = pd.read_csv(path_coords,
                        index_col=0,
                        sep=';',
                        engine='c')
df_idx_dwd_stns = pd.read_csv(path_dwd_idx,
                              index_col=0,
                              sep=';',
                              engine='c')
stn_xidx = df_idx_dwd_stns.Xidx.values.astype(int)
stn_yidx = df_idx_dwd_stns.Yidx.values.astype(int)
stn_xyidx = np.array([[0, j, i] for i, j in zip(stn_xidx, stn_yidx)])

ar = np.loadtxt(path_coords, delimiter=";", skiprows=1)
xy = ar[:, 1:]

xymin = np.array([np.min(xy[:, 0]), np.min(xy[:, 1])])

# print(xymin)

ixy = (xy - xymin) / 1000.0

ixy = ixy.astype(int)
# print(ixy)


#=======================================================================
#

def polygon_area(x, y):
    # coordinate shift
    x_ = x - x.mean()
    y_ = y - y.mean()
    # everything else is the same as maxb's code
    correction = x_[-1] * y_[0] - y_[-1] * x_[0]
    main_area = np.dot(x_[:-1], y_[1:]) - np.dot(y_[:-1], x_[1:])
    return 0.5 * np.abs(main_area + correction)


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h // nrows, nrows, -1, ncols)
               .swapaxes(1, 2)
               .reshape(-1, nrows, ncols))


def connected_areas(prc, min_pcp, stn_xyidx,
                    take_max_area):

    qelim = min_pcp

    icon = copy.deepcopy(prc)

    icon[icon < qelim] = 0

    icon[np.isnan(icon)] = 0
    # plt.imshow(icon)
    icon[icon > 0] = -1

    icon = icon.astype(int)

    jcon = copy.deepcopy(icon) * 0

    nu = np.argwhere([icon == -1])

    # nu[0]
    nall = nu.shape[0]

    icl = 1

    iret = 1

    while nu.shape[0] > 0:

        icon, jcon, iret = changenb(
            nu[0], icon, jcon, icl)
        # plt.imshow(jcon)
        # plt.imshow(icon)
        nu = np.argwhere([icon * jcon != 0])

        if nu.shape[0] > 0:

            icon, jcon, iret = changenb(
                nu[0], icon, jcon, icl)

        else:

            icl = icl + 1

            nu = np.argwhere([icon == -1])

            if nu.shape[0] > 0:

                icon, jcon, iret = changenb(
                    nu[0], icon, jcon, icl)

    siz = np.zeros([icl])

    df_blocks = pd.DataFrame(index=range(10000), columns=range(9))

    idx_start = 0
    # plt.ioff()
    # plt.figure()
    for ic in range(1, icl, 1):

        nu = np.argwhere([jcon == ic])

        siz[ic] = nu.shape[0]

        if nu.shape[0] >= 4:

            for ix, iy in zip(nu[:, 2], nu[:, 1]):
                ix_end = ix + 3
                iy_end = iy + 3

                box_pcp = prc[ix:ix_end, iy:iy_end]
                # flagged_pcp = kcon[ix:ix_end, iy:iy_end]
                if np.all(box_pcp) >= 0 and np.sum(box_pcp) > 0:
                    if box_pcp.ravel().size == 9:
                        try:
                            df_blocks.iloc[idx_start, :] = box_pcp.ravel()
                            idx_start += 1
                        except Exception as msg:
                            print('error getting blocks', msg)
                            continue
                    # kcon[ix:ix_end, iy:iy_end] = -9
                    # print(idx_start)

                    # plt.scatter(range(ix, ix_end + 1, 1),
                    #             range(iy, iy_end + 1, 1), alpha=0.1)

    # plt.show()
    df_blocks.dropna(inplace=True)
    # plt.close()
    return df_blocks


def changenb(kc, icon, jcon, icl):

    # mark neighbours for contiguous groups

    imx = icon.shape[0]

    imy = icon.shape[1]

    iret = 0

    for i1 in [-1, 0, 1]:

        for i2 in [-1, 0, 1]:

            j1 = kc[1] + i1

            j2 = kc[2] + i2

            if j1 > -1 and j1 < imx:

                if j2 > -1 and j2 < imy:

                    if icon[j1, j2] < 0:

                        jcon[j1, j2] = icl

                        iret = iret + 1

    icon[kc[1], kc[2]] = 0

    return icon, jcon, iret
#=======================================================================


def rad_area(args):
    (merged_hdf5_file,
     date_list,
     min_pcp,
     stn_xyidx,
     step_agg,
     take_max_area
     ) = args
    # xuq = np.unique(coords.X)
    # yuq = np.unique(coords.Y)

    #     pcp_vals1 = pcp_data[i_idx, :, :]
    # pcp_vals1 = pcp_data[i_idx, :, :]
    in_h5_file = h5py.File(merged_hdf5_file, 'r')
    pcp_Data = in_h5_file['data']

    df_blocks_image_combined = pd.DataFrame(
        index=range(100000), columns=range(9))
    # print(prec.shape)

    start_ix = 0
    for ii, idx in enumerate(date_list):
        print(ii, len(date_list))
        # idx = 6194
        # break
        # pcp_Data
        pcp_vals = pcp_Data[:, (idx - step_agg):idx +
                            1, ].reshape(262, 262, -1)

        R = np.zeros(shape=(pcp_vals.shape[-1], 262, 262))
        for i in range(R.shape[0]):
            R[i] = pcp_vals[:, :, i]

        R_ac = R[0].copy()
        # R_ac2 = R[0].copy()
        for i in range(R.shape[0] - 1):

            R_ac += R[i]
            if np.max(R_ac) > 15:
                print('getting data')
                plt.imshow(R_ac)
                (df_blocks_image) = connected_areas(
                    R_ac, min_pcp, stn_xyidx,
                    take_max_area)

                end_ix = start_ix + len(df_blocks_image.index)

                try:
                    df_blocks_image_combined.iloc[
                        start_ix:end_ix, :] = df_blocks_image.values
                    start_ix = end_ix
                except Exception as msg:
                    print(msg)
                    continue
                break
    return df_blocks_image_combined
#=======================================================================
# start reading radar images one by one
#=======================================================================


def find_dates_agg(df_dates, temp_res):

    int_min_periods = int(int(temp_res.split('m')[0]) / 5)

    df_cum = df_dates.rolling(temp_res,
                              min_periods=int_min_periods).sum()
    df_ti = df_cum[df_cum >= 0].dropna()
    #
    df_test = df_ti.resample(temp_res).sum()
    df_test = df_test.iloc[1:, :]
    df_ = df_test[df_test > 0].dropna()

    dates = df_.index
    df_dates_fin = pd.DataFrame(data=dates, columns=['Dates'])

    return df_dates_fin


def unq_searchsorted_v1(A, B):
    out1 = (np.searchsorted(B, A, 'right') -
            np.searchsorted(B, A, 'left')) == 1
    out2 = (np.searchsorted(A, B, 'right') -
            np.searchsorted(A, B, 'left')) == 1
    return out1, out2


if __name__ == '__main__':

    print('reading pcp data')
    take_max_area = False

    years = np.arange(2017, 2018, 1)
    min_pcp = 0.1

    n_vecs = int(1e4)
    # 5min, 15min, 30min, 60min
    # 2, 3, 4, 5
    out_freq = '5min'
    n_workers = 6
    out_path = (r'X:\staff\elhachem\ClimXtreme'
                r'\04_analysis\12_areal_rainfall\thr_%0.1f_%s_depth_image'
                % (min_pcp, out_freq))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    dfs = []
    # for area_nbr in area_nbrs:
    # print('AREA: ', area_nbr)
    for year in years:

        if not os.path.exists(os.path.join(out_path,
                                           '%d_area_pcp_depth.csv'
                                           % year)):
            print(year)
            # year_str = str(year)
            merged_nc_file = (
                r"X:\staff\elhachem\ClimXtreme\04_analysis\07_wind_displacement"
                r"\Locations_Hannover\significant_timesteps\merged_fields_201021_EDK"
                r"\%d\%d_merged_timesteps_5min_radar.nc"
                % (year, year))

            merged_hdf5_file = (
                r"X:\staff\elhachem\ClimXtreme\04_analysis\07_wind_displacement"
                r"\Locations_Hannover\significant_timesteps\merged_fields_201021_EDK"
                r"\%d\%d_merged_timesteps_5min_radar.h5"
                % (year, year))

            in_nc_file = nc.Dataset(merged_nc_file)

            # pcp_data = in_nc_file.variables['Pcp']

            # pcp_Data[:,0].reshape(262, 262)

            merged_dates = in_nc_file.variables['time'][:]
            # merged_dates = np.array(['2017-08-18 09:55:00'])
            date_Range_all = pd.DataFrame(index=merged_dates,
                                          data=np.ones(merged_dates.shape[0]))
            date_Range_all.index = pd.to_datetime(date_Range_all.index,
                                                  format='%Y-%m-%d %H:%M:%S')
            agg_timesteps = find_dates_agg(date_Range_all, out_freq)

            # np.argwhere(== )
            # np.where(merged_dates

            data_df = merged_dates[
                unq_searchsorted_v1(date_Range_all.index,
                                    agg_timesteps.values.ravel())[0]]
            data_df = data_df
            # np.where(merged_dates == '2017-08-18 09:55:00')
            idx_dates_merg = [list(merged_dates).index(
                data_df[i]) for i in range(len(data_df))]

            step_agg = int(int(out_freq.split('m')[0]) / 5)

            areas_all = []
            max_pcp_area = []
            mean_pcp_area = []
            pcp_stns_all = []

            nbr_areas_event = []

            all_timestamps_worker = np.array_split(
                idx_dates_merg, n_workers)

            print('Using %d Workers: ' % n_workers, 'for ',
                  len(idx_dates_merg), 'time steps')
            args_worker = []
            for time_list in (
                    all_timestamps_worker
            ):
                print('initial process', time_list)
                # break
                args_worker.append((
                    merged_hdf5_file,
                    time_list,
                    min_pcp,
                    stn_xyidx,
                    step_agg,
                    take_max_area
                ))
            print('run data')
            my_pool = mp.Pool(processes=n_workers)

            results = my_pool.map(
                rad_area, args_worker)

            # my_pool.terminate()

            my_pool.close()
            my_pool.join()

            df_pos = pd.concat(results).dropna()
            df_pos_abv_zero = df_pos.iloc[
                np.where(df_pos.sum(axis=1) > 1)[0]]

            for _col in df_pos_abv_zero.columns:

                df_pos_abv_zero.loc[
                    df_pos_abv_zero[_col] == 0, _col
                ] = np.random.random() * np.random.uniform(
                    0.02, 0.1,
                    len(df_pos_abv_zero.loc[
                        df_pos_abv_zero[_col] == 0]))

            # if len(df_pos_abv_zero.index) > 10:
            #
            #     tot_refr_var_arr = df_blocks_image.values.astype(
            #         float).copy('c')
            #
            #     usph_vecs = gen_usph_vecs_mp(
            #         n_vecs, df_blocks_image.columns.size, n_workers)
            #     depths2 = depth_ftn_mp(tot_refr_var_arr,
            #                            tot_refr_var_arr,
            #                            usph_vecs, n_workers, 1)
            #     print('done calculating depth')
            #
            #     df_pcp_depth = pd.DataFrame(index=df_blocks_image.index,
            #                                 data=depths2,
            #                                 columns=['d'])
            #
            #     df_low_d = df_pcp_depth[(df_pcp_depth >= 1) & (
            #         df_pcp_depth < 2)].dropna()
            #     # df_pcp_depth[df_pcp_depth.values == 1].dropna()
            #     df_low_d.shape[0] / df_pcp_depth.shape[0]
            #     de = df_blocks_image.loc[df_low_d.index]  # .sum(axis=1)#.max()
            #     # de.iloc[:10, :]
            #     de[de < 0.1] = 0
            #
            #     np.argmax(de.sum(axis=1))
            #
            #     de.iloc[160:163,:]
            #
            #     pass
#=======================================================================
#
#=======================================================================
