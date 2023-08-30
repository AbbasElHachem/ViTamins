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
from scipy.ndimage import map_coordinates
from pysteps import motion

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

    # search connected areas
    # plt.imshow(prc)
    # qelim = np.percentile(prc, elim)
    qelim = min_pcp
    # print("Q elim", qelim)
    # zmat = np.zeros([262, 262])
    # for j in range(prc.ravel().shape[0]):
    #     # print(j)
    #     zmat[ixy[j, 0], ixy[j, 1]] = prc.ravel()[j]
    icon = copy.deepcopy(prc)

    icon[icon < qelim] = 0

    icon[np.isnan(icon)] = 0
    # plt.imshow(icon)
    icon[icon > 0] = -1

    icon = icon.astype(int)

    jcon = copy.deepcopy(icon) * 0
    # plt.imshow(jcon)
    # plt.show()
    kcon = copy.deepcopy(icon)

    nu = np.argwhere([icon == -1])

    # xc = xy_coords.loc[np.argwhere(icon.ravel() == -1).ravel(), 'X']
    # yc = xy_coords.loc[np.argwhere(icon.ravel() == -1).ravel(), 'Y']

    # nu[0]
    nall = nu.shape[0]

    icl = 1

    iret = 1

    while nu.shape[0] > 0:

        icon, jcon, iret = changenb(nu[0], icon, jcon, icl)
        # plt.imshow(jcon)
        nu = np.argwhere([icon * jcon != 0])

        if nu.shape[0] > 0:

            icon, jcon, iret = changenb(nu[0], icon, jcon, icl)

        else:

            icl = icl + 1

            nu = np.argwhere([icon == -1])

            if nu.shape[0] > 0:

                icon, jcon, iret = changenb(nu[0], icon, jcon, icl)

    siz = np.zeros([icl])
    areas_shape = np.zeros([icl])
    max_area = np.zeros([icl])
    mean_area = np.zeros([icl])
    pcp_stns = np.zeros([icl])
    i_locs = {i: [] for i in range(icl)}
    j_locs = {i: [] for i in range(icl)}
    for ic in range(1, icl, 1):

        nu = np.argwhere([jcon == ic])

        siz[ic] = nu.shape[0]

        if nu.shape[0] >= 1:
            # pgon = Polygon(zip(nu[:, 1], nu[:, 2]))
            # pgon = Polygon(zip(xc, yc))
            for _u in stn_xyidx:
                for u2 in nu:
                    if all(_u == u2):
                        # print(_u)
                        pcp_stns[ic] = prc[_u[1], _u[2]]

        # else:
            areas_shape[ic] = nu.shape[0] * 1 * 1
            max_area[ic] = np.max(prc[nu[:, 1], nu[:, 2]])
            mean_area[ic] = np.mean(prc[nu[:, 1], nu[:, 2]])
            i_locs[ic] = nu[:, 2]
            j_locs[ic] = nu[:, 1]

    # plt.colorbar(im2)
    ssiz = np.sort(siz)
    areaz = np.sort(areas_shape)
    mean_areaz = np.sort(mean_area)
    max_areaz = np.sort(max_area)
    pcp_stns_sort = np.sort(pcp_stns)
    lss = np.sum(ssiz)

    if take_max_area:
        idx_sort = np.argsort(areas_shape)
        reordered_i_locs = i_locs[idx_sort[-1]]
        reordered_j_locs = j_locs[idx_sort[-1]]
        areaz = np.sort(areas_shape)
        mean_areaz = np.sort(mean_area)
        max_areaz = np.sort(max_area)
        pcp_stns_sort = np.sort(pcp_stns)
        return (jcon, areaz[-1], max_areaz[-1],
                mean_areaz[-1], pcp_stns_sort[-1],
                reordered_i_locs, reordered_j_locs)

    # print(icl - 1, ssiz[-1], nall, ssiz[-1] / nall, lss)
    # print('areaz', areaz)
    # plt.imshow(jcon)
    else:
        return jcon, areaz, max_areaz, mean_areaz, pcp_stns_sort


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
     take_max_area) = args
    # xuq = np.unique(coords.X)
    # yuq = np.unique(coords.Y)

    #     pcp_vals1 = pcp_data[i_idx, :, :]
    # pcp_vals1 = pcp_data[i_idx, :, :]
    in_h5_file = h5py.File(merged_hdf5_file, 'r')
    pcp_Data = in_h5_file['data']
    areas_all = []
    max_pcp_area = []
    mean_pcp_area = []
    pcp_stns_all = []

    nbr_areas_event = []
    coords_i_locs = []
    coords_j_locs = []
    # plt.imshow(zmat_with_nans)

    # print(prec.shape)
    for ii, idx in enumerate(date_list):
        print(ii, len(date_list))
        # idx = 6194
        # break
        # pcp_Data
        pcp_vals = pcp_Data[:, (idx - step_agg):idx +
                            1, ].reshape(262, 262, -1)
        # break
        # pcp_vals.shape
        # pcp_vals = pcp_vals.reshape(13, 262, 262)
        R = np.zeros(shape=(pcp_vals.shape[-1], 262, 262))
        for i in range(R.shape[0]):
            R[i] = pcp_vals[:, :, i]

        R_ac = R[0].copy()
        # R_ac2 = R[0].copy()
        for i in range(R.shape[0] - 1):
            # print(i, i + 2)
            # R_ac += advection_correction(R[i: (i + 2)], T=5, t=1)
            R_ac += R[i]
        # plt.imshow(R_ac)
        # zmat = np.zeros([262, 262])
        if np.max(R_ac) > min_pcp:

            # for j in range(zz.ravel().shape[0]):
            #
            #     zmat[ixy[j, 0], ixy[j, 1]] = zz.ravel()[j]

            # elim = 99.9
            # qelim = np.percentile(zz, elim)
            (jcon, areaz, max_areaz,
             mean_areaz, pcp_stns_sort,
             ilocs, jlocs) = connected_areas(
                R_ac, min_pcp, stn_xyidx,
                take_max_area)
            # plt.imshow(zz)
            # plt.figure()
            # plt.imshow(jcon)
            if not take_max_area:
                nbr_areas_event.append(len(areaz))
            else:
                nbr_areas_event.append(1)
            # plt.scatter(ilocs, jlocs, marker=',')
            arr_area = np.array([ilocs, jlocs])
            arr_area.shape
            nrows = 6
            ncols = 6
            # plt.imshow(R_ac)
            blockshaped(arr_area, nrows, ncols)

            areas_all.append(areaz)
            max_pcp_area.append(max_areaz)
            mean_pcp_area.append(mean_areaz)
            pcp_stns_all.append(pcp_stns_sort)
            coords_i_locs.append(ilocs)
            coords_j_locs.append(jlocs)
            # Assuming the OP's x,y coordinates
    if not take_max_area:
        areas_all_comb = [_aa[i] for _aa in areas_all
                          for i in range(len(_aa))
                          if i > 0]

        max_pcp_area_comb = [_aa[i] for _aa in max_pcp_area
                             for i in range(len(_aa))
                             if i > 0]

        mean_pcp_area_comb = [_aa[i] for _aa in mean_pcp_area
                              for i in range(len(_aa))
                              if i > 0]

        pcp_stns_all_comb = [_aa[i] for _aa in pcp_stns_all
                             for i in range(len(_aa))
                             if i > 0]
    else:
        areas_all_comb = [_aa for _aa in areas_all
                          if _aa > 0]

        max_pcp_area_comb = [_aa for _aa in max_pcp_area
                             if _aa > 0]

        mean_pcp_area_comb = [_aa for _aa in mean_pcp_area
                              if _aa > 0]

        pcp_stns_all_comb = [_aa for _aa in pcp_stns_all
                             if _aa > 0]
    return (areas_all_comb, max_pcp_area_comb, mean_pcp_area_comb,
            pcp_stns_all_comb, nbr_areas_event,
            coords_i_locs, coords_j_locs)
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
    take_max_area = True

    years = np.arange(2017, 2020, 1)
    min_pcp = 0.0
    # 5min, 15min, 30min, 60min
    # 2, 3, 4, 5
    out_freq = '5min'
    n_workers = 1
    out_path = (r'X:\staff\elhachem\ClimXtreme'
                r'\04_analysis\12_areal_rainfall\thr_%0.1f_%s_depth'
                % (min_pcp, out_freq))

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    dfs = []
    for year in years:

        if not os.path.exists(os.path.join(out_path,
                                           '%d_area_pcp.csv'
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

            (areas_all_comb, max_pcp_area_comb,
             mean_pcp_area_comb, pcp_stns_all_comb,
             nbr_areas_comb,
             locations_i, locations_j
             ) = [], [], [], [], [], [], []

            for nw in range(n_workers):
                print(nw)
                for az, pmax, pmean, pstn, nareas, loc_is, loc_js in zip(
                        results[nw][0],
                        results[nw][1],
                        results[nw][2],
                        results[nw][3],
                        results[nw][4],
                        results[nw][5],
                        results[nw][6]):

                    areas_all_comb.append(az)
                    max_pcp_area_comb.append(pmax)
                    mean_pcp_area_comb.append(pmean)
                    pcp_stns_all_comb.append(pstn)
                    nbr_areas_comb.append(nareas)
                    locations_i.append(loc_is)
                    locations_j.append(loc_js)

            df_year = pd.DataFrame(index=range(len(areas_all_comb)))

            df_year['areas_all_comb'] = areas_all_comb
            df_year['max_pcp_area_comb'] = max_pcp_area_comb
            df_year['mean_pcp_area_comb'] = mean_pcp_area_comb
            df_year['pcp_stns_all_comb'] = pcp_stns_all_comb
            df_year['location_i'] = locations_i
            df_year['location_j'] = locations_j
            df_year.to_csv(os.path.join(out_path,
                                        '%d_area_pcp.csv'
                                        % year))

#=======================================================================
#
#=======================================================================
