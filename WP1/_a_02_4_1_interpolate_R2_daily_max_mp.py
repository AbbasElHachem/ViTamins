# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Name:    Cross-validate the highest 4 biggest annual values
Purpose: Data quality

Created on: 2020-06-18

Parameters
----------
DWD daily or sub-hourly resolutions

Returns
-------
Interpolated data highest extremes

"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '02.03.2020'
# =============================================================
# from spinterps import OrdinaryKriging, variograms

import os

import sys
import time
import timeit
# import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import spatial
# import gstools as gs
import multiprocessing as mp
import matplotlib.dates as mdates
from pykrige.ok import OrdinaryKriging as OKpy


modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\02_data_quality'
sys.path.append(modulepath)
# from adjustText import adjust_text
from _a_02_0_additional_functions import(
    plot_config_event, plot_config_event_shifted)

# VG = variograms.vgs.Variogram

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

# plt.rcParams.update({'font.size': 26})
# plt.rcParams.update({'axes.labelsize': 26})

# Data Path
# =============================================================================
data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
# =============================================================
out_save_dir = data_path / r"Results\00_data_quality_outliers"

if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)



# Settings
#==============================================================================

# number of values to interpolate
nbr_biggest_vals = 5

min_cr_value = 3
# maximum radius for station selection
neigbhrs_radius_dwd = 10e4
nbr_neighbours = 30

# for VG fitting
vg_sill_b4_scale = 0.07
vg_range = 5e4
vg_model_str = 'spherical'

vg_model_to_scale = '0.07 Sph(%d)' % vg_range  # Sph

n_workers = 7

plot_space_config_shifted = False
#==========================================================================


def process_manager(args):

    (path_to_dwd_hdf5) = args

    HDF5_dwd_ppt = HDF5(infile=path_to_dwd_hdf5)
    all_dwd_stns_ids = HDF5_dwd_ppt.get_all_names()  
                     # stn_P03269_space_config_1948_06_13 00_00_002_transf
    # netatmo_in_coords_df.index.difference(all_netatmo_ids)
    dwd_coords = HDF5_dwd_ppt.get_coordinates(all_dwd_stns_ids)

    dwd_in_coords_df = pd.DataFrame(
        index=all_dwd_stns_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    dwd_in_coords_df.loc[:, 'Y'] = y_dwd_coords

    for ix, stn_id in enumerate(all_dwd_stns_ids):
        # stn_id = 'P03024'
        print(ix, ' / ', len(all_dwd_stns_ids))
        
        # if not os.path.exists(
            # os.path.join(out_save_dir, '%s_interp_vs_obsv.csv' % (stn_id))):
        # if True:
        if stn_id == '14002':
            # read df and get biggest values and time stamp
            stn_df = HDF5_dwd_ppt.get_pandas_dataframe(stn_id)
            stn_df_biggest = pd.DataFrame(index=stn_df.index, columns=[stn_id])
    
            for year_ in np.unique(stn_df.index.year):
                # print(year_)
                idx_yearly = np.where(stn_df.index.year == year_)[0]
                stn_df_year = stn_df.iloc[idx_yearly,:].nlargest(
                    n=nbr_biggest_vals, columns=stn_id).sort_index()
                stn_df_biggest.loc[stn_df_year.index,:] = stn_df_year.values
    
            stn_df_biggest.dropna(how='all', inplace=True)
    
            # create empty for saving results
            stn_df_biggest_interp = stn_df_biggest.copy(deep=True)
    
            # find neighbours within radius
            x_dwd_interpolate = dwd_in_coords_df.loc[stn_id, 'X']
            y_dwd_interpolate = dwd_in_coords_df.loc[stn_id, 'Y']
    
            # drop stns
            all_dwd_stns_except_interp_loc = [
                stn for stn in all_dwd_stns_ids if stn != stn_id and len(stn) > 0]
    
            all_timestamps_worker = np.array_split(
                stn_df_biggest.index, n_workers)
    
            print('Using %d Workers: ' % n_workers, 'for ',
                  len(stn_df_biggest.index), 'events')
            args_worker = []
            for time_list in all_timestamps_worker:
    
                args_worker.append((path_to_dwd_hdf5,
                                    dwd_in_coords_df,
                                    stn_id,
                                    time_list,
                                    all_dwd_stns_except_interp_loc,
                                    x_dwd_interpolate,
                                    y_dwd_interpolate,
                                    stn_df_biggest_interp,
                                    out_save_dir))
    
            my_pool = mp.Pool(processes=n_workers)
    
            results = my_pool.map(
                detect_outliers, args_worker)
    
            # my_pool.terminate()
    
            my_pool.close()
            my_pool.join()
    
            results_df = pd.concat(results)
    
            results_df.dropna(how='any', inplace=True)
            if len(results_df.columns) > 1:
                results_df.to_csv(os.path.join(out_save_dir, '%s_interp_vs_obsv.csv' % (stn_id)),
                              sep=';', float_format='%0.2f')
            # break

    return


#==============================================================================
#
def detect_outliers(args):
    (path_to_dwd_hdf5,
     dwd_in_coords_df,
     stn_id,
     time_list,
     all_dwd_stns_except_interp_loc,
     x_dwd_interpolate,
     y_dwd_interpolate,
     stn_df_biggest_interp,
     out_save_dir) = args
    # start interpolation of events

    HDF5_dwd_ppt = HDF5(infile=path_to_dwd_hdf5)

    for ex, event_date in enumerate(time_list):
        print(event_date, ex, ' / ', len(time_list))

        obsv_ppt_0 = stn_df_biggest_interp.loc[event_date, stn_id]
        
        if obsv_ppt_0 > 100:
            obsv_ppt_0_r2 = np.sqrt(obsv_ppt_0)
    
            print('Getting data')
            df_ngbrs = HDF5_dwd_ppt.get_pandas_dataframe_for_date(
                ids=all_dwd_stns_except_interp_loc, event_date=event_date
            ).dropna(axis=1, how='all')
            ids_with_data = df_ngbrs.columns.to_list()
    
            x_dwd_all = dwd_in_coords_df.loc[ids_with_data, 'X'].dropna().values
            y_dwd_all = dwd_in_coords_df.loc[ids_with_data, 'Y'].dropna().values
            dwd_neighbors_coords = np.c_[x_dwd_all.ravel(), y_dwd_all.ravel()]
            points_tree = spatial.KDTree(dwd_neighbors_coords)
            _, indices = points_tree.query(
                np.array([x_dwd_interpolate, y_dwd_interpolate]),
                k=nbr_neighbours + 1)
            ids_ngbrs = [ids_with_data[ix_ngbr] for ix_ngbr in indices]
            df_ngbrs = df_ngbrs.loc[:, ids_ngbrs].dropna(axis=1, how='all')
    
            stns_ngbrs = df_ngbrs.columns.to_list()
    
            x_ngbrs = dwd_in_coords_df.loc[stns_ngbrs, 'X']
            y_ngbrs = dwd_in_coords_df.loc[stns_ngbrs, 'Y']
    
            ppt_ngbrs = df_ngbrs.loc[:, stns_ngbrs].values.ravel()
            ppt_ngbrs_r2 = np.sqrt(ppt_ngbrs)
            try:
                assert x_ngbrs.size == y_ngbrs.size == ppt_ngbrs_r2.size
            except Exception as msg:
                print(msg)
            print('*Done getting data and coordintates* \n *Fitting variogram*\n')
    
            # bin_center, gamma = gs.vario_estimate((x_ngbrs, y_ngbrs),
                                                  # ppt_ngbrs_r2)
            # fit the variogram with a stable model. (no nugget fitted)
            # fit_model = gs.Spherical(dim=1)
            # try:
            #     fit_vg = fit_model.fit_variogram(
            #         bin_center, gamma, nugget=False)
            #
            #     vg_sill = fit_vg[0]['var']
            #     vg_range = fit_vg[0]['len_scale']
            # except Exception as msg2:
            #     print(msg2, 'erro, vg')
            vg_sill = np.var(ppt_ngbrs_r2)
            # vg_range = min(4e4, vg_range)
            vg_range = 4e4
            print('Interpolating')
            # ppt_var = np.var(ppt_ngbrs)
            # vg_scaling_ratio = ppt_var / vg_sill_b4_scale
    
            # if vg_scaling_ratio == 0:
            # vg_scaling_ratio = vg_sill_b4_scale
    
            OK_dwd_netatmo_crt = OKpy(
                x_ngbrs, y_ngbrs,
                ppt_ngbrs_r2,
                variogram_model=vg_model_str,
                variogram_parameters={
                    'sill': vg_sill,
                    'range': vg_range,
                    'nugget': 0},
                exact_values=True)
    
            # sigma = _
            try:
                interpolated_vals_O0, est_var = OK_dwd_netatmo_crt.execute(
                    'points', np.array([x_dwd_interpolate]),
                    np.array([y_dwd_interpolate]))
            except Exception as msg:
                print('ror', msg)
    
            interpolated_vals_O0 = interpolated_vals_O0.data
            interpolated_vals_O0[interpolated_vals_O0 < 0] = 0
    
            stn_df_biggest_interp.loc[event_date,
                                      'Interp'] = interpolated_vals_O0[0]**2
    
            #======================================================================
            # # calcualte standard deviation of estimated values
            #======================================================================
            std_est_vals_O0 = np.round(np.sqrt(np.abs(est_var).data), 4)
    #         import math
    #         if math.isnan(std_est_vals_O0):
    #             print('error')
    #             raise Exception
    
            # calculate difference observed and estimated  # values
            try:
                if std_est_vals_O0 > 0:
                    diff_obsv_interp_O0 = np.round(np.abs(
                        (interpolated_vals_O0 - obsv_ppt_0_r2
                         ) / std_est_vals_O0), 2)
                else:
                    diff_obsv_interp_O0 = np.round(np.abs(
                        (interpolated_vals_O0 - obsv_ppt_0_r2)), 2)
            except Exception:
                print('ERROR STD')
    
            if diff_obsv_interp_O0 < min_cr_value:
                # print('not outlier')
                stn_df_biggest_interp.loc[event_date,
                                          'Outlier O0'] = diff_obsv_interp_O0[0]
            else:
                print('outlier')
                stn_df_biggest_interp.loc[event_date,
                                          'Outlier O0'] = diff_obsv_interp_O0[0]
    
                event_start = event_date + pd.Timedelta(hours=-24)
                event_end = event_date + pd.Timedelta(hours=24)
                # dwd_hdf5_orig.get_pandas_dataframe([stn_id])
                orig_hourly = HDF5_dwd_ppt.get_pandas_dataframe_between_dates(
                    [stn_id], event_start=event_start, event_end=event_end)
    
                orig_ngbrs_hourly = HDF5_dwd_ppt.get_pandas_dataframe_between_dates(
                    stns_ngbrs, event_start=event_start, event_end=event_end)
                plt.ioff()
                fig = plt.figure(figsize=(5, 4), dpi=300)
    
                ax = fig.add_subplot(111)
                ax.plot(orig_hourly.index, orig_hourly.values,
                        c='r', marker='+', alpha=0.7, label='Stn')
                ax.plot(orig_ngbrs_hourly.index, orig_ngbrs_hourly.values, c='b',
                        marker='.', alpha=0.7)
                ax.plot(orig_ngbrs_hourly.index[0],
                        0, c='b', label='Neighbours')
                ax.grid(alpha=0.5)
                yvals = np.arange(0, np.nanmax(orig_hourly.values.ravel()) + 1, 10)
                ax.set_yticks(ticks=yvals)
                ax.set_yticklabels(labels=yvals)
                ax.xaxis.set_major_formatter(
                    mdates.DateFormatter('%Y-%m-%d'))
                ax.set_xticks(ticks=orig_ngbrs_hourly.index)
                ax.set_xticklabels(orig_ngbrs_hourly.index,
                                   rotation=45 )
                plt.legend(loc=0)
                # plt.show()
                plt.savefig(os.path.join(out_save_dir, 'stn_%s_ngbrs_%s.png' % (stn_id, str(event_date).replace(':', '_').replace('-', '_'),
                                         )), bbox_inches='tight', pad_inches=.2)
                print('plotting spacial config')
                plot_config_event(event_date=event_date,
                                  stn_one_id=stn_id, stn_one_xcoords=x_dwd_interpolate,
                                  stn_one_ycoords=y_dwd_interpolate,
                                  ppt_stn_one=obsv_ppt_0,
                                  interp_ppt=np.square(interpolated_vals_O0),
                                  stns_ngbrs=stns_ngbrs,
                                  ppt_ngbrs=ppt_ngbrs,
                                  x_ngbrs=x_ngbrs,
                                  y_ngbrs=y_ngbrs,
                                  out_dir=out_save_dir,
                                  save_acc=diff_obsv_interp_O0,
                                  transf_data=True,
                                  show_axis=True)
                print('done plotting')
    
                if plot_space_config_shifted:
                    print('Shifting dataframes')
                    # shift event date by minus/plus one day
                    shifted_event_p1 = event_date + pd.Timedelta(days=1)
                    df_ngbrs_p1 = HDF5_dwd_ppt.get_pandas_dataframe_for_date(
                        ids=ids_ngbrs, event_date=shifted_event_p1)
                    # df_ngbrs_p1 = df_ngbrs_p1.loc[:, df_ngbrs.columns]
    
                    shifted_event_m1 = event_date + pd.Timedelta(days=-1)
                    df_ngbrs_m1 = HDF5_dwd_ppt.get_pandas_dataframe_for_date(
                        ids=ids_ngbrs, event_date=shifted_event_m1)
    
                    print('Plotting configuration')
                    # plot configuration
                    plot_config_event_shifted(event_date=event_date,
                                              stn_one_id=stn_id, stn_one_xcoords=x_dwd_interpolate,
                                              stn_one_ycoords=y_dwd_interpolate,
                                              ppt_stn_one=obsv_ppt_0,
                                              interp_ppt=interpolated_vals_O0,
    
                                              stns_ngbrs=stns_ngbrs,
                                              ppt_ngbrs=ppt_ngbrs,
                                              x_ngbrs=x_ngbrs,
                                              y_ngbrs=y_ngbrs,
                                              ppt_ngbrs_shifted_p1=df_ngbrs_p1.values.ravel(),
                                              ppt_ngbrs_shifted_m1=df_ngbrs_m1.values.ravel(),
                                              out_dir=out_save_dir)

    return stn_df_biggest_interp


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    # from torch import multiprocessing
    # multiprocessing.set_start_method('spawn')
    variables = ['precipitation', 'pet', 'temperature', 'discharge_vol', 
             'humidity', 'longwave_rad','windspeed', 'discharge_spec']
    
    variables = ['precipitation']
    for var_to_test in variables:
    
        path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)

        process_manager(path_to_data)

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
