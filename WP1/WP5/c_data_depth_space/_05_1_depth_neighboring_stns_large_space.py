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


from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fiona import collection

from scipy.spatial import ConvexHull
from descartes import PolygonPatch
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)


modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf


def main():

    radar_loc = 'Hannover'  # Hannover  Tuerkheim  Feldberg
    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\05_depth_in_space")
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        # r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        # r"\dwd_comb_1min_data_agg_60min_2020_utm32.h5"
        r"\all_dwd_60min_1995_2021.h5"
        #% radar_loc
    )

    shp_path = (
        r"X:\staff\elhachem\ClimXtreme\03_data\03_Shapefiles"
        r"\DE_amd_shp_utm32\DEU_boundaries.shp")

    n_vecs = int(1e4)

    data_freq = '60min'

    n_cpus = 6  # 'auto'
    n_dims = 6
    min_pcp = 10  # mm
    neighbor_to_chose = 1000

    beg_date = '2000-01-01'
    end_date = '2021-12-31'

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    dwd_coords = dwd_hdf5.get_coordinates(ids=dwd_ids)

    in_dwd_df_coords_utm32 = pd.DataFrame(
        index=dwd_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    in_dwd_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords

    # create a tree from DWD coordinates

    dwd_coords_xy = [(x, y) for x, y in zip(
        in_dwd_df_coords_utm32.loc[:, 'X'].values,
        in_dwd_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    dwd_points_tree = cKDTree(dwd_coords_xy)

    for _dwd_id in dwd_ids[186:192]:
        print(_dwd_id)
        # break
        df_stn = dwd_hdf5.get_pandas_dataframe_between_dates(
            _dwd_id, event_start=beg_date,
            event_end=end_date)

        df_center_resample = resampleDf(df_stn, data_freq).dropna()

        out_dir = os.path.join(out_save_dir, '%s' % _dwd_id)

        print('out_dir:', out_dir)

        (xdwd, ydwd) = (
            in_dwd_df_coords_utm32.loc[_dwd_id, 'X'],
            in_dwd_df_coords_utm32.loc[_dwd_id, 'Y'])

        distances, indices = dwd_points_tree.query(
            np.array([xdwd, ydwd]),
            k=neighbor_to_chose + 1)

        stn_near_all = list(np.array(dwd_ids)[indices])

        data_zero = np.zeros(shape=(len(df_stn.index),
                                    len(stn_near_all)))

        df_depth_all = pd.DataFrame(data=data_zero,
                                    index=df_stn.index,
                                    columns=stn_near_all)
        idx_list = []

        start_idx = 0
        end_idx = n_dims
        max_nbr = 1000  # len(stn_near_all)

        plt.ioff()
        fig = plt.figure(figsize=(12, 12), dpi=200)
        for i in range(1, max_nbr):

            try:
                stn_near = list(np.array(dwd_ids)[indices[start_idx:end_idx]])
                if len(stn_near) > 0:
                    (xnear, ynear) = (
                        in_dwd_df_coords_utm32.loc[stn_near, 'X'],
                        in_dwd_df_coords_utm32.loc[stn_near, 'Y'])
                    points = np.array([(x, y) for x, y in zip(xnear, ynear)])
                    hull = ConvexHull(points)

                    for simplex in hull.simplices:
                        plt.plot(points[simplex, 0],
                                 points[simplex, 1], 'grey',
                                 alpha=0.25,
                                 linestyle='--',
                                 linewidth=1)

                    plt.plot(points[:, 0], points[:, 1], '.', color='r')
                    plt.scatter(xdwd, ydwd, marker='X', color='b')
                    start_idx = start_idx + 1
                    end_idx = end_idx + 1
                    print(start_idx, end_idx)
                    print(stn_near)

                    # plt.show()
                    # plt.scatter(in_dwd_df_coords_utm32.loc[:, 'X'],
                    #            in_dwd_df_coords_utm32.loc[:, 'Y'])
                    # plt.scatter(xnear, ynear, c='grey')

                # plt.show()
                    # print('sep distance', distances[1:])
                    print('getting data')
                    df_stn_near = dwd_hdf5.get_pandas_dataframe_between_dates(
                        stn_near, event_start=beg_date,
                        event_end=end_date)

                    df_nearby_resampled = resampleDf(
                        df_stn_near, data_freq).dropna(how='any')

                    df_nearby_resampled_norm = df_nearby_resampled.loc[
                        df_center_resample.index.intersection(
                            df_nearby_resampled.index), :]
                    df_nearby_resampled_norm = df_nearby_resampled_norm.iloc[
                        np.where(df_nearby_resampled_norm.sum(
                            axis=1) > min_pcp)[0], :]

                    # hourly: 0.1, 0.05,
                    # daily
                    for _col in df_nearby_resampled_norm.columns:

                        df_nearby_resampled_norm.loc[
                            df_nearby_resampled_norm[_col] == 0, _col
                        ] = np.random.random() * np.random.uniform(
                            0.02, 0.1,
                            len(df_nearby_resampled_norm.loc[
                                df_nearby_resampled_norm[_col] == 0]))

                    df_pos = df_nearby_resampled_norm
                    if len(df_pos.index) > 10:
                        #.iloc[
                        #    np.where(df_nearby_resampled.sum(axis=1) > 0)[0], :]
                        # df_pos.shape, usph_vecs.shape
                        tot_refr_var_arr = df_pos.values.copy('c')
                        usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)
                        depths2 = depth_ftn_mp(tot_refr_var_arr,
                                               tot_refr_var_arr,
                                               usph_vecs, n_cpus, 1)
                        print('done calculating depth')

                    # plt.ioff()
                    # plt.figure(figsize=(12, 8), dpi=200)
                    # plt.scatter(df_pos.sum(axis=1),
                    #             depths2,
                    #             c=depths2,
                    #             s=10, marker='o',
                    #             alpha=0.75,
                    #             cmap=plt.get_cmap('jet_r'),
                    #             vmin=1, vmax=5)
                    #
                    # plt.colorbar(label='Depth')
                    # # plt.ylim([0, 15])
                    # plt.xlabel('Sum over 6 Dimensions', fontsize=16)
                    # plt.ylabel('Depth of each point', fontsize=16)
                    # plt.grid(alpha=0.5)
                    # plt.tight_layout()
                    # # plt.show()
                    # plt.savefig(
                    #     r'X:\staff\elhachem\ClimXtreme\04_analysis\10_depth_function\6d_test_sp2.png')
                    # depths2.shape
                    # plt.close()
                        df_pcp_depth = pd.DataFrame(index=df_pos.sum(axis=1).index,
                                                    data=depths2,
                                                    columns=['d'])

                        df_low_d = df_pcp_depth[(df_pcp_depth >= 1) & (
                            df_pcp_depth < 2)].dropna()
                        # df_pcp_depth[df_pcp_depth.values == 1].dropna()
                        df_low_d.shape
                        de = df_pos.loc[df_low_d.index]  # .sum(axis=1)#.max()
                        de[de < 0.1] = 0
                        df_depth_all.loc[de.index, de.columns] += 1

                        idx_list.append(de.index)
                        # break
            except Exception as msg:
                print(msg, _dwd_id)
                continue
        print('done')

        plt.grid(alpha=0.05)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(
            out_save_dir, 'hulls_%s_2a_nbr.png' % _dwd_id))
        plt.close()

        # len(idx_list)

        print('done')

        df_results = df_depth_all[df_depth_all.sum(axis=1) > 0]
        # df_results.sum(axis=1).max()

        # dwd_ids[:1000][
        # list(np.where(df_results.iloc[0, :] > 0)[0].astype(int))]

      # len(idx_list)
        percent_unusual = 100 * (
            df_results.index.shape[0] / df_depth_all.index.shape[0])
        print(percent_unusual)
        pcp_data_events = pd.DataFrame(index=df_results.index,
                                       columns=df_results.columns)

        for ii, _idx in enumerate(df_results.index):
            print(_idx, ii, len(df_results.index))
            ids_event = df_results.columns[
                np.where(df_results.loc[_idx, :] > 0)[0].astype(int)]
            # if len(ids_event) > 7:
            # print('dd')
            df_depth_low = dwd_hdf5.get_pandas_dataframe_for_date(
                ids_event,
                event_date=_idx)
            pcp_data_events.loc[_idx, ids_event] = df_depth_low.values

        pcp_data_events.dropna(axis=1, how='all', inplace=True)
        pcp_data_events.sum(axis=1).plot(legend=False)
        # plt.show()
        max_Events = pcp_data_events.sum(axis=1).sort_values()
        max_event = pcp_data_events.loc[
            max_Events.index[-15:], :].dropna(axis=1, how='all')  # .plot()

        for _idx in max_event.index:
            data_event = max_event.loc[_idx, :].dropna()
            xcoords = in_dwd_df_coords_utm32.loc[data_event.index, 'X']
            ycoords = in_dwd_df_coords_utm32.loc[data_event.index, 'Y']

            df_ = dwd_hdf5.get_pandas_dataframe_for_date(
                dwd_ids,
                event_date=str(_idx))

            plt.ioff()
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
            ax1.scatter(in_dwd_df_coords_utm32.loc[df_.columns, 'X'],
                        in_dwd_df_coords_utm32.loc[df_.columns, 'Y'],
                        c=df_.values,
                        marker='X', s=50,
                        cmap=plt.get_cmap('YlGnBu'))

            pcp = ax1.scatter(xcoords, ycoords, c=data_event.values,
                              marker='X', s=50,
                              cmap=plt.get_cmap('YlGnBu'))
            with collection(shp_path, "r") as input:
                for f in input:
                    ax1.add_patch(
                        PolygonPatch(
                            f['geometry'], fc='grey',
                            ec='grey',
                            alpha=0.75,
                            zorder=3, fill=False))
            # plt.scatter()
            fig.colorbar(pcp, shrink=0.75, label='mm/hr')
            ax1.grid(alpha=0.25)
            ax1.set_title('%s sum=%0.2f mm' % (_idx,
                                               df_.values.sum()))
            plt.tight_layout()
            plt.savefig(
                os.path.join(out_save_dir,
                             'max_event_sum_%s_%s.png'
                             % (str(_idx).replace(':', '_').replace(' ', '_'),
                                 _dwd_id)))

            plt.close()
        # de.sum(axis=1).plot()
        # df_pos.loc['2016-06-23 18:00:00 ':'2016-06-23 20:00:00 ', :]

    #             ad_plot.plot_ecops()
    return


if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
