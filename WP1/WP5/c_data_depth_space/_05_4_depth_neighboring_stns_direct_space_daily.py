'''

@author: Abbas El Hachem




'''
import sys
import os
import timeit
import time
import tqdm

from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
from fiona import collection
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from descartes import PolygonPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import pyximport
pyximport.install()

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WPX\depth_funcs'
sys.path.append(modulepath)

from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)


modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf


def main():

   # radar_loc = 'Hannover'  # Hannover  Tuerkheim  Feldberg
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
        r"\dwd_comb_1440min_gk3.h5"
        #% radar_loc
    )

    shp_path = (
        r"X:\staff\elhachem\Shapefiles\DE_amd_shp_utm32\DEU_0_gk3.shp")

    n_vecs = int(1e5)

    data_freq = '1440min'

    n_cpus = 7  # 'auto'
    #n_dims = 6
    min_pcp = 30  # mm
    neighbor_to_chose = 9

    beg_date = '1950-01-01'
    end_date = '2021-12-31'

    date_range = pd.date_range(start=beg_date, end=end_date, freq=data_freq)

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

    data_zero = np.zeros(shape=(len(date_range),
                                len(dwd_ids)))

    df_depth_all = pd.DataFrame(data=data_zero,
                                index=date_range,
                                columns=dwd_ids)

    idx_list = []

    # plt.ioff()
    # plt.figure(figsize=(6, 6), dpi=300)
    # for _dwd_id in tqdm.tqdm(dwd_ids):
    #     print(_dwd_id)
    #     # break
    #     df_stn = dwd_hdf5.get_pandas_dataframe(_dwd_id).dropna()
    #     df_stn_preiod = df_stn.loc[beg_date:end_date]

    #     # df_center_resample = resampleDf(df_stn, data_freq).dropna()

    #     # out_dir = os.path.join(out_save_dir, '%s' % _dwd_id)

    #     # print('out_dir:', out_dir)
    #     (xdwd, ydwd) = (
    #         in_dwd_df_coords_utm32.loc[_dwd_id, 'X'],
    #         in_dwd_df_coords_utm32.loc[_dwd_id, 'Y'])

    #     distances, indices = dwd_points_tree.query(
    #         np.array([xdwd, ydwd]),
    #         k=neighbor_to_chose + 100)

    #     stn_near = list(np.array(dwd_ids)[indices])
    #     df_neighbors = pd.DataFrame(index=df_stn_preiod.index,
    #                                 columns=stn_near)
    #     nbr_idx = 0

    #     for _st in stn_near:
    #         if nbr_idx < 7:
    #             try:
    #                 df_st = dwd_hdf5.get_pandas_dataframe_between_dates(
    #                     _st,
    #                     event_start=df_stn_preiod.index[0],
    #                     event_end=df_stn_preiod.index[-1]).dropna()

    #                 cmn_idx = df_neighbors.index.intersection(
    #                     df_st.index)

    #                 if cmn_idx.size > 0.9 * df_stn.index.size:
    #                     # print(df_st.head())

    #                     df_neighbors.loc[
    #                         cmn_idx,
    #                         _st] = df_st.loc[cmn_idx, :].values.ravel()
    #                     nbr_idx += 1
    #                     print(nbr_idx)
    #             except Exception as msg:
    #                 print(msg)
    #                 continue
    #                 # break
    #     df_stn_near_period = df_neighbors.dropna(how='all', axis=1)

    #     # plt.show()
    #     # plt.scatter(in_dwd_df_coords_utm32.loc[:, 'X'],
    #     #            in_dwd_df_coords_utm32.loc[:, 'Y'])
    #     # plt.scatter(xnear, ynear, c='grey')

    #     # plt.show()
    #     # print('sep distance', distances[1:])

    #     if len(df_stn_near_period.columns) > 0:
    #         # df_stn_near[df_stn_near > 90] = np.nan
    #         # df_nearby_resampled = resampleDf(
    #         # df_stn_near, data_freq).dropna(how='any')
    #         nearby_Stns = df_stn_near_period.columns.to_list()
    #         (xnear, ynear) = (
    #             in_dwd_df_coords_utm32.loc[nearby_Stns, 'X'],
    #             in_dwd_df_coords_utm32.loc[nearby_Stns, 'Y'])
    #         points = np.array([(x, y) for x, y in zip(xnear, ynear)])
    #         try:
    #             hull = ConvexHull(points)
    #         except Exception as msg:
    #             print(msg)
    #         for simplex in hull.simplices:
    #             plt.plot(points[simplex, 0],
    #                      points[simplex, 1], 'grey',
    #                      alpha=0.25,
    #                      linestyle='--',
    #                      linewidth=1)

    #         plt.plot(points[:, 0], points[:, 1], '.', color='r')
    #         plt.scatter(xdwd, ydwd, marker='X', color='b')

    #         print(nearby_Stns)

    #         df_nearby_resampled_norm = df_stn_near_period.loc[
    #             df_stn_near_period.index.intersection(
    #                 df_stn_preiod.index), :].dropna(axis=0, how='any')

    #         df_nearby_resampled_norm = df_nearby_resampled_norm.iloc[
    #             np.where(df_nearby_resampled_norm.max(
    #                 axis=1) > min_pcp)[0], :]  # .fillna(0)
    #         # df_nearby_resampled_norm.iloc[:,-1].dropna()
    #         # df_nearby_resampled_norm
    #         # hourly: 0.1, 0.05,
    #         # daily
    #         for _col in df_nearby_resampled_norm.columns:

    #             df_nearby_resampled_norm.loc[
    #                 df_nearby_resampled_norm[_col] == 0, _col
    #             ] = np.random.random() * np.random.uniform(
    #                 0.02, 0.1,
    #                 len(df_nearby_resampled_norm.loc[
    #                     df_nearby_resampled_norm[_col] == 0]))

            # df_pos = df_nearby_resampled_norm
            # if len(df_pos.index) > 10:

            #     tot_refr_var_arr = df_pos.values.astype(float).copy('c')
            #     tot_refr_var_arr.shape
            #     usph_vecs = gen_usph_vecs_mp(
            #         n_vecs, df_pos.columns.size, n_cpus)

            #     depths2 = depth_ftn_mp(tot_refr_var_arr,
            #                            tot_refr_var_arr,
            #                            usph_vecs, n_cpus, 1)
            #     print('done calculating depth')

            #     df_pcp_depth = pd.DataFrame(index=df_pos.index,
            #                                 data=depths2,
            #                                 columns=['d'])
            #     # df_pcp_depth.plot()
            #     df_low_d = df_pcp_depth[(df_pcp_depth > 1) & (
            #         df_pcp_depth < 3)].dropna()

            #     df_high_d = df_pcp_depth[(df_pcp_depth > np.mean(df_pcp_depth.values)) & (
            #         df_pcp_depth < np.max(df_pcp_depth.values))].dropna()


            #     # df_pcp_depth[df_pcp_depth.values == 1].dropna()
            #     df_low_d.shape[0] / df_pcp_depth.shape[0]
            #     de = df_pos.loc[df_low_d.index]  # .sum(axis=1)#.max()
            #     # de.iloc[:10, :]
            #     de[de < 0.1] = 0
            #     de = de[de.sum(axis=1) > 1]

            #     de_max = df_pos.loc[df_high_d.index]  # .sum(a♀xis=1)#.max()
            #     # de_max.sum().plot()
            #     # de.sum().plot()
            #     # plt.show()
            #     # stns_cols = [stn[0] for stn in de.columns.to_list()]
            #     df_depth_all.loc[de.index, de.columns.to_list()] += 1
            #     idx_list.append(de.index)
            # break
    # plt.grid(alpha=0.25)
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(os.path.join(
    #     out_save_dir, 'hulls_%s_22a_daily.png' % _dwd_id))
    # plt.close()

    # print('done')

    # df_results = df_depth_all[df_depth_all.sum(axis=1) > 0]
    # df_results.sum(axis=1).max()

    # dwd_ids[:1000][
    # list(np.where(df_results.il

  # len(idx_list)
    # percent_unusual = 100 * (
    #     df_results.index.shape[0] / df_depth_all.index.shape[0])
    # print(percent_unusual)
    # pcp_data_events = pd.DataFrame(index=df_results.index,
    #                                columns=df_results.columns)

    # for ii, _idx in enumerate(df_results.index):
    #     print(_idx, ii, len(df_results.index))
    #     ids_event = df_results.columns[
    #         np.where(df_results.loc[_idx, :] > 0)[0].astype(int)]
    #     # if len(ids_event) > 7:
    #     # print('dd')
    #     df_depth_low = dwd_hdf5.get_pandas_dataframe_for_date(
    #         ids_event,
    #         event_date=_idx)
    #     pcp_data_events.loc[_idx, ids_event] = df_depth_low.values

    # pcp_data_events.dropna(how='all', axis=1, inplace=True)
    # plot()
    print('READING DF RESULTS')
    pcp_data_events = pd.read_csv(os.path.join(out_save_dir, 'df_dwd_unusuals_daily.csv'),
                                  sep=';', index_col=0,
                                  parse_dates=True,
                                  infer_datetime_format=True,
                                  engine='c')
    pcp_data_events.sum(axis=1).plot(legend=False)
    max_Events = pcp_data_events.sum(axis=1).sort_values()

    # max_Events.plot()
    # plt.plot(max_Events.index.astype(str), max_Events.values)
    # plt.show()
    # np.where(max_Events.index == '2021-07')
    # max_Events.loc['2021-07-01':'2021-07-16'].sort_index()

    # df_ = dwd_hdf5.get_pandas_dataframe_for_date(
    #         dwd_ids,
    #         event_date=str(max_Events.index[-1:]))

    # max_Events.plot()
    max_event = pcp_data_events.loc[
        max_Events.index[-10:],
        # max_Events.loc['2021-07-13':'2021-07-16'].index,
        :].dropna(axis=1, how='all').sort_index()  # .plot()

    for _idx in max_event.index:
        print(_idx)
        data_event = max_event.loc[_idx, :].dropna()
        xcoords = in_dwd_df_coords_utm32.loc[data_event.index, 'X']
        ycoords = in_dwd_df_coords_utm32.loc[data_event.index, 'Y']

        df_ = dwd_hdf5.get_pandas_dataframe_for_date(
            dwd_ids,
            event_date=str(_idx))


        xcoords = in_dwd_df_coords_utm32.loc[df_.columns, 'X']
        ycoords = in_dwd_df_coords_utm32.loc[df_.columns, 'Y']


        #  new cmap
        bound_ppt = np.linspace(
            df_.values.min(),
            df_.values.max()+5, 10)
        bound_ppt[0] = 0.1  # bound_ppt[1] / 10

        cvals = bound_ppt  # , 20, 30, 40, 50, 60, 70]
        colors = ['lightcyan',  "cyan",

                  'royalblue', 'lime', 'green',
                  'yellow', 'orange',

                  'red', 'm',
                  'darkviolet', 'k']
        len(colors), len(bound_ppt)
        # colors = ["white", 'cyan',
        # 'royalblue', 'blue', 'darkblue']
        norm_ppt = plt.Normalize(
            min(cvals), max(cvals))
        tuples = list(
            zip(map(norm_ppt, cvals), colors))
        cmap_ppt = mcolors.LinearSegmentedColormap.from_list(
            "", tuples)

        cmap_ppt.set_under('lightyellow')


        # df_.dropna()
        plt.ioff()
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        pcp = ax1.scatter(xcoords, ycoords, c=df_.values,
                          marker='X', s=df_.values,
                          # vmin=0.5,
                          cmap=cmap_ppt,
                          norm=norm_ppt#plt.get_cmap('viridis')
                          )

        # ax1.scatter(in_dwd_df_coords_utm32.loc[df_.columns, 'X'],
        #             in_dwd_df_coords_utm32.loc[df_.columns, 'Y'],
        #             c=df_.values,
        #             marker='X', s=50,
        #             cmap=plt.get_cmap('Greys'),
        #             vmin=0.5)

        with collection(shp_path, "r") as input:
            for f in input:
                ax1.add_patch(
                    PolygonPatch(
                        f['geometry'], fc='grey',
                        ec='grey',
                        alpha=0.75,
                        zorder=3, fill=False))
        # plt.scatter()
        #fig.colorbar(pcp, shrink=0.75, label='mm/day')

        cb = fig.colorbar(pcp, ax=[ax1
                                   ], ticks=bound_ppt,
                          # norm=norm_ppt,
                          orientation="vertical",
                          shrink=0.95,
                          extend='min',

                          # location='right',
                          pad=0.1,  label='[mm/day]')
        ax1.grid(alpha=0.25)
        ax1.set_xlabel('Easting [m]')
        ax1.set_ylabel('Northing [m]')

        ax1.set_title('Date: %s  - Event sum=%0.2f mm' % (str(_idx).split(' ')[0],  # data_event.sum() +
                                           df_.values.sum()))
        # plt.tight_layout()
        plt.savefig(
            os.path.join(out_save_dir,
                         'max_event_sum_%s_dwde.png'
                         % str(_idx).replace(':', '_').replace(' ', '_')), bbox_inches='tight')

        plt.close()
        break
    # plt.show()
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
