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
from pysteps import motion
from fiona import collection
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.ndimage import map_coordinates
from descartes import PolygonPatch
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as
    gen_usph_vecs_mp, depth_ftn_mp)

import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import time
date_form = DateFormatter("%d-%H")
date_form2 = DateFormatter("%Y")
# plt.rcParams.update({'font.size': 16})
# plt.rcParams.update({'axes.labelsize': 16})


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf


modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\e_radklim'
sys.path.append(modulepath)

from _e_02_4_continuous_areas_Reutlingen import (
    read_nc_radklim, create_mask_shapefile)


def advection_correction(R, T=5, t=1):
    """
    R = np.array([qpe_previous, qpe_current])
    T = time between two observations (5 min)
    t = interpolation timestep (1 min)
    """

    # Evaluate advection
    oflow_method = motion.get_method("LK")
    fd_kwargs = {'max_num_features': None,
                 'minref': 0,
                 'maxref': 100,
                 'mindiff': 100,
                 'minsize': 0,
                 'minmax': 1000}

    interp_method = 'idwinterp2d'
    fd_method = "tstorm"  # "tstorm"
    #
    V = oflow_method(R,  interp_method=interp_method,
                     verbose=False,
                     fd_kwargs=fd_kwargs,
                     fd_method=fd_method)

    # Perform temporal interpolation
    Rd = np.zeros((R[0].shape))
    x, y = np.meshgrid(
        np.arange(R[0].shape[1], dtype=float), np.arange(
            R[0].shape[0], dtype=float)
    )
    for i in range(t, T + t, t):

        pos1 = (y - i / T * V[1], x - i / T * V[0])
        R1 = map_coordinates(R[0], pos1, order=1)

        pos2 = (y + (T - i) / T * V[1], x + (T - i) / T * V[0])
        R2 = map_coordinates(R[1], pos2, order=1)

        Rd += (T - i) * R1 + i * R2

    return t / T ** 2 * Rd


def main():

    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\05_depth_in_space_catchments")
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

    df_coords_in_bw = pd.read_csv(
        r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge"
        r"\Neckar_hourly\Headwater_catchments\dwd_stns_in_catch.csv",
        index_col=0,
        sep=',').index.to_list()

    df_discharge_data = pd.read_csv(
        r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge"
        r"\Neckar_hourly\abbas"
        r"\hourly_bw_discharge__2008__2019.csv",
        sep=';', index_col=0,
        parse_dates=True,
        infer_datetime_format=True)
    df_discharge_data=df_discharge_data.drop(columns='420')


    shp_path = (
    r"X:\staff\elhachem\Shapefiles\Neckar\watersheds_wgs84.shp")

    # path_to_dwd_hdf5 = (
    #     r"X:\staff\elhachem\ClimXtreme"
    #     r"\03_data\00_DWD\dwd_comb_1440min_gk3.h5"
    # )
    #

    #
    # df_discharge_data = pd.read_csv(
    #     r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge"
    #     r"Neckar_daily\disch_data_with_catch"
    #     r"neckar_daily_discharge_1961_2015.csv",
    #     sep=';', index_col=0,
    #     parse_dates=True,
    #     infer_datetime_format=True)
    #
    catch_loc = 'Kocher'

    catch_path = (
        r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge"
        r"\Neckar_hourly\Headwater_catchments\3_catch_comb.shp")

    # if catch_loc == 'Enz':
    #     path_to_shp = (
    #         r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge\Neckar_hourly"
    #         r"\Headwater_catchments\sub_catch_Enz_q3421_wgs84.shp")
    #     _q = '3421'
    # if catch_loc == 'Jagst':
    #     path_to_shp = (
    #         r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge\Neckar_hourly"
    #         r"\Headwater_catchments\sub_catch_Jagst_q3470_wgs84.shp")
    #     _q = '3470'
    # if catch_loc == 'Kocher':
    #     path_to_shp = (
    #         r"X:\staff\elhachem\ClimXtreme\03_data\07_Discharge\Neckar_hourly"
    #         r"\Headwater_catchments\sub_catch_Kocher_q3465_wgs84.shp")
    #     _q = '3465'

    qstns_csv = pd.read_csv(
        r"X:\staff\elhachem\ClimXtreme\03_data"
        r"\07_Discharge\Neckar_hourly\q_stns_gk3.csv",
        sep=',', index_col=0)

    pcp_stns_q = pd.read_csv(
        r"X:\staff\elhachem\ClimXtreme\03_data"
        r"\07_Discharge\Neckar_hourly\stns_in_3470.csv",
        sep=',', index_col=0)

    path_radklim = (
        r"X:\exchange\seidel\RADKLIM\RADKLIM YW\RADKLIM_YW2017.002_2018.nc"
    )
    (lons, lats, xs, ys,
     pcp_data, df_dates) = read_nc_radklim(path_radklim)
    mask = create_mask_shapefile(catch_path,
                                 lons,
                                 lats)

    n_vecs = int(1e4)

    data_freq = '60min'

    n_cpus = 1  # 'auto'

    min_pcp = 10  # mm
    neighbor_to_chose = 9

    beg_date = '2008-01-01'
    end_date = '2018-12-31'

    date_range = pd.date_range(start=beg_date,
                               end=end_date, freq=data_freq)

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    dwd_coords = dwd_hdf5.get_coordinates(ids=dwd_ids)

    in_dwd_df_coords_utm32 = pd.DataFrame(
        index=dwd_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    in_dwd_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords

    in_dwd_df_coords_utm32.loc[:, 'lon'] = dwd_coords['lon']
    in_dwd_df_coords_utm32.loc[:, 'lat'] = dwd_coords['lat']

    cmn_stns = in_dwd_df_coords_utm32.index.intersection(
        df_coords_in_bw)

    # create a tree from DWD coordinates
    # in_dwd_df_coords_utm32.loc[cmn_stns]
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

    plt.ioff()
    fig, (ax1) = plt.subplots(1, 1,
                              figsize=(4, 4), dpi=200)
    for _dwd_id in tqdm.tqdm(cmn_stns[:2]):
        print(_dwd_id)
        # break
        df_stn = dwd_hdf5.get_pandas_dataframe_between_dates(
            _dwd_id, event_start=beg_date,
            event_end=end_date)
        # df_stn[df_stn > 90] = np.nan
        df_center_resample = resampleDf(df_stn, data_freq).dropna()

        # out_dir = os.path.join(out_save_dir, '%s' % _dwd_id)

        # print('out_dir:', out_dir)

        (xdwd, ydwd) = (
            in_dwd_df_coords_utm32.loc[_dwd_id, 'X'],
            in_dwd_df_coords_utm32.loc[_dwd_id, 'Y'])

        (londwd, latdwd) = (
            in_dwd_df_coords_utm32.loc[_dwd_id, 'lon'],
            in_dwd_df_coords_utm32.loc[_dwd_id, 'lat'])

        distances, indices = dwd_points_tree.query(
            np.array([xdwd, ydwd]),
            k=neighbor_to_chose + 100)

        # stn_near = list(np.array(dwd_ids)[indices[:7]])

        stn_near = list(np.array(dwd_ids)[indices])
        df_neighbors = pd.DataFrame(index=df_center_resample.index,
                                    columns=stn_near)
        nbr_idx = 0

        for _st in stn_near:
            if nbr_idx < 10:
                try:
                    df_st = dwd_hdf5.get_pandas_dataframe_between_dates(
                        _st,
                        event_start=df_center_resample.index[0],
                        event_end=df_center_resample.index[-1]).dropna()
                    df_st_resample = resampleDf(df_st, data_freq).dropna()
                    cmn_idx = df_neighbors.index.intersection(
                        df_st_resample.index)

                    if cmn_idx.size > 0.9 * df_stn.index.size:
                        # print(df_st.head())

                        df_neighbors.loc[
                            cmn_idx,
                            _st] = df_st_resample.loc[cmn_idx, :].values.ravel()
                        nbr_idx += 1
                        print(nbr_idx)
                except Exception as msg:
                    print(msg)
                    continue
        df_stn_near_period = df_neighbors.dropna(how='all', axis=1)

        if len(df_stn_near_period.columns) > 0:
            (xnear, ynear) = (
                in_dwd_df_coords_utm32.loc[df_stn_near_period.columns,
                                           'lon'],
                in_dwd_df_coords_utm32.loc[df_stn_near_period.columns,
                                           'lat'])
            points = np.array([(x, y) for x, y in zip(xnear, ynear)])
            hull = ConvexHull(points)

            for simplex in hull.simplices:
                ax1.plot(points[simplex, 0],
                         points[simplex, 1], 'grey',
                         alpha=0.25,
                         linestyle='--',
                         linewidth=1)

            ax1.plot(points[:, 0], points[:, 1], '.', color='r')
            ax1.scatter(londwd, latdwd, marker='X', color='b')

            with collection(catch_path, "r") as input:
                for f in input:
                    ax1.add_patch(
                        PolygonPatch(
                            f['geometry'], fc='grey',
                            ec='grey',
                            alpha=0.75,
                            zorder=3, fill=False))

            print(df_stn_near_period.columns)

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
            df_stn_near[df_stn_near > 90] = np.nan
            df_nearby_resampled = resampleDf(
            df_stn_near, data_freq).dropna(how='any')

            df_nearby_resampled_norm = df_stn_near_period.loc[
                df_center_resample.index.intersection(
                    df_stn_near_period.index), :].dropna(axis=0, how='any')
            # df_nearby_resampled_norm
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

                tot_refr_var_arr = df_pos.values.astype(float).copy('c')

                usph_vecs = gen_usph_vecs_mp(
                    n_vecs, df_pos.columns.size, n_cpus)
                depths2 = depth_ftn_mp(tot_refr_var_arr,
                                        tot_refr_var_arr,
                                        usph_vecs, n_cpus, 1)
                print('done calculating depth')

                df_pcp_depth = pd.DataFrame(index=df_pos.index,
                                            data=depths2,
                                            columns=['d'])

                df_low_d = df_pcp_depth[(df_pcp_depth >= 1) & (
                    df_pcp_depth < 2)].dropna()
                # df_pcp_depth[df_pcp_depth.values == 1].dropna()
                df_low_d.shape[0] / df_pcp_depth.shape[0]
                de = df_pos.loc[df_low_d.index]  # .sum(axis=1)#.max()
                # de.iloc[:10, :]

                de[de < 0.1] = 0
                # de.loc[
                #     de.sum(axis=1).sort_values().index,:
                # ].iloc[-5:,:].to_csv(
                #     os.path.join(
                #         out_save_dir,'max_evt_test.csv'))
                # de.loc[de.sum(axis=1).sort_values().in
                df_depth_all.loc[de.index, de.columns] += 1
                idx_list.append(de.index)
            break

    with collection(catch_path, "r") as input:
        for f in input:
            ax1.add_patch(
                PolygonPatch(
                    f['geometry'], fc='blue',
                    ec='darkblue',
                    alpha=0.5,
                    zorder=3, fill=False))

    ax1.plot(points[:, 0], points[:, 1], '.', color='r',
             label='DWD')

    ax1.grid(alpha=0.25)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.legend(loc=0)
    # plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_save_dir, 'hulls_%s_22aw2.png' % _dwd_id),
        bbox_inches='tight')
    plt.close()

    print('done')

    df_results = df_depth_all[df_depth_all.sum(axis=1) > 0]
    arg_idx = np.where(df_results.sum(axis=1) > 10)[0]
    df_results_simul = df_results.iloc[arg_idx, :]
    df_results = df_results_simul
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
        # break
        # _idx= '2011-05-22 14:00:00'
        # 163 645
        try:
            ids_event = df_results.columns[
                np.where(df_results.loc[_idx, :] > 0)[0].astype(int)]
        # if len(ids_event) > 7:
        # print('dd')
            df_depth_low = dwd_hdf5.get_pandas_dataframe_for_date(
                ids_event,
                event_date=_idx)
            ids_event_with_data = df_depth_low.columns.intersection(ids_event)
            pcp_data_events.loc[_idx, ids_event_with_data] = df_depth_low.loc[
                _idx, ids_event_with_data].values
        except Exception as msg:
            print(msg)
            continue
    # pcp_data_events.dropna(how='all', axis=1, inplace=True)
    # plot()
    # pcp_data_events.read_csv(os.path.join(out_save_dir, 'df_dwd_unusuals.csv'),
            # sep=';', index_col=0,
            # parse_dates=True,
            # infer_datetime_format=True)
    # pcp_data_events.sum(axis=1).plot(legend=False)
    pcp_data_events.dropna(how='all', inplace=True, axis=1)
    pcp_data_events_evt = pcp_data_events.loc[:, cmn_stns]
    max_Events = pcp_data_events_evt.sum(axis=1).sort_values()

    # max_Events.plot()
    # plt.plot(max_Events.index.astype(str), max_Events.values)
    # plt.show()
    # np.where(max_Events.index == '2021-07')
    # max_Events.loc['2021-07-01':'2021-07-16'].sort_index()

    # df_ = dwd_hdf5.get_pandas_dataframe_for_date(
    #         dwd_ids,
    #         event_date=str(max_Events.index[-1:]))

    # max_Events.plot()
    max_event = pcp_data_events_evt.loc[
        max_Events.index[-10:],
        # max_Events.loc['2021-07-13':'2021-07-16'].index,
        :].dropna(axis=0, how='all').sort_index()  # .plot()

    for _idx in max_event.index:

        _idx = pd.DatetimeIndex(['2016-05-29 18:00:00'])[0]
        year = _idx.year
        print(_idx, year)
        path_radklim = (
            r"X:\exchange\seidel\RADKLIM\RADKLIM YW\RADKLIM_YW2017.002_%s.nc"
            % str(year))

        (lons, lats, xs, ys,
         pcp_data, df_dates) = read_nc_radklim(path_radklim)
        coorect_dates = df_dates.index.round('5min')#.sum().index

        # coorect_dates.intersection([_idx])
        df_new = pd.DataFrame(index=coorect_dates,
                              data=range(len(coorect_dates)),
                              columns=['Index'])

        idx_pxp_radar = np.where(df_new.index == _idx)[0][0]
        pcp_Radar = pcp_data[idx_pxp_radar - 12:idx_pxp_radar + 3]
        pcp_Radar_hourly = np.zeros(shape=lons.shape)

        for _xf in range(12):
            pcp_Radar_hourly += pcp_Radar[_xf].data

            #pcp_Radar_hourly += advection_correction(
            #    pcp_Radar[_xf: (_xf + 2)], T=5, t=1)
        pcp_Radar_hourly[mask] = -99
        # mask.shape
        # pcp_Radar.shape
        rw_maskes = np.ma.masked_array(pcp_Radar_hourly,
                                       pcp_Radar_hourly < 0.)
        ipcp, jpcp = np.where(rw_maskes >= 0)

        # plt.show()

        df_ = dwd_hdf5.get_pandas_dataframe_for_date(
            cmn_stns,
            event_date=str(_idx))

        # data_event = max_event.loc[_idx, cmn_stns].dropna()
        xcoords = in_dwd_df_coords_utm32.loc[df_.columns, 'lon']
        ycoords = in_dwd_df_coords_utm32.loc[df_.columns, 'lat']

        plt.close()

        plt.ioff()

        fig, ax1 = plt.subplots(1, 1,
                                figsize=(4, 4),
                                dpi=300)

        pcp = ax1.scatter(xcoords, ycoords, c=df_.values,
                          marker='o', s=df_.values*5,
                          vmin=0.,
                          cmap=plt.get_cmap('plasma_r'),
                          label='DWD gauges')
        fig.colorbar(pcp, ax=ax1, shrink=0.75, label='mm/hr', orientation='vertical')
        # fig2, ax2 = plt.subplots(1, 1,
        #                         figsize=(6, 4),
        #                         dpi=300)


        # rpcp = ax2.scatter(lons[ipcp, jpcp],
        #                    lats[ipcp, jpcp],
        #                    c=rw_maskes.data[ipcp, jpcp],
        #                    marker=',',
        #                    s=2,
        #                    vmin=0,
        #                    cmap=plt.get_cmap('YlGnBu')
        #                    )


        with collection(shp_path, "r") as input:
            for f in input:
                ax1.add_patch(
                    PolygonPatch(
                        f['geometry'], fc='grey',
                        ec='grey',
                        alpha=0.25,
                        zorder=3, fill=False))
                # ax2.add_patch(
                #     PolygonPatch(
                #         f['geometry'], fc='grey',
                #         ec='grey',
                #         alpha=0.5,
                #         zorder=3, fill=False))
        with collection(catch_path, "r") as input:
            for f in input:
                ax1.add_patch(
                    PolygonPatch(
                        f['geometry'], fc='k',
                        ec='k',
                        alpha=0.75,
                        zorder=3, fill=False))
                # ax2.add_patch(
                #     PolygonPatch(
                #         f['geometry'], fc='blue',
                #         ec='darkblue',
                #         alpha=0.75,
                #         zorder=3, fill=False))
        # plt.scatter()

        ax1.grid(alpha=0.25)
        # ax1.set_title('%s sum=%0.2f mm' % (_idx,  # data_event.sum() +
        #                                    df_.values.sum()))

        idx_b4 = str(_idx - pd.Timedelta(hours=2))
        idx_after = str(_idx + pd.Timedelta(hours=24))
        # ax2.scatter(df_discharge_data.loc[_idx, :].index,
        # df_discharge_data.loc[_idx, :].values,

        fig3, ax3 = plt.subplots(1, 1,
                                figsize=(4, 4),
                                dpi=300)
        ax4 = ax3.twinx()
        for _pcp in df_.columns:

            df_pcp = dwd_hdf5.get_pandas_dataframe_between_dates(
                _pcp,
                idx_b4, idx_after)

            xs = in_dwd_df_coords_utm32.loc[_pcp, 'lon']
            ys = in_dwd_df_coords_utm32.loc[_pcp, 'lat']

            if _pcp in pcp_stns_q.index:
                q_stn = str(pcp_stns_q.loc[_pcp, 'q_stn'].astype(int))

                if q_stn == '3470':
                    c_plot = 'b'
                elif q_stn == '3465':
                    c_plot = 'r'
                elif q_stn == '3421':
                    c_plot = 'g'
               # elif q_stn == '420':
               #     c_plot = 'm'
                else:
                    c_plot = 'grey'
            else:
                c_plot = 'grey'

            ax3.bar(
                x=df_pcp.loc[idx_b4:idx_after, _pcp].index,
                height=df_pcp.loc[idx_b4:idx_after,
                           _pcp].values.ravel(),
                color=c_plot,#'lightblue',
                edgecolor='darkblue',
                width=0.05,
                alpha=0.5
                #label='Precipitation'
                )

            # ax3.scatter(df_pcp.loc[idx_b4:idx_after, _pcp].index,
            #             df_pcp.loc[idx_b4:idx_after,
            #                        _pcp].values.ravel(),
            #             marker='x',
            #             c=c_plot)
            # ax3.plot(df_pcp.loc[idx_b4:idx_after, _pcp].index,
            #          df_pcp.loc[idx_b4:idx_after,
            #                     _pcp].values.ravel(),
            #          marker='x',
            #          c=c_plot)

            # ax3.vlines(_idx, ymin=0,
            #            ymax=np.max(df_pcp.loc[idx_b4:idx_after, :].values),
            #            linestyle='-.',
            #            color='r', linewidth=2)

            # ax1.scatter(xs, ys,
            #             marker='o',
            #             s=df_pcp.loc[_idx].values*5,

            #             c=df_pcp.loc[_idx].values,

            #             cmap=plt.get_cmap('jet')
            #             # vmin=0.,
            #             )
            # marker='X', s=100,
            # vmin=0.,
            # cmap=plt.get_cmap('Blues'))
        ax3.bar(
            x=df_pcp.loc[idx_b4:idx_after, _pcp].index,
            height=df_pcp.loc[idx_b4:idx_after,
                       _pcp].values.ravel(),
            color=c_plot,#'lightblue',
            edgecolor='darkblue',
            width=0.05,
            alpha=0.5,
            label='Pcp'
            )
        # fig4, ax4 = plt.subplots(1, 1,
                                # figsize=(4, 4),
                                # dpi=300)

        # marker='o', label='Q [m3/hr]')
        for _q in df_discharge_data.columns:
            if _q == '3470':
                c_plot = 'b'
            elif _q == '3465':
                c_plot = 'r'
            elif _q == '3421':
                c_plot = 'g'
            elif _q == '420':
                c_plot = 'm'
            else:
                c_plot = 'grey'

            # ax4.scatter(df_discharge_data.loc[idx_b4:idx_after, _q].index,
            #             df_discharge_data.loc[idx_b4:idx_after,
            #                                   _q].values.ravel(),
            #             marker='o',
            #             c=c_plot)
            ax4.plot(df_discharge_data.loc[idx_b4:idx_after, _q].index,
                     df_discharge_data.loc[idx_b4:idx_after,
                                           _q].values.ravel(),
                     marker='o',
                     c=c_plot,
                     alpha=0.5)

            # ax3.vlines(_idx, ymin=0,
            #            ymax=np.max(
            #                df_discharge_data.loc[idx_b4:idx_after, :].values),
            #            linestyle='-.',
            #            color='grey')

            ax1.scatter(qstns_csv.loc[int(_q), 'lon'],
                        qstns_csv.loc[int(_q), 'lat'],
                        marker='D',
                        s=100,
                        c=c_plot)

        ax4.plot(df_discharge_data.loc[idx_b4:idx_after, _q].index,
                 df_discharge_data.loc[idx_b4:idx_after,
                                       _q].values.ravel(),
                 marker='o',
                 c=c_plot,
                 label='Q',
                 alpha=0.5)
        ax1.scatter(qstns_csv.loc[int(_q), 'lon'],
                    qstns_csv.loc[int(_q), 'lat'],
                    marker='D',
                    s=100,
                    c=c_plot,
                    label='Q gauges')

        ax4.set_ylabel('Discharge Q [m3/s]')
        # ax3.set_title('Precipitation')
        ax3.set_ylabel('Pcp [mm/h]')

        # ax2.set_title('QPE - RadKlim')
        ax1.set_xlabel('Lon [°]')
        ax1.set_ylabel('Lat [°]')

        # ax2.grid(alpha=0.25)
        ax3.grid(alpha=0.25)
        ax4.grid(alpha=0.25)

        ax3.set_xlabel('Day-Hour')
        ax3.xaxis.set_major_locator(
            mdates.HourLocator(interval=6))
        ax3.xaxis.set_major_formatter(date_form)
        ax3.xaxis.set_minor_locator(
            mdates.HourLocator(interval=3))
        ax1.legend(loc=0)
        ax3.legend(loc=0)
        ax4.legend(loc='upper left', edgecolor=(0, 0, 0, 1.), facecolor=(1, 1, 1, 0.1))
        # ax4.set_xlabel('Day-Hour')
        # ax4.xaxis.set_major_locator(
        #     mdates.HourLocator(interval=6))
        # ax4.xaxis.set_major_formatter(date_form)
        # ax4.xaxis.set_minor_locator(
        #     mdates.HourLocator(interval=3))


        # ax4.set_xticks([])
        # f.tight_layout()
        fig.savefig(
            os.path.join(out_save_dir,
                         'max_event_sum_%s_dwd.png'
                         % str(_idx).replace(':', '_').replace(' ', '_')),
            bbox_inches='tight')

        # fig2.savefig(
        #     os.path.join(out_save_dir,
        #                  'max_event_sum_%s_dwd2.png'
        #                  % str(_idx).replace(':', '_').replace(' ', '_')),
        #     bbox_inches='tight')

        fig3.savefig(
            os.path.join(out_save_dir,
                         'max_event_sum_%s_dwd3.png'
                         % str(_idx).replace(':', '_').replace(' ', '_')),
            bbox_inches='tight')

        # fig4.savefig(
        #     os.path.join(out_save_dir,
        #                  'max_event_sum_%s_dwd4.png'
        #                  % str(_idx).replace(':', '_').replace(' ', '_')),
        #     bbox_inches='tight')
        plt.close()

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
