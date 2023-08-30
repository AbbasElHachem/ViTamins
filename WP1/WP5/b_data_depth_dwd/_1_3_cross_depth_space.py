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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.dates as mdates
# plt.rcParams.update({'font.size': 14})
# plt.rcParams.update({'axes.labelsize': 14})
from matplotlib.dates import DateFormatter
date_form = DateFormatter("%Y-%d-%m %H")
# date_form2 = DateFormatter("%y-%b")
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP1\a_dwd\00_get_data'
sys.path.append(modulepath)

from _00_0_functions import resampleDf

DEBUG_FLAG = True


def main():

    # =============================================================
    radar_loc = 'Hannover'  # Hannover  Tuerkheim  Feldberg
    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\01_cross_depth")
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    # In[2]:
    # path_to_dwd_hdf5 = (
    #     r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
    #     r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
    #     % radar_loc)

    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        # r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        r"\all_dwd_60min_1995_2021.h5"
        # r"\%s_dwd_stns_1440min_1880_2019.h5"
    )

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

    # neighbor_to_chose = 1
    n_vecs = int(1e4)
    n_cpus = 7

    start_date = '1900-01-01 00:00:00'
    end_date = '2019-12-31 23:00:00'

    remove_date = pd.DatetimeIndex(['2020-01-01 00:00:00'])
    # upp_freq = '60mi6'
    low_freq = '60min'

    # for _dwd_id in dwd_ids:

    _dwd_id = 'P00044'

    print(_dwd_id)

    df_stn = dwd_hdf5.get_pandas_dataframe(
        _dwd_id).dropna(how='all')

    (xdwd, ydwd) = (
        in_dwd_df_coords_utm32.loc[_dwd_id, 'X'],
        in_dwd_df_coords_utm32.loc[_dwd_id, 'Y'])

    distances, indices = dwd_points_tree.query(
        np.array([xdwd, ydwd]),
        k=10)

    stn_near = list(np.array(dwd_ids)[indices[1:]])

    (xdwd_near, ydwd_near) = (
        in_dwd_df_coords_utm32.loc[stn_near, 'X'],
        in_dwd_df_coords_utm32.loc[stn_near, 'Y'])
    plt.ioff()
    plt.scatter(xdwd, ydwd, c='r', marker='X')
    plt.scatter(xdwd_near, ydwd_near, c='b', marker='o')
    plt.grid(alpha=0.5)
    plt.axis('equal')
    plt.savefig(os.path.join(out_save_dir,
                             (r'%s_loc.png'
                              % (_dwd_id))),
                bbox_inches='tight', pad_inches=.2)
    plt.close()
    # min_dist_ppt_dwd = np.round(
    #     distances[neighbor_to_chose], 2)

    print('sep distance', distances)
    df_stn_near = dwd_hdf5.get_pandas_dataframe_between_dates(
        stn_near, event_start=df_stn.index[0],
        event_end=df_stn.index[-1])

    # cmn_idx = df_stn.index.intersection(
    #     df_stn_near.dropna(how='any').index)
    #
    # df_stn_cmn = df_stn.loc[cmn_idx]
    # df_stn_near_cmn = df_stn_near.loc[cmn_idx]
    # #
    # df_stn_cmn_res = resampleDf(
    #     df_stn_cmn, low_freq)
    # df_stn_near_cmn_res = resampleDf(
    #     df_stn_near_cmn, low_freq)
    #
    # df_low_freq = df_low_freq.loc[
    #     df_low_freq.index.difference(remove_date), :]
    # df_upp_freq = df_upp_freq.loc[
    #     df_upp_freq.index.difference(remove_date), :]
    # # df_60min_max = df_60min[
    # # df_60min >= df_60min.quantile(.99)].dropna()
    # # df_upp_freq = df_upp_freq
    # start_events = df_low_freq.index.intersection(
    #     df_upp_freq.index - pd.Timedelta(
    #         minutes=int(low_freq.split('m')[0])))
    # d10 = df_low_freq.loc[start_events, :]
    # d11 = df_low_freq.loc[df_low_freq.index.intersection(
    #     df_upp_freq.index), :]
    #
    # d10 = df_stn_cmn_res
    df_stn_near.columns
    for ii, _stn in enumerate(df_stn_near.columns):
        print(_stn)
        _stn = 'P03791'
        d11 = df_stn_near.loc[:, _stn].dropna()

        if d11.values.size > 0:
            d11_df = pd.DataFrame(
                index=d11.index, data=d11.values, columns=[_stn])
            # break
            cmn_range = df_stn.index.intersection(
                d11_df.index)

            d10 = df_stn.loc[cmn_range, :]
            d11 = d11_df.loc[cmn_range, :]

            assert d10.shape == d11.shape

            ref_vals = d10.values.copy(order='c')
            test_vals = d11.values.copy(order='c')

            n_dims = 1

            usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)
            print(ref_vals.shape, test_vals.shape, usph_vecs.shape)
            print('calculating depth')
            depths1 = depth_ftn_mp(
                ref_vals, test_vals, usph_vecs, n_cpus, 1)
            depths2 = depth_ftn_mp(
                test_vals, ref_vals, usph_vecs, n_cpus, 1)
            print('done calculating depth')
            # depth_appearing = depths1 == 0
            # appe_ = d10.index[depth_appearing]
            df_deepth = pd.DataFrame(data=depths1,
                                     columns=['ref_in_test'],
                                     index=d11.index)
            df_deepth.loc[:, 'test_in_ref'] = depths2

            df_deepth_0 = df_deepth[df_deepth['ref_in_test'] == 0]

            df_deepth_0_appear = df_deepth_0[
                df_deepth_0['test_in_ref'] > 0]

            # d10.loc[df_deepth_0_appear.index].plot()
            # d11.loc[df_deepth_0_appear.index].plot()
            #
            # plt.show()
#

            df_deepth_0_2 = df_deepth[df_deepth['test_in_ref'] == 0]

            df_deepth_0_2_appear = df_deepth_0_2[df_deepth_0_2['ref_in_test'] > 0]

            depth10 = depths1[depths1 < 1]
            ref_vals_low_depth10 = ref_vals[depth10]
            test_vals_low_depth10 = test_vals[depth10]
            # depths.shape
            low_depths = (1 < depths1) & (depths1 <= 10)
            ref_vals_low_depth = ref_vals[low_depths]
            test_vals_low_depth = test_vals[low_depths]

            idx_events_ref_vals = d10.index[low_depths]
            idx_events_test_vals = d11.index[low_depths]

            depth20 = depths2[depths2 < 1]
            ref_vals_low_depth20 = ref_vals[depth20]
            test_vals_low_depth20 = test_vals[depth20]

            low_depths2 = (1 < depths2) & (depths2 <= 10)
            ref_vals_low_depth2 = ref_vals[low_depths2]
            test_vals_low_depth2 = test_vals[low_depths2]

            idx_events_test_in_ref_vals = d10.index[low_depths2]
            idx_events_test_test_vals = d11.index[low_depths2]

            plt.ioff()
            fig, (axs) = plt.subplots(
                1, 1, figsize=(6, 4), dpi=300, sharex=False)

            axs.scatter(ref_vals, test_vals,  # c='k',
                        marker='.', alpha=0.5,
                        edgecolor='k',
                        facecolor='gray'
                        # cmap=plt.get_cmap('spring')
                        )

            im = axs.scatter(ref_vals_low_depth, test_vals_low_depth,
                             c=depths1[low_depths],
                             cmap=plt.get_cmap('inferno'),
                             marker='x', s=60,
                             vmin=0.1,
                             vmax=10,
                             label='%s' % _dwd_id)

            axs.scatter(ref_vals_low_depth2, test_vals_low_depth2,
                        c=depths2[low_depths2],
                        cmap=plt.get_cmap('inferno'),
                        vmin=0.1,
                        vmax=10,
                        marker='o', s=60,
                        label='%s' % _stn)

            pt_d, pt_ix1, pt_ix2 = np.intersect1d(depths1[low_depths],
                                                  depths2[low_depths2],
                                                  return_indices=True)
            if len(pt_ix2) > 0:
                circle2 = plt.Circle(
                    (ref_vals_low_depth2[pt_ix2[0]],
                     test_vals_low_depth2[pt_ix2[0]]),
                    0.95, color='r', fill=False)
                axs.add_patch(circle2)
            # axs.scatter(ref_vals_low_depth10,
            #             test_vals_low_depth10,
            #             c='r',
            #             marker='1', s=50)
            #
            # axs.scatter(ref_vals_low_depth20,
            #             test_vals_low_depth20,
            #             c='r',
            #             marker='2', s=50)
            # axs.scatter(d10.loc[df_deepth_0_appear.index].values,
            #             d11.loc[df_deepth_0_appear.index].values,
            #             label='appear',
            #             c='r')
            # axs.scatter(d10.loc[df_deepth_0_2_appear.index].values,
            #             d11.loc[df_deepth_0_2_appear.index].values,
            #             label='disappear',
            #             c='b')
            plt.colorbar(im, ax=axs, shrink=0.75, label='1 < d <= 10',
                         ticks=np.arange(1, 11, 1))

            axs.set_title('Cross-depth Station %s-%s'
                          % (_dwd_id, _stn))

            axs.set_xlabel('%s [mm/%s]'
                           % (_dwd_id, low_freq))

            axs.set_ylabel('%s [mm/%s]'
                           % (_stn, low_freq))

            axs.set_xlim(
                [-0.2, max(d10.values.max(), d11.values.max()) + 1])
            axs.set_ylim(
                [-0.2, max(d10.values.max(), d11.values.max()) + 1])
            # plt.axis('equal')
            plt.grid(alpha=0.25, linestyle='-.')
            plt.legend(loc=0)
            fig.subplots_adjust()
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir,
                                     (r'%s_%s_cross_depth_%s.png'
                                      % (_dwd_id, _stn, low_freq))),
                        bbox_inches='tight', pad_inches=.2)
            plt.close()
            # break
        # plt.show()

        #===============================================================
        #
        #===============================================================

            for ii, (_ix, dd) in enumerate(zip(idx_events_test_test_vals,
                                               depths2[low_depths2])):
                if dd < 10:
                    # break
                    print(_ix)
                    plt.ioff()
                    fig, (axs) = plt.subplots(
                        1, 1, figsize=(5, 3), dpi=300, sharex=False)
                    start_ = _ix - pd.Timedelta(
                        minutes=int(low_freq.split('m')[0]))
                    end_ = _ix + pd.Timedelta(
                        minutes=int(low_freq.split('m')[0]))

                    start_str = str(start_).replace(':', '_').replace(
                        '-', '_').replace(' ', '_')

                    # axs.plot(df_stn_near.loc[start_:end_,:].index,
                    #          df_stn_near.loc[start_:end_,
                    #                              :].values,  # c=depths1,
                    #          marker='.', alpha=0.75,
                    #          c='g',
                    #          label='Neighbors')
                    axs.plot(df_stn_near.loc[start_:end_].index,
                             # c=depths1,
                             df_stn_near.loc[start_:end_].values,
                             marker='.', alpha=0.5,
                             c='g')

                    axs.plot(df_stn_near.loc[start_:end_, _stn].index,
                             df_stn_near.loc[start_:end_,
                                             _stn].values,  # c=depths1,
                             marker='o', alpha=0.75,
                             label='%s' % _stn,
                             c='r')
                    axs.plot(df_stn.loc[start_:end_, _dwd_id].index,
                             # c=depths1,
                             df_stn.loc[start_:end_, _dwd_id].values,
                             marker='x', alpha=0.75,
                             label='%s' % _dwd_id,
                             c='b')

                    # axs.vlines(_ix, ymin=0, ymax=14, color='m', linestyle='-',
                    # alpha=0.75)
                    axs.set_title('%s-%s depth d=%d' % (start_, end_, dd))
                    axs.set_ylabel('Pcp [mm/60min]')

                    axs.xaxis.set_major_locator(
                        mdates.HourLocator(interval=1))
                    axs.xaxis.set_major_formatter(date_form)
                    axs.xaxis.set_minor_locator(
                        mdates.MinuteLocator(interval=30))
                    # plt.legend(loc=0)
                    fig.subplots_adjust()
                    plt.tight_layout()
                    plt.legend(loc=0)
                    plt.grid(alpha=0.25, linestyle='-.')

                    plt.savefig(os.path.join(out_save_dir,
                                             (r'%s_%s_depth_%d_%s_%s.png'
                                              % (_dwd_id, _stn, dd, start_str,
                                                 low_freq))),
                                bbox_inches='tight', pad_inches=.2)
                    plt.close()

    return

    # all_stns = dwd_hdf5.get_pandas_dataframe_between_dates(
    # dwd_ids, event_start=start_date,
    # event_end=end_date)
    # all_other_stns = np.setdiff1d(dwd_ids, dwd_ids[1])
    # df_other_stns = all_stns.loc[
    # data_1.index, all_other_stns].dropna(how='any', axis=1)
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
