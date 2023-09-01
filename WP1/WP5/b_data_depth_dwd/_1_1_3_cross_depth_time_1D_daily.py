'''
@author: Abbas-Uni-Stuttgart

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

plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'axes.labelsize': 14})

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
        r"\10_depth_function\01_cross_depth\one_stn_time_period")
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_1440min_gk3.h5"
    )

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    n_vecs = int(1e4)
    n_cpus = 3

    # start_date = '1900-01-01 00:00:00'
    # end_date = '2019-12-31 23:55:00'

    # remove_date = pd.DatetimeIndex(['2020-01-01 00:00:00'])

    for _dwd_id in dwd_ids[:15]:
        print(_dwd_id)
        df_stn = dwd_hdf5.get_pandas_dataframe(_dwd_id)

        if df_stn.index.size % 2 != 0:
            df_stn = df_stn.iloc[:-1, :]

        df_stn_parts = np.array_split(df_stn, 2)

        df_part1 = df_stn_parts[0]
        df_part2 = df_stn_parts[1]

        cmn_range = min(len(df_part1.index), len(df_part2.index))
        d10 = df_part1.iloc[:cmn_range, :]
        d11 = df_part2.iloc[:cmn_range, :]

        assert d10.shape == d11.shape

        ref_vals = d10.values.copy(order='c')
        test_vals = d11.values.copy(order='c')

        n_dims = ref_vals.shape[1]

        usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

        ref_vals.shape, test_vals.shape, usph_vecs.shape

        depths1 = depth_ftn_mp(ref_vals, test_vals, usph_vecs, n_cpus, 1)
        depths2 = depth_ftn_mp(test_vals, ref_vals, usph_vecs, n_cpus, 1)

        depth00 = np.where(depths1 == 0)
        ref_vals_low_depth100 = ref_vals[depth00]
        test_vals_low_depth100 = test_vals[depth00]

        # ref_vals_low_depth00.max()

        depth10 = depths1[depths1 == 1]
        ref_vals_low_depth10 = ref_vals[depth10]
        test_vals_low_depth10 = test_vals[depth10]
        # depths.shape
        low_depths = (0 < depths1) & (depths1 <= 10)
        ref_vals_low_depth = ref_vals[low_depths]
        test_vals_low_depth = test_vals[low_depths]

        idx_events_ref_vals = d10.index[low_depths]
        idx_events_test_vals = d11.index[low_depths]

        depth200 = np.where(depths2 == 0)
        ref_vals_low_depth200 = ref_vals[depth200]
        test_vals_low_depth200 = test_vals[depth200]

        depth20 = depths2[depths2 == 1]
        ref_vals_low_depth20 = ref_vals[depth20]
        test_vals_low_depth20 = test_vals[depth20]

        low_depths2 = (0 < depths2) & (depths2 <= 10)
        ref_vals_low_depth2 = ref_vals[low_depths2]
        test_vals_low_depth2 = test_vals[low_depths2]

        # idx_events_test_in_ref_vals = d10.index[low_depths2]
        # idx_events_test_test_vals = d11.index[low_depths2]

        plt.ioff()
        fig, (axs) = plt.subplots(
            1, 1, figsize=(12, 8), dpi=100, sharex=False)
        axs.scatter(ref_vals, test_vals,  # c=depths1,
                    marker='.', alpha=0.95,
                    c='grey')

        im = axs.scatter(ref_vals_low_depth, test_vals_low_depth,
                         c=depths1[low_depths],
                         cmap=plt.get_cmap('viridis'),
                         marker='x', s=30,
                         vmin=0.1,
                         vmax=10,
                         label='Data 1')

        axs.scatter(ref_vals_low_depth2, test_vals_low_depth2,
                    c=depths2[low_depths2],
                    cmap=plt.get_cmap('viridis'),
                    vmin=0.1,
                    vmax=10,
                    marker='o', s=30,
                    label='Data 2')

        # axs.scatter(ref_vals_low_depth100,
        #             test_vals_low_depth100,
        #             c='r',
        #             marker='1', s=30)
        #
        # axs.scatter(ref_vals_low_depth200,
        #             test_vals_low_depth200,
        #             c='m',
        #             marker='3', s=30)

        plt.colorbar(im, ax=axs, shrink=0.75, label='0 < d <= 10',
                     ticks=np.arange(0, 11, 1))
        axs.set_title('Cross-depth Station %s' % _dwd_id)

        axs.set_xlabel('Values %s->%s [mm/d]'
                       % (d10.index[0].year, d10.index[-1].year))

        axs.set_ylabel('Values %s->%s [mm/d]'
                       % (d11.index[0].year, d11.index[-1].year))

        axs.set_xlim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
        axs.set_ylim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
        # plt.axis('equal')
        plt.grid(alpha=0.25, linestyle='-.')
        plt.legend(loc=0)
        fig.subplots_adjust()
        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir,
                                 (r'%s_cross_depth_daily_periods.png'
                                  % (_dwd_id))),
                    bbox_inches='tight', pad_inches=.2)
        plt.close()
        # break
        # plt.show()

        #===============================================================
        #
        #===============================================================

        # for ii, (_ix, dd) in enumerate(zip(idx_events_ref_vals,
        #                                    depths1[low_depths])):
        #     # break
        #     plt.ioff()
        #     fig, (axs) = plt.subplots(
        #         1, 1, figsize=(12, 8), dpi=100, sharex=False)
        #     start_ = _ix - pd.Timedelta(
        #         minutes=int('1440min'.split('m')[0]))
        #     end_ = _ix + pd.Timedelta(
        #         minutes=int('1440min'.split('m')[0]))
        #
        #     start_str = str(start_).replace(':', '_').replace(
        #         '-', '_').replace(' ', '_')
        #     axs.plot(df_stn.loc[start_:_ix].index,
        #              df_stn.loc[start_:_ix].values,  # c=depths1,
        #              marker='x', alpha=0.95,
        #              c='b')
        #
        #     axs.plot(df_stn.loc[_ix:end_].index,
        #              df_stn.loc[_ix:end_].values,  # c=depths1,
        #              marker='o', alpha=0.95,
        #              c='r')
        #     axs.set_title('%s-%s depth d=%d' % (start_, end_, dd))
        #     axs.set_ylabel('Pcp [mm/d]')
        #     # plt.legend(loc=0)
        #     # fig.subplots_adjust()
        #     # plt.tight_layout()
        #     # plt.show()
        #     plt.grid(alpha=0.25, linestyle='-.')
        #
        #     plt.savefig(os.path.join(out_save_dir,
        #                              (r'%s_depth_%d_%s.png'
        #                               % (_dwd_id, ii, start_str))),
        #                 bbox_inches='tight', pad_inches=.2)
        #     plt.close()
        #
        # for ii, (_ix, dd) in enumerate(zip(idx_events_test_vals,
        #                                    depths2[low_depths2])):
        #     # break
        #     plt.ioff()
        #     fig, (axs) = plt.subplots(
        #         1, 1, figsize=(12, 8), dpi=100, sharex=False)
        #     start_ = _ix - pd.Timedelta(
        #         minutes=int('1440min'.split('m')[0]))
        #     end_ = _ix + pd.Timedelta(
        #         minutes=int('1440min'.split('m')[0]))
        #
        #     start_str = str(start_).replace(':', '_').replace(
        #         '-', '_').replace(' ', '_')
        #     axs.plot(df_stn.loc[start_:_ix].index,
        #              df_stn.loc[start_:_ix].values,  # c=depths1,
        #              marker='x', alpha=0.95,
        #              c='b')
        #
        #     axs.plot(df_stn.loc[_ix:end_].index,
        #              df_stn.loc[_ix:end_].values,  # c=depths1,
        #              marker='o', alpha=0.95,
        #              c='r')
        #     axs.set_title('%s-%s depth d=%d' % (start_, end_, dd))
        #     axs.set_ylabel('Pcp [mm/d]')
        #     # plt.legend(loc=0)
        #     # fig.subplots_adjust()
        #     # plt.tight_layout()
        #     # plt.show()
        #     plt.grid(alpha=0.25, linestyle='-.')
        #
        #     plt.savefig(os.path.join(out_save_dir,
        #                              (r'%s_cross_depth_%d_%s.png'
        #                               % (_dwd_id, ii, start_str))),
        #                 bbox_inches='tight', pad_inches=.2)
        #     plt.close()
        # break
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
