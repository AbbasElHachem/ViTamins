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
        r"\10_depth_function\04_cross_depth_tranformed_pcp")

    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        % radar_loc)

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    n_vecs = int(1e4)
    n_cpus = 3

    start_date = '2000-01-01 00:00:00'
    end_date = '2019-12-31 23:55:00'

    remove_date = pd.DatetimeIndex(['2020-01-01 00:00:00'])
    upp_freq = '60min'
    low_freq = '30min'

    # transform to ranks

    # transform log-tramsform

    # box-cox transformation

    for _dwd_id in dwd_ids[:15]:
        print(_dwd_id)
        df_stn = dwd_hdf5.get_pandas_dataframe_between_dates(
            _dwd_id, event_start=start_date,
            event_end=end_date)

        df_low_freq = resampleDf(df_stn, low_freq)
        df_upp_freq = resampleDf(df_stn, upp_freq)

        df_low_freq = df_low_freq.loc[
            df_low_freq.index.difference(remove_date), :]
        df_upp_freq = df_upp_freq.loc[
            df_upp_freq.index.difference(remove_date), :]
        # df_60min_max = df_60min[
        # df_60min >= df_60min.quantile(.99)].dropna()
        # df_upp_freq = df_upp_freq
        start_events = df_low_freq.index.intersection(
            df_upp_freq.index - pd.Timedelta(
                minutes=int(low_freq.split('m')[0])))
        d10 = df_low_freq.loc[start_events, :]
        d11 = df_low_freq.loc[df_low_freq.index.intersection(
            df_upp_freq.index), :]

        cmn_range = min(len(d10.index), len(d11.index))
        d10 = d10.iloc[:cmn_range, :]
        d11 = d11.iloc[:cmn_range, :]

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
            1, 1, figsize=(12, 8), dpi=300, sharex=False)
        axs.scatter(ref_vals, test_vals,  # c=depths1,
                    marker='.', alpha=0.95,
                    c='grey')

        im = axs.scatter(ref_vals_low_depth, test_vals_low_depth,
                         c=depths1[low_depths],
                         cmap=plt.get_cmap('viridis'),
                         marker='x', s=60,
                         vmin=0.1,
                         vmax=10,
                         label='Data 1 in 2')

        axs.scatter(ref_vals_low_depth2, test_vals_low_depth2,
                    c=depths2[low_depths2],
                    cmap=plt.get_cmap('viridis'),
                    vmin=0.1,
                    vmax=10,
                    marker='o', s=60,
                    label='Data 2 in 1')

        # axs.scatter(ref_vals_low_depth100,
        #             test_vals_low_depth100,
        #             c='r',
        #             label='d2=0',
        #             marker='1', s=30)
        #
        # axs.scatter(ref_vals_low_depth200,
        #             test_vals_low_depth200,
        #             c='m',
        #             label='d1=0',
        #             marker='3', s=30)

        plt.colorbar(im, ax=axs, shrink=0.75, label='0 < d <= 10',
                     ticks=np.arange(0, 11, 1))
        axs.set_title('Cross-depth Station %s' % _dwd_id)

        axs.set_xlabel('Values 0->%s [mm/%s]'
                       % (low_freq, low_freq))

        axs.set_ylabel('Values %s->%s [mm/%s]'
                       % (low_freq, upp_freq, low_freq))

        axs.set_xlim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
        axs.set_ylim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
        # plt.axis('equal')
        plt.grid(alpha=0.25, linestyle='-.')
        plt.legend(loc=0)
        fig.subplots_adjust()
        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir,
                                 (r'%s_cross_depth_%s_%s.png'
                                  % (_dwd_id, low_freq, upp_freq))),
                    bbox_inches='tight', pad_inches=.2)
        plt.close()
        # break
        # plt.show()

        #===============================================================
        #
        #===============================================================

        for ii, (_ix, dd) in enumerate(zip(idx_events_ref_vals,
                                           depths1[low_depths])):
            # break
            plt.ioff()
            fig, (axs) = plt.subplots(
                1, 1, figsize=(12, 8), dpi=200, sharex=False)
            start_ = _ix - pd.Timedelta(
                minutes=int(low_freq.split('m')[0]))
            end_ = _ix + pd.Timedelta(
                minutes=int(low_freq.split('m')[0]))

            start_str = str(start_).replace(':', '_').replace(
                '-', '_').replace(' ', '_')
            axs.plot(df_stn.loc[start_:_ix].index,
                     df_stn.loc[start_:_ix].values,  # c=depths1,
                     marker='x', alpha=0.95,
                     c='b')

            axs.plot(df_stn.loc[_ix:end_].index,
                     df_stn.loc[_ix:end_].values,  # c=depths1,
                     marker='o', alpha=0.95,
                     c='r')
            axs.set_title('%s-%s depth d=%d' % (start_, end_, dd))
            axs.set_ylabel('Pcp [mm/5min]')
            # plt.legend(loc=0)
            # fig.subplots_adjust()
            # plt.tight_layout()
            # plt.show()
            plt.grid(alpha=0.25, linestyle='-.')

            plt.savefig(os.path.join(out_save_dir,
                                     (r'%s_cross_depth_%d_%s_%s_%s.png'
                                      % (_dwd_id, ii, start_str,
                                         low_freq, upp_freq))),
                        bbox_inches='tight', pad_inches=.2)
            plt.close()

        for ii, (_ix, dd) in enumerate(zip(idx_events_test_vals,
                                           depths2[low_depths2])):
            # break
            plt.ioff()
            fig, (axs) = plt.subplots(
                1, 1, figsize=(12, 8), dpi=100, sharex=False)
            start_ = _ix - pd.Timedelta(
                minutes=int(low_freq.split('m')[0]))
            end_ = _ix + pd.Timedelta(
                minutes=int(low_freq.split('m')[0]))

            start_str = str(start_).replace(':', '_').replace(
                '-', '_').replace(' ', '_')
            axs.plot(df_stn.loc[start_:_ix].index,
                     df_stn.loc[start_:_ix].values,  # c=depths1,
                     marker='x', alpha=0.95,
                     c='b')

            axs.plot(df_stn.loc[_ix:end_].index,
                     df_stn.loc[_ix:end_].values,  # c=depths1,
                     marker='o', alpha=0.95,
                     c='r')
            axs.set_title('%s-%s depth d=%d' % (start_, end_, dd))
            axs.set_ylabel('Pcp [mm/5min]')
            # plt.legend(loc=0)
            # fig.subplots_adjust()
            # plt.tight_layout()
            # plt.show()
            plt.grid(alpha=0.25, linestyle='-.')

            plt.savefig(os.path.join(out_save_dir,
                                     (r'%s_cross_depth_%d_%s_%s_%s.png'
                                      % (_dwd_id, ii, start_str,
                                         low_freq, upp_freq))),
                        bbox_inches='tight', pad_inches=.2)
            plt.close()
        break
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
