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
from sklearn.decomposition import PCA
from sklearn.utils._testing import assert_allclose
from altair_transform.utils.tests.test_data import df
from pandas.tests.series.test_reductions import test_validate_sum_initial

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
        r"\10_depth_function\04_cross_depth_all_dwd")

    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        # r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        r"\%s_dwd_stns_1440min_1880_2019.h5"
        % radar_loc)

    n_vecs = int(1e5)
    n_cpus = 5
    thr = 3
    plot_timesreies_events = False
    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()

    dwd_coords = dwd_hdf5.get_coordinates(ids=dwd_ids)

    in_dwd_df_coords_utm32 = pd.DataFrame(
        index=dwd_ids,
        data=dwd_coords['easting'], columns=['X'])
    y_dwd_coords = dwd_coords['northing']
    in_dwd_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords

    # date_range = pd.date_range(start='1950', end='2019', freq='D')
    # df_all_pcp_area = pd.DataFrame(index=date_range,  # range(0,int(1e4), 1),
    #                                columns=['sum_stns'])
    #
    # for ii, _date in enumerate(df_all_pcp_area.index):
    #     print(ii, len(df_all_pcp_area.index))
    #     df_stn = dwd_hdf5.get_pandas_dataframe_for_date(
    #         dwd_ids, _date).dropna(how='all')
    #
    #     # break
    #     if df_stn.index.size > 0:
    #         date_sum = df_stn.sum(axis=1)
    #         date_sum_vs_nbr_stns = date_sum / len(df_stn.columns)
    #         df_all_pcp_area.loc[_date,
    #                             'sum_stns'] = date_sum_vs_nbr_stns.values.ravel()
    #         # break
    # df_all_pcp_area_v = [np.round(v[0], 2)
    #                      for v in df_all_pcp_area.values.ravel()]
    #
    # df_all_pcp_area.loc[:, 'sum_stns'] = df_all_pcp_area_v
    # df_all_pcp_area.to_csv(os.path.join(out_save_dir, 'pcp_daily_sum_.csv'),
    #                        sep=';')
    df_all_pcp_area = pd.read_csv(os.path.join(out_save_dir, 'pcp_daily_sum_.csv'),
                                  sep=';', index_col=0,
                                  parse_dates=True, infer_datetime_format=True)
    n_dims = df_all_pcp_area.shape[1]

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    depths1 = depth_ftn_mp(df_all_pcp_area.values.copy(order='c').astype(float),
                           df_all_pcp_area.values.copy(
                               order='c').astype(float),
                           usph_vecs, n_cpus, 1)

    # ids_low_de_1_2 = np.where(depths1 <= 10)[0]
    plt.ioff()
    fig = plt.figure(figsize=(12, 8), dpi=100)
    im0 = plt.scatter(df_all_pcp_area.index,
                      df_all_pcp_area.values, c=depths1,
                      cmap=plt.get_cmap('jet'),
                      vmin=0, vmax=20)

    fig.colorbar(im0, shrink=0.8, label='10<Depth')
    plt.xlabel('Time')
    plt.ylabel('Pcp')
    plt.grid()
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_save_dir,
                             'depth_pcp_stn_area.png'))
    plt.close()
    pass


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
