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
        r"\10_depth_function\01_cross_depth")

    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_1440min_gk3.h5")

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

    n_vecs = int(1e4)
    n_cpus = 3

    for _dwd_id in dwd_ids[10:15]:
        print(_dwd_id)
        df_stn = dwd_hdf5.get_pandas_dataframe(
            _dwd_id)

        (xdwd, ydwd) = (
            in_dwd_df_coords_utm32.loc[_dwd_id, 'X'],
            in_dwd_df_coords_utm32.loc[_dwd_id, 'Y'])

        distances, indices = dwd_points_tree.query(
            np.array([xdwd, ydwd]),
            k=10)

        stn_near = dwd_ids[indices[8]]

        (xdwd_near, ydwd_near) = (
            in_dwd_df_coords_utm32.loc[stn_near, 'X'],
            in_dwd_df_coords_utm32.loc[stn_near, 'Y'])

        print('sep distance', distances)
        df_stn_near = dwd_hdf5.get_pandas_dataframe(
            stn_near)

        cmn_idx = df_stn.index.intersection(
            df_stn_near.dropna(how='any').index)
        if cmn_idx.size > 0:
            df_stn_cmn = df_stn.loc[cmn_idx]
            df_stn_near_cmn = df_stn_near.loc[cmn_idx]
            #
            d10 = df_stn_cmn
            d11 = df_stn_near_cmn

            assert d10.shape == d11.shape

            ref_vals = d10.values.copy(order='c')
            test_vals = d11.values.copy(order='c')

            n_dims = ref_vals.shape[1]

            usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

            ref_vals.shape, test_vals.shape, usph_vecs.shape

            depths1 = depth_ftn_mp(ref_vals, test_vals, usph_vecs, n_cpus, 1)
            depths2 = depth_ftn_mp(test_vals, ref_vals, usph_vecs, n_cpus, 1)
            plt.scatter(ref_vals, depths1)
            plt.show()

            df = pd.DataFrame(index=cmn_idx, data=depths1)
            df.loc[:, 'depths2'] = depths2
            df_low1 = df[df[0] < 10]

            low_depths = (0 <= depths1) & (depths1 <= 10)
            ref_vals_low_depth = ref_vals[low_depths]
            test_vals_low_depth = test_vals[low_depths]

            low_depths2 = (0 <= depths2) & (depths2 <= 10)
            ref_vals_low_depth2 = ref_vals[low_depths2]
            test_vals_low_depth2 = test_vals[low_depths2]

            # idx_events_test_in_ref_vals = d10.index[low_depths2]
            # idx_events_test_test_vals = d11.index[low_depths2]

            plt.ioff()
            fig, (axs) = plt.subplots(
                1, 1, figsize=(12, 8), dpi=100, sharex=False)
            axs.scatter(ref_vals, test_vals,  # c=depths1,
                        marker='o', alpha=0.95,
                        s=5,
                        c='k')

            im = axs.scatter(ref_vals_low_depth, test_vals_low_depth,
                             c='r',
                             marker='+', s=30,)

            axs.scatter(ref_vals_low_depth2, test_vals_low_depth2,
                        c='b',
                        marker='+', s=30)
            axs.set_title('Two dimensions of precipitation series'
                          '%s-%s' % (_dwd_id, stn_near))

            axs.set_xlabel('X [mm/d]')

            axs.set_ylabel('Y [mm/d]')

            axs.set_xlim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
            axs.set_ylim([-0.2, max(d10.values.max(), d11.values.max()) + 1])
            # plt.axis('equal')
            plt.grid(alpha=0.25, linestyle='-.')
            plt.legend(loc=0)
            fig.subplots_adjust()
            plt.tight_layout()
            plt.savefig(os.path.join(out_save_dir,
                                     (r'%s_%s_2d_depth.png'
                                      % (_dwd_id, stn_near))),
                        bbox_inches='tight', pad_inches=.2)
            plt.close()
            # break
            # plt.show()
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
