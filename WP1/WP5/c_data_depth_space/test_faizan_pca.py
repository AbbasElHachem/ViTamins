'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

plt.ioff()

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.3f}'.format})


def depth_ftn(x, y, ei):
    mins = x.shape[0] * np.ones((y.shape[0],))  # initial value

    for i in ei:  # iterate over unit vectors
        d = np.dot(i, x.T)  # scalar product

        dy = np.dot(i, y.T)  # scalar product

        # argsort gives the sorting indices then we used it to sort d
        d = d[np.argsort(d)]

        dy_med = np.median(dy)
        dy = ((dy - dy_med) * (1 - (1e-10))) + dy_med

        # find the index of each project y in x to preserve order
        numl = np.searchsorted(d, dy)
        # numl is number of points less then projected y
        numg = d.shape[0] - numl

        # find new min
        mins = np.min(
            np.vstack([mins, np.min(np.vstack([numl, numg]), axis=0)]), 0)

    return mins


if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(r'P:\Synchronize\IWS\2016_DFG_SPATE\data')

    path_to_opt = main_dir / r'ncar_cp_classi_obj_2_all_cats_Neckar\ob01000_c19800101_19891231_v19900101_20091231_cps12_idxsct10_pl010_trda0995_mxn15000000_mxk01000000_mxm00020000_rp03'
#     path_to_opt = main_dir / r'ncep_cp_classi_obj_2_test\ob01000000_c19800101_19891231_v19900101_20091231_cps12_idxsct40_pl015_trda0990_mxn03000000_mxm00008000_rp00'

    anomaly_pickle = path_to_opt / r'anomaly.pkl'
    cp_assign_path = path_to_opt / r'cp_assign_all.pkl'
    cp_wettness_path = path_to_opt / r'wettness_idxs_all.pkl'

    out_dir = path_to_opt / 'eigen_value_stuff_5D'

    fig_size = (15, 8)

    os.chdir(str(main_dir))

    if not out_dir.exists():
        out_dir.mkdir()

    with open(str(anomaly_pickle), 'rb') as _pkl_hdl:
        anomaly_obj = pickle.load(_pkl_hdl)
        vals_tot_anom = anomaly_obj.vals_tot_anom
        dates_tot = anomaly_obj.times_tot
        dates_tot = pd.DatetimeIndex(dates_tot.date)
    print('vals_tot_anom shape:', vals_tot_anom.shape)

    with open(str(cp_assign_path), 'rb') as _pkl_hdl:
        assign_cps = pickle.load(_pkl_hdl)
#         cp_rules = assign_cps.cp_rules
        sel_cps_arrs = assign_cps.sel_cps_arr
        n_cps = assign_cps.n_cps

    with open(str(cp_wettness_path), 'rb') as _pkl_hdl:
        wettness_obj = pickle.load(_pkl_hdl)
        ppt_arr = wettness_obj.ppt_arr[:, 2]

    cps_list = [9, 10, 11]

    cp_no = 9

    hist_bins = np.array([0, 1, 2, 3, 4, 5, 10000])

    assert sel_cps_arrs.shape[0] == vals_tot_anom.shape[0]

    labs_list = ['summer', 'winter']
    _summer_months = np.arange(5, 11, 1)
    _winter_months = np.array([1, 2, 3, 4, 11, 12])

    months_list = [_summer_months, _winter_months]

    _ppt_arr_thresh_idxs = (ppt_arr >= 20)

    # stuff for depth ftn
    ndims = 5
    ei = -1 + (2 * np.random.randn(100000000, ndims))
    normc = np.sqrt((ei ** 2).sum(axis=1))
    norm_idxs = normc < 1.0
    normc = normc[norm_idxs]
    print('final ei shape:', normc.shape)
    ei = ei[norm_idxs] / normc[:, None]
    # end depth ftn prep

    b_j_idxs_list = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

    for lab_i, season in enumerate(labs_list):
        print('\n%s' % season)
        idxs = np.zeros(vals_tot_anom.shape[0], dtype=bool)
        for month in months_list[lab_i]:
            idxs = idxs | (dates_tot.month == month)

        winter_cps = sel_cps_arrs[idxs]
        # idxs = idxs & (sel_cps_arrs == cp_no)
        curr_ppt_arr_thresh_idxs = _ppt_arr_thresh_idxs[idxs]

        anom = vals_tot_anom[idxs, :]

        corr_mat = np.corrcoef(anom.T)
        eig_val, eig_mat = np.linalg.eig(corr_mat)
        sort_idxs = np.argsort(eig_val)
        eig_val_sort = eig_val[sort_idxs]
        eig_val_sum = eig_val_sort.sum()
        eig_val_cum_sum = np.cumsum(eig_val_sort[::-1]) / eig_val_sum
        # print(eig_val_cum_sum)

        b_j_s = np.dot(anom, eig_mat)
#         b_j_s = np.dot(anom, eig_mat[:, sort_idxs])

        # print(b_j_s.shape)
        # print(b_j_s)

        # print(sel_cps_arrs.shape, b_j_s.shape)

        for b_j_idx in b_j_idxs_list:

            plt.figure(figsize=(fig_size))
            # plt.plot(b_j_s[:, 0], alpha=0.7)
            plt.scatter(b_j_s[:, b_j_idx[0]],
                        b_j_s[:, b_j_idx[1]], alpha=0.3, label='all')
#             plt.scatter(b_j_s[:, b_j_idx[0]][winter_cps == cp_no],
#                         b_j_s[:, b_j_idx[1]][winter_cps == cp_no],
#                         alpha=0.3,
#                         label='ge_ppt')
            plt.scatter(b_j_s[:, b_j_idx[0]][curr_ppt_arr_thresh_idxs],
                        b_j_s[:, b_j_idx[1]][curr_ppt_arr_thresh_idxs],
                        alpha=0.3,
                        label='ge_ppt')
            plt.title('%s b_j_s: %d, %d' % (season, *b_j_idx))
            plt.grid()
            plt.legend()
            # plt.show()
            plt.savefig(str(out_dir / ('%s_b_j_s_%d_%d.png' %
                                       (season, *b_j_idx))), bbox_inches='tight')
            plt.close()

        xs = b_j_s[:, :ndims]
        ys = b_j_s[:, :ndims][curr_ppt_arr_thresh_idxs, :]

        mins_heavy = depth_ftn(ys, xs, ei)
        mins_heavy_heavy = depth_ftn(ys, ys, ei)
        # mins_all = depth_ftn(xs, xs, ei)
        print('%s y vs. y min and max' %
              season, mins_heavy.min(), mins_heavy.max())
        print('mins_heavy_heavy histogram:\n', np.histogram(
            mins_heavy_heavy, hist_bins), sep='')

        plt.figure(figsize=fig_size)
        # plt.hist(mins_all, bins=hist_bins, alpha=0.3, normed=True, label='x vs. x')
        plt.hist(mins_heavy, bins=hist_bins, alpha=0.3,
                 normed=True, label='y vs. x')
        plt.hist(mins_heavy_heavy, bins=hist_bins,
                 alpha=0.3, normed=True, label='y vs. y')
        plt.xticks(hist_bins[:-1] + 0.5, hist_bins[:-1] + 0.5)
        plt.xlim(hist_bins[0], hist_bins[-2])

        plt.legend()
        plt.grid()
        plt.savefig(str(out_dir / ('%s_b_j_s_depths_hist.png' %
                                   season)), bbox_inches='tight')
        plt.close()
        # plt.show()

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
