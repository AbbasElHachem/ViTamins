'''
@author: Faizan-Uni-Stuttgart

'''

import os
import timeit
import time
import pickle
import itertools
from pathlib import Path
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from depth_funcs import (gen_usph_vecs_mp,
                         depth_ftn_mp as depth_ftn,
                         plot_depths_hist)


plt.ioff()

np.set_printoptions(precision=3,
                    threshold=2000,
                    linewidth=200000,
                    formatter={'float': '{:0.3f}'.format})


def depth_ftn_py(x, y, ei):
    mins = x.shape[0] * np.ones((y.shape[0],))  # initial value

    for i in ei:  # iterate over unit vectors
        d = np.dot(i, x.T)  # scalar product

        dy = np.dot(i, y.T)  # scalar product

        # argsort gives the sorting indices then we used it to sort d
        # d = d[np.argsort(d)]
        d.sort()

        dy_med = np.median(dy)

        dy = ((dy - dy_med) * (1 - (1e-7))) + dy_med
        # find the index of each project y in x to preserve order
        numl = np.searchsorted(d, dy)

        # numl is number of points less then projected y
        numg = d.shape[0] - numl
        # find new min
        mins = np.min(np.vstack([mins,
                                 np.min(np.vstack([numl, numg]), axis=0)]), 0)
    return mins.astype(np.uint64)

# def depth_ftn(x, y, ei, n_cpus):
#     return depth_ftn_py(x, y, ei)


plt.ioff()
if __name__ == '__main__':
    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main_dir = Path(
        r'X:\staff\elhachem\AdvancedPython\code_data_fr_Faizan\depth_funcs_abbas')

    path_to_opt = main_dir / r'anomaly_pca'

    out_dir_str = Path(
        r'X:\staff\elhachem\AdvancedPython\code_data_fr_Faizan\depth_funcs_abbas_2021')
    cats_names_df_path = path_to_opt / r'cats_ppt_coords_19800101_20091231.csv'
    cats_areas_df_path = path_to_opt / r'cat_diff_cumm_areas.csv'
    sep = ';'

    anomaly_pickle = path_to_opt / r'ncar_anomaly.pkl'
    comp_rand_cps_path = path_to_opt / r'ncar_comp_rand_cps_all.pkl'
    cp_wettness_path = path_to_opt / r'ncar_wettness_idxs_all.pkl'

#     comp_rand_cps_path = path_to_opt / r'ncep_comp_rand_cps_all.pkl'
#     cp_wettness_path = path_to_opt / r'ncep_wettness_idxs_all.pkl'
#     anomaly_pickle = path_to_opt / r'ncep_500_anomaly.pkl'
#     anomaly_pickle = path_to_opt / r'ncep_1000_anomaly.pkl'

    out_dir_suff = 'ncar'

    n_cpus = 7
    hist_bins = np.concatenate((np.arange(4), [10000]))

    fig_size = (15, 8)

    _summer_months = np.arange(5, 11, 1)
    _winter_months = np.array([1, 2, 3, 4, 11, 12])
    b_j_idxs_list = [(0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3)]

    plot_eig = True
    plot_bjs = True
    plot_bj_dists_vols = True
    cmpt_depths = True
    cmpt_ppt_depths = True
    cmpt_dis_dif_depths = True
    cmpt_ppt_dis_df_depth = True

#     plot_eig = False
# #    plot_bjs = False
#     plot_bj_dists_vols = False
#     cmpt_depths = False
#     cmpt_ppt_depths = False
#     cmpt_dis_dif_depths = False
#    cmpt_ppt_dis_df_depth = False

    os.chdir(str(main_dir))

    cats_names_df = pd.read_csv(cats_names_df_path, sep=sep, index_col=0)
    cats_areas_df = pd.read_csv(cats_areas_df_path, sep=sep, index_col=0)

    cats_areas_df = cats_areas_df.loc[cats_names_df.index]

    with open(str(anomaly_pickle), 'rb') as _pkl_hdl:
        anomaly_obj = pickle.load(_pkl_hdl)
        vals_tot_anom = anomaly_obj.vals_tot_anom
        dates_tot = anomaly_obj.times_tot
        dates_tot = pd.DatetimeIndex(dates_tot.date)
        grid_shape = anomaly_obj.x_coords_mesh.shape
    print('vals_tot_anom shape:', vals_tot_anom.shape)

    with open(str(cp_wettness_path), 'rb') as _pkl_hdl:
        wettness_obj = pickle.load(_pkl_hdl)

        _max_area = cats_areas_df.loc[4416, 'cumm_area']
        _area_ratios = cats_areas_df.loc[:, 'diff_area'].values / _max_area
        ppt_arr = (wettness_obj.ppt_arr * _area_ratios).sum(axis=1)

    with open(str(comp_rand_cps_path), 'rb') as _pkl_hdl:
        comp_rand_cps_obj = pickle.load(_pkl_hdl)
        # 0 is 427, 1 = 4416
        dis_diff_arr = comp_rand_cps_obj.cat_ppt_arr[:, 1]

    assert (vals_tot_anom.shape[0] ==
            ppt_arr.shape[0] ==
            dis_diff_arr.shape[0])

    labs_list = ['summer', 'winter']

    months_list = [_summer_months, _winter_months]
    _empty_2d_arr = np.array([])

    # def nbr of dims, eis, ppt threshold and dis threshold
    wanted_dims = [5, 6, 7]
    wanted_eis = [1e3, 5e3]
    wanted_ppt_thr = np.percentile(ppt_arr, np.arange(97, 100, 1))
    wanted_dis_thr = np.percentile(dis_diff_arr, np.arange(97, 100, 1))

    my_list = [wanted_dims, wanted_eis, wanted_ppt_thr, wanted_dis_thr]
    my_iter = list(itertools.product(*my_list))

    print('Nbr of available combinations: ', len(my_iter))

    def plot_all_combinations(nbr_dims, nbr_eis,
                              ppt_thr, dis_thr):

        # select ppt and dis values above threshold
        _ppt_arr_thresh_idxs = (ppt_arr >= ppt_thr)
        _dis_diff_thresh_idxs = (dis_diff_arr >= dis_thr)

        assert _ppt_arr_thresh_idxs.sum() and _dis_diff_thresh_idxs.sum()

        # generate ei values
        ei = gen_usph_vecs_mp(nbr_eis, nbr_dims, n_cpus)

        for lab_i, season in enumerate(labs_list):
            print('\n%s' % season)
            idxs = np.zeros(vals_tot_anom.shape[0], dtype=bool)
            for month in months_list[lab_i]:
                idxs = idxs | (dates_tot.month == month)

            curr_ppt_thresh_idxs = _ppt_arr_thresh_idxs[idxs]
            curr_dis_diff_thresh_idxs = _dis_diff_thresh_idxs[idxs]

            anom = vals_tot_anom[idxs, :]
            anom.shape
            corr_mat = np.corrcoef(anom.T)
            eig_val, eig_mat = np.linalg.eig(corr_mat)
            sort_idxs = np.argsort(eig_val)[::-1]
            eig_val = eig_val[sort_idxs]
            eig_mat = eig_mat[:, sort_idxs]
            # eig_val_sum = eig_val.sum()
            # eig_val_cum_sum = np.cumsum(eig_val) / eig_val_sum
            # print('sum of eigen valus:', eig_val_cum_sum)

            # second plot b_j_s
            b_j_s = np.dot(anom, eig_mat.T)

            mid_idx = b_j_s.shape[0] // 2
            b_j_s_left = b_j_s[:mid_idx, :]
            b_j_s_right = b_j_s[mid_idx:, :]

            curr_ppt_thresh_idxs_left = curr_ppt_thresh_idxs[:mid_idx]
            curr_ppt_thresh_idxs_right = curr_ppt_thresh_idxs[mid_idx:]

            curr_dis_diff_thresh_idxs_left = \
                curr_dis_diff_thresh_idxs[:mid_idx]
            curr_dis_diff_thresh_idxs_right = \
                curr_dis_diff_thresh_idxs[mid_idx:]

            # depths
            xs_left = b_j_s_left[:, :nbr_dims]
            xs_right = b_j_s_right[:, :nbr_dims]
            print('xs_left and xs_right sizes:', xs_left.shape, xs_right.shape)

            ys_left = b_j_s_left[:, :nbr_dims][curr_ppt_thresh_idxs_left, :]
            ys_right = b_j_s_right[:, :nbr_dims][curr_ppt_thresh_idxs_right, :]
            print('ys_left and ys_right sizes:', ys_left.shape, ys_right.shape)

            zs_left = b_j_s_left[:, :nbr_dims][
                curr_dis_diff_thresh_idxs_left, :]
            zs_right = b_j_s_right[:, :nbr_dims][
                curr_dis_diff_thresh_idxs_right, :]
            print('zs_left and zs_right sizes:', zs_left.shape, zs_right.shape)

            if plot_eig:
                print('Plotting Eigen vectors...')
                _vmin = eig_mat[:, :nbr_dims].min()
                _vmax = eig_mat[:, :nbr_dims].max()
                plt.ioff()
                plt.figure(figsize=fig_size)
                for i in range(nbr_dims):
                    plt.imshow(eig_mat[:, i].reshape(grid_shape),
                               origin='upper',
                               vmin=_vmin,
                               vmax=_vmax)
                    plt.colorbar(label='eig_vec')
                    plt.title('%s Eigen vector: %d' % (season, i))
#                     out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                             None, None, nbr_eis)

                    out_dir = out_dir_str
                    if not out_dir.exists():
                        out_dir.mkdir()

                    plt.savefig(str(out_dir / (r'%s_eig_vec_%d.png'
                                               % (season, i))),
                                bbox_inches='tight')
                    plt.clf()
                    # plt.show()
                plt.close()

            if plot_bj_dists_vols:
                print('Plotting distribution of b_js...')
                # left n-D volumes
                b_j_s_left_mins = b_j_s_left.min(axis=0)[:nbr_dims]

                b_j_s_left_maxs = b_j_s_left.max(axis=0)[:nbr_dims]

                b_j_s_left_con_hull = ConvexHull(b_j_s_left[:, :nbr_dims])
                b_j_s_left_thresh_con_hull = (
                    ConvexHull(b_j_s_left[curr_ppt_thresh_idxs_left,
                                          :nbr_dims]))

                _arr = [np.array([b_j_s_left_maxs[_], b_j_s_left_mins[_]])
                        for _ in range(nbr_dims)]
                _arr = np.array(list(product(*_arr)))
                b_j_s_left_rect_con_hull = ConvexHull(_arr)

                b_j_s_left_con_hull_vol = b_j_s_left_con_hull.volume
                b_j_s_left_thresh_con_hull_vol = \
                    b_j_s_left_thresh_con_hull.volume
                b_j_s_left_rect_con_hull_vol = b_j_s_left_rect_con_hull.volume

                assert b_j_s_left_rect_con_hull_vol >= b_j_s_left_con_hull_vol
                assert (b_j_s_left_rect_con_hull_vol >=
                        b_j_s_left_thresh_con_hull_vol)

                # right n-D volumes
                b_j_s_right_mins = b_j_s_right.min(axis=0)[:nbr_dims]

                b_j_s_right_maxs = b_j_s_right.max(axis=0)[:nbr_dims]

                b_j_s_right_con_hull = ConvexHull(b_j_s_right[:, :nbr_dims])
                b_j_s_right_thresh_con_hull = (
                    ConvexHull(b_j_s_right[curr_ppt_thresh_idxs_right,
                                           :nbr_dims]))

                _arr = [np.array([b_j_s_right_mins[_], b_j_s_right_maxs[_]])
                        for _ in range(nbr_dims)]
                _arr = np.array(list(product(*_arr)))
                b_j_s_right_rect_con_hull = ConvexHull(_arr)

                b_j_s_right_con_hull_vol = b_j_s_right_con_hull.volume

                b_j_s_right_thresh_con_hull_vol = \
                    b_j_s_right_thresh_con_hull.volume

                b_j_s_right_rect_con_hull_vol = \
                    b_j_s_right_rect_con_hull.volume

                assert (b_j_s_right_rect_con_hull_vol >=
                        b_j_s_right_con_hull_vol)
                assert (b_j_s_right_rect_con_hull_vol >=
                        b_j_s_right_thresh_con_hull_vol)

                b_j_s_left_sq = b_j_s_left ** 2
                b_j_s_left_sq_main_sum = np.sort(
                    b_j_s_left_sq[:, :nbr_dims].sum(axis=1))
                b_j_s_left_sq_rema_sum = np.sort(
                    b_j_s_left_sq[:, nbr_dims:].sum(axis=1))

                b_j_s_left_sq_main_thresh_sum = (
                    np.sort(b_j_s_left_sq[curr_ppt_thresh_idxs_left,
                                          :nbr_dims].sum(axis=1)))
                b_j_s_left_sq_rema_thresh_sum = (
                    np.sort(b_j_s_left_sq[curr_ppt_thresh_idxs_left,
                                          nbr_dims:].sum(axis=1)))

                b_j_s_right_sq = b_j_s_right ** 2
                b_j_s_right_sq_main_sum = np.sort(
                    b_j_s_right_sq[:, :nbr_dims].sum(axis=1))
                b_j_s_right_sq_rema_sum = np.sort(
                    b_j_s_right_sq[:, nbr_dims:].sum(axis=1))

                b_j_s_right_sq_main_thresh_sum = (
                    np.sort(b_j_s_right_sq[curr_ppt_thresh_idxs_right,
                                           :nbr_dims].sum(axis=1)))
                b_j_s_right_sq_rema_thresh_sum = (
                    np.sort(b_j_s_right_sq[curr_ppt_thresh_idxs_right,
                                           nbr_dims:].sum(axis=1)))

                probs_left = (np.arange(b_j_s_left_sq_main_sum.shape[0]) /
                              (b_j_s_left_sq_main_sum.shape[0] + 1))
                probs_right = (np.arange(b_j_s_right_sq_main_sum.shape[0]) /
                               (b_j_s_right_sq_main_sum.shape[0] + 1))

                probs_left_thresh = (np.arange(
                    curr_ppt_thresh_idxs_left.sum()) /
                    (curr_ppt_thresh_idxs_left.sum() + 1))
                probs_right_thresh = (np.arange(
                    curr_ppt_thresh_idxs_right.sum()) /
                    (curr_ppt_thresh_idxs_right.sum() + 1))
                plt.ioff()
                plt.figure(figsize=(fig_size))
                plt.plot(b_j_s_left_sq_main_sum,
                         probs_left,
                         alpha=0.4,
                         label='b_j_s_left_sq_main_sum')

                plt.plot(b_j_s_left_sq_rema_sum,
                         probs_left,
                         alpha=0.4,
                         label='b_j_s_left_sq_rema_sum')

                plt.plot(b_j_s_left_sq_main_thresh_sum,
                         probs_left_thresh,
                         alpha=0.4,
                         label='b_j_s_left_sq_main_thresh_sum')

                plt.plot(b_j_s_left_sq_rema_thresh_sum,
                         probs_left_thresh,
                         alpha=0.4,
                         label='b_j_s_left_sq_rema_thresh_sum')

                plt.plot(b_j_s_right_sq_main_sum,
                         probs_right,
                         alpha=0.4,
                         label='b_j_s_right_sq_main_sum')

                plt.plot(b_j_s_right_sq_rema_sum,
                         probs_right,
                         alpha=0.4,
                         label='b_j_s_right_sq_rema_sum')

                plt.plot(b_j_s_right_sq_main_thresh_sum,
                         probs_right_thresh,
                         alpha=0.4,
                         label='b_j_s_right_sq_main_thresh_sum')

                plt.plot(b_j_s_right_sq_rema_thresh_sum,
                         probs_right_thresh,
                         alpha=0.4,
                         label='b_j_s_right_sq_rema_thresh_sum')

                dist_title = ''
                dist_title += (('%s b_j_s distributions, ndims: %d\nHigh '
                                'precipitation threshold: %0.1f') %
                               (season, nbr_dims, ppt_thr))
                dist_title += (('\n%dD left rectangle, b_j_s (n=%d), '
                                'b_j_s_ge (n=%d) volumes: ') %
                               (nbr_dims,
                                b_j_s_left.shape[0],
                                curr_ppt_thresh_idxs_left.sum()))
                dist_title += ('%0.2f, %0.2f (%0.3f), %0.2f (%0.3f)' %
                               (b_j_s_left_rect_con_hull_vol,
                                b_j_s_left_con_hull_vol,
                                (100 * (b_j_s_left_con_hull_vol /
                                        b_j_s_left_rect_con_hull_vol)),
                                b_j_s_left_thresh_con_hull_vol,
                                (100 * (b_j_s_left_thresh_con_hull_vol /
                                        b_j_s_left_rect_con_hull_vol)),
                                ))

                dist_title += (('\n%dD right rectangle, b_j_s (n=%d), '
                                'b_j_s_ge (n=%d) volumes: ') %
                               (nbr_dims,
                                b_j_s_right.shape[0],
                                curr_ppt_thresh_idxs_right.sum()))
                dist_title += ('%0.2f, %0.2f (%0.3f), %0.2f (%0.3f)' %
                               (b_j_s_right_rect_con_hull_vol,
                                b_j_s_right_con_hull_vol,
                                (100 * (b_j_s_right_con_hull_vol /
                                        b_j_s_right_rect_con_hull_vol)),
                                b_j_s_right_thresh_con_hull_vol,
                                (100 * (b_j_s_right_thresh_con_hull_vol /
                                        b_j_s_right_rect_con_hull_vol)),
                                ))

                plt.title(dist_title)
                plt.grid()
                plt.legend()
                # plt.show()
#                 out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                         ppt_thr, None, nbr_eis)

                out_dir = out_dir_str

                if not out_dir.exists():
                    out_dir.mkdir()

                plt.savefig(str(out_dir / ('%s_distributions_b_j_s_.png' %
                                           (season))),
                            bbox_inches='tight')
                plt.close()

    #         raise Exception

            if plot_bjs:
                print('Plotting bjs...')
                plt.figure(figsize=(fig_size))
                for b_j_idx in b_j_idxs_list:
                    plt.scatter(b_j_s_left[:, b_j_idx[0]],
                                b_j_s_left[:, b_j_idx[1]],
                                alpha=0.15,
                                label='left')
                    plt.scatter(b_j_s_right[:, b_j_idx[0]],
                                b_j_s_right[:, b_j_idx[1]],
                                alpha=0.15,
                                label='right')

                    plt.scatter(b_j_s_left[:, b_j_idx[0]]
                                [curr_ppt_thresh_idxs_left],
                                b_j_s_left[:, b_j_idx[1]]  # was missing comma
                                [curr_ppt_thresh_idxs_left],
                                alpha=0.75,
                                label='ge_ppt_left')
                    plt.scatter(b_j_s_right[:, b_j_idx[0]]
                                [curr_ppt_thresh_idxs_right],
                                b_j_s_right[:, b_j_idx[1]]
                                [curr_ppt_thresh_idxs_right],
                                alpha=0.75,
                                label='ge_ppt_right')

                    plt.scatter(b_j_s_left[:, b_j_idx[0]]
                                [curr_dis_diff_thresh_idxs_left],
                                b_j_s_left[:, b_j_idx[1]]
                                [curr_dis_diff_thresh_idxs_left],
                                alpha=0.75,
                                label='ge_dis_diff_left')

                    plt.scatter(b_j_s_right[:, b_j_idx[0]]
                                [curr_dis_diff_thresh_idxs_right],
                                b_j_s_right[:, b_j_idx[1]]
                                [curr_dis_diff_thresh_idxs_right],
                                alpha=0.75,
                                label='ge_dis_diff_right')

                    plt.title((('%s b_j_s (n_left=%d, n_right=%d): %d, %d, '
                                'ndims: %d\nHigh precipitation'
                                'threshold: %0.1f')
                               % (season,
                                  b_j_s_left.shape[0],
                                  b_j_s_right.shape[0],
                                  *b_j_idx,
                                  nbr_dims,
                                  ppt_thr)))
                    plt.grid()
                    plt.legend()
                    # plt.show()

#                     out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                             ppt_thr, dis_thr, nbr_eis)
                    out_dir = path_to_opt / out_dir_str

                    if not out_dir.exists():
                        out_dir.mkdir()

                    plt.savefig(str(out_dir / ('%s_b_j_s_%d_%d.png' %
                                               (season, *b_j_idx))),
                                bbox_inches='tight')
                    plt.clf()
                plt.close()

            if cmpt_depths:
                print('Plotting left and right depths...')

#                 out_dir_str = genOutDir(out_dir_suff, nbr_dims, None, None,
#                                         nbr_eis)
                out_dir = out_dir_str
                if not out_dir.exists():
                    out_dir.mkdir()

                out_loc = str(out_dir / ('%s_b_j_s_depths_hist_.png'
                                         % (season)))
                _labs = ['xs_left', '', 'xs_right', '']

                plot_depths_hist(xs_left,
                                 _empty_2d_arr,
                                 xs_right,
                                 _empty_2d_arr,
                                 ei,
                                 '',
                                 out_loc,
                                 n_cpus,
                                 fig_size,
                                 labs=_labs)
            if cmpt_ppt_depths:
                print('Plotting left and right depths with ppt thresh events:')

#                 out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                         ppt_thr, None, nbr_eis)
                out_dir = out_dir_str

                if not out_dir.exists():
                    out_dir.mkdir()

                out_loc = str(out_dir / ('%s_b_j_s_depths_ppt_hist_.png'
                                         % (season)))
                _labs = ['xs_left', 'xs_right', 'ys_left', 'ys_right']

                plot_depths_hist(xs_left,
                                 xs_right,
                                 ys_left,
                                 ys_right,
                                 ei,
                                 '',
                                 out_loc,
                                 n_cpus,
                                 fig_size,
                                 labs=_labs)

            if cmpt_dis_dif_depths:
                print(('Plotting left and right depths',
                       ' with dis_diff thresh events...'))
#
#                 out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                         None, dis_thr, nbr_eis)
                out_dir = out_dir_str
                if not out_dir.exists():
                    out_dir.mkdir()

                out_loc = str(out_dir / ('%s_b_j_s_depths_dis_diff_hist_.png'
                                         % (season)))
                _labs = ['zs_left', '', 'zs_right', '']

                plot_depths_hist(zs_left,
                                 _empty_2d_arr,
                                 zs_right,
                                 _empty_2d_arr,
                                 ei,
                                 '',
                                 out_loc,
                                 n_cpus,
                                 fig_size,
                                 labs=_labs)

            if cmpt_ppt_dis_df_depth:
                print('Plotting left and right depths with ppt and dis_diff '
                      'thresh events...')

#                 out_dir_str = genOutDir(out_dir_suff, nbr_dims,
#                                         ppt_thr, dis_thr, nbr_eis)
                out_dir = out_dir_str

                if not out_dir.exists():
                    out_dir.mkdir()

                out_loc = str(out_dir / (('%s_b_j_s_depths_hist'
                                          '_ppt_dis_diff_.png')
                                         % (season)))
                _labs = ['ys_left', 'ys_right', 'zs_left', 'zs_right']

                plot_depths_hist(ys_left,
                                 ys_right,
                                 zs_left,
                                 zs_right,
                                 ei,
                                 '',
                                 out_loc,
                                 n_cpus,
                                 fig_size,
                                 labs=_labs)
            all_vars = globals()
            all_keys = list(globals().keys())
            out_dict = {}
            for _key in all_keys:
                _val = all_vars[_key]
                if isinstance(_val, np.ndarray):
                    out_dict[_key] = _val

            if out_dict:
                with open(str(out_dir /
                              ('%s_vars.pkl' % season)), 'wb') as _pkl_hdl:
                    pickle.dump(out_dict, _pkl_hdl)

    for combination in my_iter:
        _start = timeit.default_timer()  # Ending time

        plot_all_combinations(combination[0], combination[1],
                              combination[2], combination[3])
        _stop = timeit.default_timer()  # Ending time
        print(('\n\a\a\a Done with Combination: %s.\nTotal run time was'
               ' about %0.4f seconds \a\a\a' % (str(combination),
                                                _stop - _start)))

STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s.\nTotal run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
