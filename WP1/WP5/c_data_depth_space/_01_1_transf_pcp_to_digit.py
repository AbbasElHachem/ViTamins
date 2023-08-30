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
        # r"\dwd_comb_5min_data_agg_5min_2020_flagged_%s.h5"
        r"\%s_dwd_stns_1440min_1880_2019.h5"
        % radar_loc)

    n_vecs = int(1e6)
    n_cpus = 3
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

    # create a tree from DWD coordinates

    dwd_coords_xy = [(x, y) for x, y in zip(
        in_dwd_df_coords_utm32.loc[:, 'X'].values,
        in_dwd_df_coords_utm32.loc[:, 'Y'].values)]

    # create a tree from coordinates
    dwd_points_tree = cKDTree(dwd_coords_xy)

    for _dwd_id in dwd_ids[5:10]:

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

        dwd_pcp = dwd_hdf5.get_pandas_dataframe(
            _dwd_id).dropna(how='all')

        dwd_pcp_daily = resampleDf(dwd_pcp, '1440min')

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
            stn_near, event_start=dwd_pcp_daily.index[0],
            event_end=dwd_pcp_daily.index[-1]).dropna(how='all', axis=1)

        # df_stn_near.index.intersection()
        # dwd_pcp_daily.loc['1941-01-13':'1941-01-19']
        df_transf = tranform_num_to_digit(df=dwd_pcp_daily,
                                          pcp_thr=thr,
                                          nbr_days=6)

        plt.ioff()
        fig = plt.figure(figsize=(12, 8), dpi=150)
        plt.plot(dwd_pcp_daily.index,
                 dwd_pcp_daily.values.ravel())
        plt.grid()
        plt.ylabel('mm/d')
        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir,
                                 'cpcp_stn_%s.png' % _dwd_id))
        plt.close()
        # plt.show()
        plt.ioff()
        fig = plt.figure(figsize=(12, 8), dpi=150)
        plt.plot(df_transf.index,
                 df_transf.values.ravel())

        plt.grid()
        plt.tight_layout()

        plt.savefig(os.path.join(out_save_dir,
                                 'tranf_pcp_stn_%s.png' % _dwd_id))

        plt.close()

        dwd_pcp_daily_res = resampleDf(dwd_pcp_daily, '6D')
        dwd_pcp_daily_res.index = range(len(
            dwd_pcp_daily_res.index))
        # dwd_pcp_daily_res.iloc[range(len(
        # df_transf.index))] = df_transf.values
        plt.ioff()
        fig = plt.figure(figsize=(12, 8), dpi=150)
        plt.scatter(dwd_pcp_daily_res.iloc[range(len(
            df_transf.index))].values.ravel(),
            df_transf.values.ravel())

        plt.grid()
        plt.tight_layout()

        plt.savefig(os.path.join(out_save_dir,
                                 'scatter_tranf_pcp_stn_%s.png' % _dwd_id))

        plt.close()

        # df_transf_near_all = pd.DataFrame(index=dwd_pcp_daily.index,
        #                                   columns=df_stn_near.columns)

        n_dims = df_transf.shape[1]

        usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

        for _stn_near in df_stn_near.columns:

            one_stn_near = df_stn_near.loc[:, _stn_near].dropna(how='all')
            # break
            ont_stn_near_int = one_stn_near.loc[
                dwd_pcp_daily.index.intersection(
                    one_stn_near.index)]
            if one_stn_near.values.size > 0:
                df_transf_near = tranform_num_to_digit(df=ont_stn_near_int,
                                                       pcp_thr=thr,
                                                       nbr_days=6)

                df_transf_near = df_transf_near.loc[
                    df_transf_near.index.intersection(df_transf.index)]

                df_transf_orig = df_transf.loc[
                    df_transf.index.intersection(df_transf_near.index)]

                if df_transf_orig.values.size > 0:
                    # plt.ioff()
                    # fig = plt.figure(figsize=(12, 8), dpi=150)
                    # plt.plot(ont_stn_near_int.index,
                    #          ont_stn_near_int.values.ravel())
                    # plt.grid()
                    # plt.ylabel('mm/d')
                    # plt.tight_layout()
                    # plt.savefig(os.path.join(out_save_dir,
                    #                          'cpcp_stn_%s.png' % _stn_near))
                    # plt.close()
                    # # plt.show()
                    # plt.ioff()
                    # fig = plt.figure(figsize=(12, 8), dpi=150)
                    # plt.plot(df_transf_near.index,
                    #          df_transf_near.values.ravel())
                    #
                    # plt.grid()
                    # plt.tight_layout()
                    #
                    # plt.savefig(os.path.join(out_save_dir,
                    #                          'tranf_pcp_stn_%s.png' % _stn_near))
                    #
                    # plt.close()

                    depths1 = depth_ftn_mp(df_transf_orig.values.copy(order='c').astype(float),
                                           df_transf_near.values.copy(
                                               order='c').astype(float),
                                           usph_vecs, n_cpus, 1)

                    depths2_1 = depth_ftn_mp(df_transf_near.values.copy(order='c').astype(float),
                                             df_transf_orig.values.copy(
                        order='c').astype(float),
                        usph_vecs, n_cpus, 1)

                    ids_low_de_1_2 = np.where(depths1 <= 10)[0]
                    ids_low_de_2_1 = np.where(depths2_1 <= 10)[0]
                    # low_d1_1_2_ = depths1[depths1 <= 10]
                    # PiYG_r
                    plt.ioff()
                    fig = plt.figure(figsize=(12, 8), dpi=100)
                    im0 = plt.scatter(df_transf_orig,
                                      df_transf_near, c=depths1,
                                      cmap=plt.get_cmap('spring'),
                                      vmin=10,
                                      s=10)

                    plt.scatter(df_transf_near,
                                df_transf_orig, c=depths2_1,
                                cmap=plt.get_cmap('spring'),
                                vmin=10,
                                s=10)

                    plt.scatter(
                        df_transf_orig.iloc[ids_low_de_1_2],
                        df_transf_near.iloc[ids_low_de_1_2],
                        c=depths1[ids_low_de_1_2],
                        cmap=plt.get_cmap('inferno_r'),
                        marker='X', s=50,
                        label='Depth 1 in 2')
                    im1 = plt.scatter(
                        df_transf_orig.iloc[ids_low_de_2_1],
                        df_transf_near.iloc[ids_low_de_2_1],
                        s=50,
                        c=depths2_1[ids_low_de_2_1],
                        cmap=plt.get_cmap('inferno_r'),
                        marker='d',
                        label='Depth 2 in 1')
                    fig.colorbar(im0, shrink=0.8, label='10<Depth')
                    fig.colorbar(im1, shrink=0.8, label='Depth<=10')
                    plt.xlabel('%s' % _dwd_id)
                    plt.ylabel('%s' % _stn_near)
                    plt.grid()
                    plt.legend(loc='lower left')
                    plt.tight_layout()
                    # plt.show()
                    plt.savefig(os.path.join(out_save_dir,
                                             'depth_pcp_stn_%s_%s.png'
                                             % (_dwd_id, _stn_near)))

                    plt.close()
                    if plot_timesreies_events:
                        for i, (_idx, dd) in enumerate(
                                zip(
                                    df_transf_near.iloc[ids_low_de_2_1].index,
                                    depths2_1[ids_low_de_2_1])):

                            evet_Start = _idx - pd.Timedelta(days=5)

                            dwd_center = dwd_pcp_daily.loc[evet_Start:_idx]
                            dwd_near = ont_stn_near_int.loc[evet_Start:_idx]

                            # break
                            df_stn_near_all = df_stn_near.loc[
                                evet_Start:_idx, :].dropna(how='all', axis=1)
                            plt.ioff()
                            fig = plt.figure(figsize=(12, 8), dpi=150)

                            for _d in df_stn_near_all.columns:
                                plt.plot(df_stn_near_all.index,
                                         df_stn_near_all.loc[:,
                                                             _d].values.ravel(),
                                         c='k', alpha=.5)
                            plt.plot(df_stn_near_all.index,
                                     df_stn_near_all.loc[:, _d].values.ravel(),
                                     label='Neighbors',
                                     c='k', alpha=.5)
                            plt.plot(dwd_center.index,
                                     dwd_center.values.ravel(),
                                     label='Center station',
                                     c='r')
                            plt.plot(dwd_near.index,
                                     dwd_near.values.ravel(),
                                     label='Neighboring station',
                                     c='b')

                            plt.title('%s - Depth value d=%d' % (_idx, dd))
                            plt.grid()
                            plt.ylabel('mm/d')
                            plt.legend(loc=0)
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(out_save_dir,
                                             'cpcp_stn_%d_2_1_%s_%s.png' %
                                             (i, _stn_near, _dwd_id, )))
                            plt.close()

                            pass

                        for i, (_idx, dd) in enumerate(
                                zip(
                                    df_transf_orig.iloc[ids_low_de_1_2].index,
                                    depths1[ids_low_de_1_2])):

                            evet_Start = _idx - pd.Timedelta(days=5)

                            dwd_center = dwd_pcp_daily.loc[evet_Start:_idx]
                            dwd_near = ont_stn_near_int.loc[evet_Start:_idx]
                            print(dwd_center)
                            df_stn_near_all = df_stn_near.loc[
                                evet_Start:_idx, :].dropna(how='all', axis=1)
                            plt.ioff()
                            fig = plt.figure(figsize=(12, 8), dpi=150)

                            for _d in df_stn_near_all.columns:
                                plt.plot(df_stn_near_all.index,
                                         df_stn_near_all.loc[:,
                                                             _d].values.ravel(),
                                         c='k', alpha=.5)
                            plt.plot(df_stn_near_all.index,
                                     df_stn_near_all.loc[:, _d].values.ravel(),
                                     label='Neighbors',
                                     c='k', alpha=.5)
                            plt.plot(dwd_center.index,
                                     dwd_center.values.ravel(),
                                     label='Center station',
                                     c='r')
                            plt.plot(dwd_near.index,
                                     dwd_near.values.ravel(),
                                     label='Neighboring station',
                                     c='b')

                            plt.title('%s - Depth value d=%d' % (_idx, dd))
                            plt.grid()
                            plt.ylabel('mm/d')
                            plt.legend(loc=0)
                            plt.tight_layout()
                            plt.savefig(
                                os.path.join(out_save_dir,
                                             'cpcp_stn_%d_1_2_%s_%s.png'
                                             % (i, _dwd_id, _stn_near)))
                            plt.close()

                        pass
        #     df_transf_near_all.loc[
        #         df_transf_near.index, _stn_near] = df_transf_near.values.ravel()
        #
        # df_transf_near_all_nonan = df_transf_near_all.dropna(how='all')

    # res_test_df = cnvt_ser_to_mult_dims_df(in_test_ser, n_dims)
    # tot_test_var_arr = res_test_df.values.copy('c')
    # n_dims = tot_test_var_arr.shape[1]
    #
    # usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)
    #
    # depths1 = depth_ftn_mp(
    #     tot_test_var_arr,
    #     tot_test_var_arr,
    #     usph_vecs, n_cpus, 1)

    print('done')
    # dwd_pcp_daily.plot()
    # df_transf.plot()
    #
    # df_depth_vals = pd.DataFrame(index=df_transf.index,
    #                              data=df_transf.values,
    #                              columns=['Stn'])
#
#     df_depth_vals['Depth'] = depths1
#     df_transf.hist()
#
#     # plt.show()
#     df_depth_vals.plot()
#

#
#     dwd_pcp = dwd_hdf5.get_pandas_dataframe(
#         dwd_ids[:20])
#
#     pcp_data = resampleDf(dwd_pcp, '1440min')
#
#     pcp_data.fillna(0, inplace=True)
#     pca = PCA(n_components=0.9, svd_solver='full')
#     X = pcp_data  # .iloc[0,:]
#     X_r = pca.fit(X).transform(X)
#
#     from scipy.stats import pearsonr
#
#     pearsonr(pcp_data.iloc[:, 0], pcp_data.iloc[:, 1])
#     #corr_mtx = np.empty()
#     X_r.shape
#
#     # pca.explained_variance_ratio_
#     print("Components = ", pca.n_components_, ";\nTotal explained variance = ",
#           round(pca.explained_variance_ratio_.sum(), 5))
#
#     print(pca.explained_variance_ratio_)
#     print(pca.singular_values_)
# # df_transf.hist()


def tranform_num_to_digit(df, pcp_thr, nbr_days):
    df_binary = df.copy()
    df_binary[df_binary < pcp_thr] = 0
    df_binary[df_binary >= pcp_thr] = 1

    transf_vals = []
    index_vals_all = []
    for i in range(nbr_days,
                   df_binary.shape[0],
                   nbr_days):
        # print(i)
        index_vals = df_binary.index[i - nbr_days:i][-1]
        vals_in_window = df_binary.values[i - nbr_days:i].ravel().astype(int)
        num = str(''.join(map(str, vals_in_window)))

        # num = '000001'
        decimal = []
        for i, digit in enumerate(num):

            decimal.append(2 ** (i) * int(digit))

            # print(decimal)
            # decimal = decimal * 2 + int(digit)

        # print(decimal)
        decimal_all = np.sum(decimal)
        transf_vals.append(decimal_all)
        index_vals_all.append(index_vals)
        # break
    transf_vals_arr = np.array(transf_vals)
    # date_range = pd.date_range(start=df_binary.index[0],
    #                            end=df_binary.index[-1],
    #                            freq='%dD' % nbr_days)
    date_range = pd.DatetimeIndex(index_vals_all)
    df_ne = pd.DataFrame(data=transf_vals_arr,
                         index=date_range[:transf_vals_arr.shape[0]])

    # df_ne.loc['1964-02']
    # df_binary.loc['1964-02']
    # df.loc['1964-02']
    return df_ne


# def test_pca(pcp_data, svd_solver='auto', n_components):
#     X = pcp_data
#     pca = PCA(n_components=n_components, svd_solver=svd_solver)

#     # check the shape of fit.transform
#     X_r = pca.fit(X).transform(X)
#     assert X_r.shape[1] == n_components

#     # check the equivalence of fit.transform and fit_transform
#     X_r2 = pca.fit_transform(X)
#     assert_allclose(X_r, X_r2)
#     X_r = pca.transform(X)
#     assert_allclose(X_r, X_r2)

#     # Test get_covariance and get_precision
#     cov = pca.get_covariance()
#     precision = pca.get_precision()
#     assert_allclose(np.dot(cov, precision), np.eye(X.shape[1]), atol=1e-12)


# dwd_pcp_daily_boolean = dwd_pcp_daily.copy()
# dwd_pcp_daily_boolean[dwd_pcp_daily_boolean < thr] = 0
# dwd_pcp_daily_boolean[dwd_pcp_daily_boolean >= thr] = 1


# new_vals = []
# nbr_steps = round(dwd_pcp_daily_boolean.shape[0]/6)
# for i in range(6, dwd_pcp_daily_boolean.shape[0]+1, 6):
#     print(i)
#     dwd_pcp_daily_boolean_test = dwd_pcp_daily_boolean.values[i-6:i].ravel().astype(int)
#     num = str(''.join(map(str,dwd_pcp_daily_boolean_test)))

#     decimal = 0
#     for digit in num:
#         decimal = decimal*2 + int(digit)

#     print(decimal)

#     new_vals.append(decimal)
#     #break
# new_vals_arr = np.array(new_vals)
# df_ne = pd.DataFrame(data=new_vals_arr,
#                      index=pd.date_range(start=dwd_pcp_daily_boolean.index[0],
#                                          end=dwd_pcp_daily_boolean.index[-1],
#                                          freq='6D'))
# df_ne.hist()
# df_ne.plot()
# #print(round(dwd_pcp_daily_boolean.shape[0]/6),0)


# Functionally:


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
