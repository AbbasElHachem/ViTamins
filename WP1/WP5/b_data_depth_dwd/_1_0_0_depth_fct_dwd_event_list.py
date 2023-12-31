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
import tqdm
import imageio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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

    out_save_dir = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis"
        r"\10_depth_function\03_events_depth")

    # In[2]:
    path_to_dwd_hdf5 = (
        r"X:\staff\elhachem\ClimXtreme\03_data\00_DWD"
        r"\dwd_comb_5min_data_agg_5min_2020_flagged_Hannover.h5"
        # r"\dwd_comb_1440min_gk3.h5"
    )

    beg_time = '1995-01-01 00:00:00'
    end_time = '2019-12-31 23:55:00'
    freq = '60min'

    path_to_intense_evts = (
        r"X:\staff\elhachem\ClimXtreme\04_analysis\00_events_select"
        r"\select_events_thr_based"
        r"\rem_timesteps_60min_Hannover_thr_6.00_.csv"
        # r"\rem_timesteps_1440min_Hannover_thr_50.00_.csv"
    )
    data_df_nonan = pd.read_csv(
        os.path.join(out_save_dir,
                     'hourly_data_1995_2019_complete_hannover.csv'),
        sep=';',
        index_col=0, engine='c',
        parse_dates=True,
        infer_datetime_format=True)

    time_fmt = '%Y-%m-%d %H:%M:%S'
    sep = ';'

    dwd_hdf5 = HDF5(infile=path_to_dwd_hdf5)
    dwd_ids = dwd_hdf5.get_all_names()
    df_coords = dwd_hdf5.get_coordinates(dwd_ids)
    in_df_coords_utm32 = pd.DataFrame(
        index=dwd_ids,
        data=df_coords['easting'], columns=['X'])
    y_dwd_coords = df_coords['northing']
    in_df_coords_utm32.loc[:, 'Y'] = y_dwd_coords
    in_df_coords_utm32.drop_duplicates(keep='first', inplace=True)
    in_df_coords_utm32.loc[:, 'lon'] = df_coords['lon']
    in_df_coords_utm32.loc[:, 'lat'] = df_coords['lat']

    xmin = min(in_df_coords_utm32.X.values) - 1000
    xmax = max(in_df_coords_utm32.X.values) + 1000

    ymin = min(in_df_coords_utm32.Y.values) - 1000
    ymax = max(in_df_coords_utm32.Y.values) + 1000
    # dwd_ids = dwd_ids[:1000]

    df_intense_events = pd.read_csv(
        path_to_intense_evts, sep=sep, index_col=0,
        engine='c')
    df_intense_events.index = pd.to_datetime(
        df_intense_events.index, format=time_fmt)

    dates_many_stns = df_intense_events.iloc[
        np.where(df_intense_events.sum(axis=1) > 10)[0], :].index

    bound_ppt = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    interval_ppt = np.linspace(0.1, 0.95)
    colors_ppt = plt.get_cmap('YlGnBu')(interval_ppt)
    cmap_ppt = LinearSegmentedColormap.from_list('name', colors_ppt)
    cmap_ppt.set_over('m')
    cmap_ppt.set_under('lightgrey')  # lightgrey
    norm_ppt = mcolors.BoundaryNorm(bound_ppt, cmap_ppt.N)


#     in_ref_files = [
#         r'stn1.csv',
#         r'stn2.csv']
    # df_intense_events.iloc[:, 1].dropna()
    # df = dwd_hdf5.get_pandas_dataframe(dwd_ids[1])

    # df.loc[beg_time:end_time, :].plot()

    # df.loc['2014-07-10']
#     in_tst_files = [
#         r'stn1_max.csv',
#         r'stn2_max.csv']

    n_vecs = int(1e6)
    n_cpus = 2

    # data_df = pd.DataFrame(
    #     index=pd.date_range(beg_time, end_time, freq=freq),
    #     dtype=float)
    # #
    # for _stn in tqdm.tqdm(dwd_ids):
    #     df = dwd_hdf5.get_pandas_dataframe(_stn)
    #     df_resampled = resampleDf(df, freq)
    #
    #     df = df_resampled.reindex(data_df.index)  # .dropna(how='all')
    #
    #     data_df.loc[:, _stn] = df.values[:, 0]
    # print('done getting data')
    # data_df = data_df.loc[beg_time: end_time, :]
    # # data_df = data_df.dropna(how='all', axis=1)
    # data_df_nonan = data_df.fillna(0.)

    sum_events = data_df_nonan.loc[dates_many_stns, :].sum(axis=1)
    # data_df_nonan.loc['2019-10-16 01:00:00',:].plot()
    # plt.title('2019-10-16 01:00:00')

    # plt.show()
    # data_df_nonan.to_csv(
    #     os.path.join(out_save_dir,
    #                  'hourly_data_1995_2019_complete_hannover.csv'),
    #     sep=';')

    # data_df_nonan.sum(axis=1).sort_values()

    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)
    for _evt in dates_many_stns:

        plt.plot(range(len(data_df_nonan.columns)),
                 data_df_nonan.loc[_evt, :],
                 alpha=0.75)

    plt.xlabel('Stn ID', fontsize=14)
    plt.ylabel('Pcp [mm/%s]' % freq, fontsize=14)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, 'events_%s.png' % freq))
    plt.close()

    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)

    plt.plot(sum_events.index, sum_events.values, c='b')
    plt.scatter(sum_events.index, sum_events.values, marker='o', c='b')
    plt.xlabel('Event ID', fontsize=14)
    plt.ylabel('Pcp [mm/%s]' % freq, fontsize=14)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, 'sum_events_%s.png' % freq))
    plt.close()

    assert np.all(np.isfinite(data_df_nonan.values))

    # tst_times_dict = {}  # _id:[] for _id in dwd_ids}
    # for _stn in dwd_ids:
    #     if _stn in df_intense_events.columns:
    #         stns_events = df_intense_events.loc[
    #             :, _stn].dropna(how='all')
    #         if stns_events.index.size > 0:
    #             tst_times_dict[_stn] = stns_events.index.intersection(
    #                 data_df_nonan.index)
    #
    # _stns_with_evts = data_df_nonan.columns.intersection(
    #     list(tst_times_dict.keys()))
    # data_df_nonan = data_df_nonan.loc[:, _stns_with_evts]
    # print(data_df_nonan)
    # print(tst_times_dict)

    # assert data_df.shape[1] == len(tst_times_dict)

    n_dims = data_df_nonan.shape[1]

    usph_vecs = gen_usph_vecs_mp(n_vecs, n_dims, n_cpus)

    ref_pts = data_df_nonan.values.copy(order='c')
    ref_pts.shape
    # usph_vecs.shape
    # tst_depths_dict = {}

    # for tst_stn in tst_times_dict:

    # tst_stn in data_df.columns

    # df = dwd_hdf5.get_pandas_dataframe_for_date([tst_stn],
    # event_date='2017-06-30')
    # test_pts = data_df_nonan.loc[
    # tst_times_dict[tst_stn], :].values.copy(order='c')
    test_pts = data_df_nonan.loc[dates_many_stns, :].values.copy(order='c')

    # test_pts.shape
    ref_pts_recent = data_df_nonan.loc['2010':, :].values.copy(order='c')
    depths_all = depth_ftn_mp(
        ref_pts_recent, ref_pts_recent, usph_vecs, n_cpus, 1)
    print(depths_all.shape)

    data_ser_depth_all = pd.DataFrame(index=data_df_nonan.index,
                                      data=depths_all)
    depths_all_high_idx = np.where(depths_all <= 10)[0]
    depths_all_high = depths_all[depths_all <= 10]

    plt.ioff()
    plt.figure(figsize=(16, 8), dpi=200)
    plt.scatter(range(len(depths_all)), depths_all, label='d>10', c='b')
    plt.scatter(depths_all_high_idx, depths_all_high, c='r', label='d<=10')
    plt.xlabel('Hourly time steps 1995-2019', fontsize=16)
    plt.ylabel('Depth d', fontsize=16)
    plt.legend(loc=0)
    plt.grid(alpha=0.25)
    xticks = int(len(data_df_nonan.index) / (24 * 365))
    plt.xticks(np.linspace(0, len(data_df_nonan.index), 25),
               labels=np.unique(data_df_nonan.index.year))
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, r'hourly_events_depth.png'))
    plt.close()

    plt.ioff()
    plt.figure(figsize=(16, 8), dpi=200)
    for dval in range(1, 11):
        idx_unusual_events = data_ser_depth_all[
            data_ser_depth_all <= dval].dropna()
        idx_unusual_events_daily = idx_unusual_events.groupby(
            idx_unusual_events.index.year).sum()

        plt.plot(
            idx_unusual_events_daily.index,
            idx_unusual_events_daily.values,
            label='d<=%d' % dval)
        plt.scatter(
            idx_unusual_events_daily.index,
            idx_unusual_events_daily.values,
            marker='d')

    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Sum of unusual hours per year', fontsize=16)
    plt.legend(loc=0)
    plt.grid(alpha=0.25)

    plt.xticks(idx_unusual_events_daily.index,
               labels=np.unique(data_df_nonan.index.year))
    plt.ylim([0, max(idx_unusual_events_daily.values) + 5])
    plt.yticks([0, 10, 20, 50, 100, 200, 300, 400, 500])
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, r'sum_hourly_events_depth5.png'))
    plt.close()

    timesteps_event = data_df_nonan.iloc[depths_all_high_idx, :]

    # timesteps_event.T.describe()
    for _ii, _dd in zip(timesteps_event.index, depths_all_high):

        df_nonan = timesteps_event.loc[_ii, :].dropna(how='all')

        xvals = in_df_coords_utm32.loc[df_nonan.index, 'X'].values.ravel()
        yvals = in_df_coords_utm32.loc[df_nonan.index, 'Y'].values.ravel()

        fig, (ax) = plt.subplots(
            1, 1, figsize=(12, 8), dpi=100)
        im0 = ax.scatter(xvals, yvals, c=df_nonan.values,
                         marker='x',
                         cmap=cmap_ppt)

        ax.set(xlabel='X [m]', ylabel='Y [m]',
               title='%s - Depth D=%d' % (_ii, _dd))

        # IMPORTANT ANIMATION CODE HERE
        # Used to keep the limits constant
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set(xticks=[], yticks=[])
        ax.grid(alpha=0.25)
        for i, txt in enumerate(df_nonan.values):
            if txt > 0.2:
                ax.annotate(round(txt, 1), (xvals[i],
                                            yvals[i]),
                            color='b')

            elif txt < 0.2:
                ax.annotate(round(txt, 1), (xvals[i],
                                            yvals[i]),
                            color='grey')
        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir, r'hourly_%s.png'
                                 % str(_ii).replace(':', '_')))
        plt.close()
        break

    for dval in range(1, 11):

        plt.ioff()
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12, 8), dpi=100)

        df_depth_val = data_ser_depth_all[
            data_ser_depth_all.values == dval]
        # max_sum_ = max(timesteps_event.loc[df_depth_val.index, :].sum(axis=1))

        for time_idx in df_depth_val.index:
            df_nonan = timesteps_event.loc[time_idx, :].dropna(how='all')

            cum_pcp_event = np.cumsum(df_nonan.values)
            sum_pcp_event = df_nonan.values.sum()
            x_lorenz = cum_pcp_event / sum_pcp_event
            x_lorenz = np.insert(x_lorenz, 0, 0)

            # df_nonan_pos = df_nonan[df_nonan > 0]

            # from statsmodels import distributions
            # ax.scatter(np.arange(x_lorenz.size) / (
            #     x_lorenz.size - 1), x_lorenz,
            #     marker='x')
            ax1.plot(np.arange(x_lorenz.size) / (
                x_lorenz.size - 1), x_lorenz, alpha=0.5)

            # break
        # line plot of equality
        ax1.plot([0, 1], [0, 1], color='k')
        ax1.set(xlabel='Station ID', ylabel='Pcp contribution',
                title='Depth D=%d' % (dval))

        # ax.set_ylim(0, 1.01)
        # ax.set(xticks=[], yticks=np.unique([cum_pcp_event]))
        ax1.grid(alpha=0.25)

        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir, r'_lorenzhourly_%d.png'
                                 % dval))
        plt.close()
        break

    depths_all_high.shape

    # plt.scatter(range(len(depths_all)), depths_all)
    depths = depth_ftn_mp(ref_pts, test_pts, usph_vecs, n_cpus, 1)

    depths.shape
    print('done calculating depth values')

    depths_ser = pd.Series(
        # index=tst_times_dict[tst_stn],
        index=dates_many_stns,
        data=depths)

    depths_all_high_idx = np.where(depths_ser <= 10)[0]
    depths_all_high = depths_ser[depths_ser <= 10]
    print('\n')
    # break

    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)

    plt.plot(depths_ser.index, depths_ser.values, c='r')
    plt.scatter(depths_ser.index, depths_ser.values, marker='o', c='r')
    plt.xlabel('Timestep', fontsize=14)
    plt.ylabel('Depth d', fontsize=14)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, 'depth_events_%s.png' % freq))
    plt.close()

    plt.ioff()
    plt.figure(figsize=(12, 8), dpi=200)

    plt.scatter(sum_events.values, depths_ser.values, marker='X', c='g', s=30)
    plt.xlabel('Sum event [mm/%s]' % freq, fontsize=14)
    plt.ylabel('Depth d', fontsize=14)
    plt.grid(alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_save_dir, 'scatter_depth_events_%s.png' % freq))
    plt.close()

    # TODO: comtine here
    # depths_ser.loc['2019']

    depths_ser_d3 = depths_ser[depths_ser == 1]
    depths_ser_d10 = depths_ser[depths_ser == 8]

    vals_d3 = data_df_nonan.loc[depths_ser_d3.index, :]

    # from pykrige import OrdinaryKriging as OK

    for event_idx in depths_ser_d10.index:
        depth_val = depths_ser.loc[event_idx]
        event_start = event_idx - pd.Timedelta(minutes=int(freq.split('m')[0]))

        df = dwd_hdf5.get_pandas_dataframe_between_dates(
            dwd_ids,
            event_start=event_start,
            event_end=event_idx)
        df_nonan = df.dropna(how='all', axis=1)

        xvals = in_df_coords_utm32.loc[df_nonan.columns, 'X'].values.ravel()
        yvals = in_df_coords_utm32.loc[df_nonan.columns, 'Y'].values.ravel()
        all_images = []

        for ii, _evt in enumerate(df_nonan.index):
            pcp_vals1 = df_nonan.iloc[ii, :].values
            plt.ioff()
            fig, (ax) = plt.subplots(
                1, 1, figsize=(12, 8), dpi=100)
            im0 = ax.scatter(xvals, yvals, c=pcp_vals1,
                             marker='x',
                             cmap=cmap_ppt)

            ax.set(xlabel='X [m]', ylabel='Y [m]',
                   title='%s - Depth D=%d' % (_evt, depth_val))

            # IMPORTANT ANIMATION CODE HERE
            # Used to keep the limits constant
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set(xticks=[], yticks=[])
            ax.grid(alpha=0.25)
            for i, txt in enumerate(df_nonan.iloc[ii, :].values):
                if txt > 0.2:
                    ax.annotate(round(txt, 1), (xvals[i],
                                                yvals[i]),
                                color='b')

                elif txt < 0.2:
                    ax.annotate(round(txt, 1), (xvals[i],
                                                yvals[i]),
                                color='grey')
            # plt.colorbar(im0, ax=ax,
            #              ticks=bound_ppt,
            #              orientation="vertical",  shrink=0.85,
            #              pad=0.04,  label='[mm/5min]')

            plt.grid(alpha=0.25)
            # Used to return the plot as an image rray
            fig.canvas.draw()  # draw the canvas, cache the renderer
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # , vmax=max(max_stn_pcp,
            all_images.append(image)
            plt.close('all')

        print(' saving image')
        #         kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
        imageio.mimsave(
            os.path.join(out_save_dir, r'radar_event_date_%s_%s.gif'
                         % (str(event_start).replace(' ', '_').replace(':', '_'),
                            '5min')),
            all_images, fps=1)
        print(' done saving image')
        break

    #===================================================================
    # calculate copulas
    #===================================================================
    # if you want to specify an outputfile -> os.path.join(savefolder,
    import scipy.stats as stats

    modulepath = r'C:\Users\hachem\Desktop\WPy64-3880\python-3.8.8.amd64\Lib\site-packages\rmwspy'
    sys.path.append(modulepath)

    from random_mixing_whittaker_shannon import RMWS
    import gcopula_sparaest as sparest
    import covariancefunction as cov_func
    xs = in_df_coords_utm32.lon.min()
    # xinc = 0.014798
    xinc, yinc = 0.02, 0.02
    xsize = int((in_df_coords_utm32.lon.max() -
                 in_df_coords_utm32.lon.min()) / xinc) + 1

    ys = in_df_coords_utm32.lat.min()
    ysize = int((in_df_coords_utm32.lat.max() -
                 in_df_coords_utm32.lat.min()) / yinc) + 1
    # data_df_nonan.loc[dates_many_stns, :].values.copy(order='c')

    neigbhrs_radius_dwd_list = np.arange(0, 53000, 2000)
    nlag = len(neigbhrs_radius_dwd_list)
    from scipy.spatial import distance
    # nor_fact = 10
    plt.ioff()
    fig, (ax) = plt.subplots(
        1, 1, figsize=(12, 8), dpi=100)
    for _dd, cc in zip(range(1, 10),
                       ['grey', 'b', 'g', 'r', 'c', 'm', 'orange', 'k', 'darkred']):
        depths_ser_d10 = data_ser_depth_all[data_ser_depth_all == _dd].dropna()

        for _ii, _dd in zip(depths_ser_d10.index, depths_ser_d10.values):
            print(_ii)
            df_nonan = data_df_nonan.loc[_ii, :].dropna(how='all')
            # break
            xvals = in_df_coords_utm32.loc[df_nonan.index, 'X'].values.ravel()
            yvals = in_df_coords_utm32.loc[df_nonan.index, 'Y'].values.ravel()

            dwd_xy_ngbrs = np.c_[xvals, yvals]
            pre = df_nonan.values
            pva = np.var(pre)
            vg = np.zeros(nlag)
            ng = np.zeros(nlag)
            nact = dwd_xy_ngbrs.shape[0]
            ds = distance.cdist(
                dwd_xy_ngbrs, dwd_xy_ngbrs, metric='euclidean')
            dx1 = np.empty((nact, nact))
            dx1[:, :] = dwd_xy_ngbrs[:, 0]
            dx2 = dx1.transpose()
            dxa = dx1 - dx2
            dy1 = np.empty((nact, nact))
            dy1[:, :] = dwd_xy_ngbrs[:, 1]
            dy2 = dy1.transpose()
            dya = dy1 - dy2
            lag = 5000.0
            v1 = np.empty((nact, nact))
            v1[:, :] = pre
            # print(v1)
            v2 = v1.transpose()
            v3 = (v1 - v2)**2
            # print(v3)
            ds = ds.flatten()
            v3 = v3.flatten()
            v3 = v3[ds < nlag * lag]
            ds = ds[ds < nlag * lag]
            for i in range(nlag):
                va = v3[ds < (i + 0.5) * lag]
                da = ds[ds < (i + 0.5) * lag]
                va = va[da > (i - 0.5 * lag)]
                # print(va)
                vg[i] = np.mean(va)
            vg_normed = vg / vg[-1]
            ax.plot(neigbhrs_radius_dwd_list,
                    vg_normed, c=cc, alpha=0.25)

        ax.plot(neigbhrs_radius_dwd_list,
                vg_normed, c=cc, label='%d' % _dd)
        # dwd_lat_lon = np.c_[xvals, yvals]
        # p_xy = np.copy(dwd_lat_lon)
        #
        # p_xy[:, 0] = (p_xy[:, 0] - xs) / xinc
        # p_xy[:, 1] = (p_xy[:, 1] - ys) / yinc
        # p_xy = p_xy.astype(int)
        #
        # u = (stats.rankdata(df_nonan) - 0.5) / df_nonan.shape[0]
        #
        # # observations in copula (rank) space
        # covmods = ['Exp']
        # #['Mat', 'Exp', 'Sph',]
        # # covariance function that will be tried for the fitting
        # ntries = 6
        # # number of tries per covariance function with random subsets
        # cmods = sparest.paraest_multiple_tries(
        #     np.copy(p_xy),
        #     u,
        #     ntries=[ntries, ntries],
        #     n_in_subset=5,               # number of values in subsets
        #     neighbourhood='nearest',     # subset search algorithm
        #     covmods=covmods,             # covariance functions
        #     outputfile=None)       # store all fitted models in an output file
        #
        # # take the copula model with the highest likelihood
        # # reconstruct from parameter array
        # likelihood = -555
        # for model in range(len(cmods)):
        #     for tries in range(ntries):
        #         if cmods[model][tries][1] * -1. > likelihood:
        #             likelihood = cmods[model][tries][1] * -1.
        #             cmod = '0.01 Nug(0.0) + 0.99 %s(%1.3f)' % (
        #                 covmods[model], cmods[model][tries][0][0])
        #             if covmods[model] == 'Mat':
        #                 cmod += '^%1.3f' % (cmods[model][tries][0][1])
        #
        # cov_mod_sill = float(cmod.split('+')[1].split(' ')[1])
        # cov_mod_range = float(cmod.split(
        #     '+')[1].split(' ')[2].split('(')[1].strip(')'))
        # h = np.arange(0, cov_mod_range + 100)
        # exp_cov = cov_func.type_exp(h, Range=cov_mod_range, Sill=cov_mod_sill)

        # plt.show()
        # ax.set_xlim(0, 250)
        # ax.set_ylim(0, 1.01)
    ax.set_xlabel('Range [m]')
    ax.set_ylabel('Gamma (h)')
    ax.legend(loc='lower left', ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(
        out_save_dir, 'vg_events_all_normed10.png'))
    plt.close()

    return


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
