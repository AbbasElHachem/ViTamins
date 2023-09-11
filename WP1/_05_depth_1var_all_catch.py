'''
@author: Abbas-Uni-Stuttgart

September, 2023

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
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import tqdm
# from depth_funcs import (
    # gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

from depth_funcs_new import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

# from depth_funcs_new import depth_ftn_mp

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5


def main():
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    out_save_dir = data_path / r"Results\01_DD_plots"
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    
    var_to_test = 'precipitation'
    
    path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)

    #===========================================================================
    # Depth func parameters
    n_vecs = int(1e4)
    n_cpus = 5
    
    neighbor_to_chose = 10
    
    
    beg_date = '1970-01-01'
    end_date = '2015-09-30'

    date_range = pd.date_range(start=beg_date, end=end_date, freq='D')
    
    #===========================================================================
    data_hdf5 = HDF5(infile=path_to_data)
    catch_ids = data_hdf5.get_all_names()
    
    catch_coords = data_hdf5.get_coordinates(ids=catch_ids)

    df_coords = pd.DataFrame(index=catch_ids,
        data=catch_coords['easting'], columns=['X'])
    y_dwd_coords = catch_coords['northing']
    df_coords.loc[:, 'Y'] = y_dwd_coords

    # create a tree from DWD coordinates

    coords_xy = [(x, y) for x, y in zip(df_coords.loc[:, 'X'].values,
                                         df_coords.loc[:, 'Y'].values)]

    # create a tree from coordinates
    catch_points_tree = cKDTree(coords_xy)

    data_zero = np.zeros(shape=(len(date_range), len(catch_ids)))

    df_depth_all = pd.DataFrame(data=data_zero, index=date_range, columns=catch_ids)
    idx_list = []
    
    plt.ioff()
    plt.figure(figsize=(5, 5), dpi=300)
    for catch_id in tqdm.tqdm(catch_ids):
        print(catch_id)
        # break
        df_stn = data_hdf5.get_pandas_dataframe_between_dates(
            catch_id, event_start=beg_date, event_end=end_date)
        df_stn = df_stn.dropna(how='all')
        # normalize by median
        
        # df_stn_norm_orig = df_stn# / df_stn.median()
        # df_stn[df_stn > 90] = np.nan

        (xdwd, ydwd) = (df_coords.loc[catch_id, 'X'], df_coords.loc[catch_id, 'Y'])

        distances, indices = catch_points_tree.query(np.array([xdwd, ydwd]), k=neighbor_to_chose)

        # stn_near = list(np.array(dwd_ids)[indices[:7]])
        
        # indices = indices[distances > 0]
        # distances = distances[distances > 0]
        stn_near = list(np.array(catch_ids)[indices])
        assert len(stn_near) == len(distances)
        
        
        df_neighbors = pd.DataFrame(index=df_stn.index, columns=stn_near)
        
        nbr_idx = 0
        # max_nbr = 10
        # start_idx = 0
        # end_idx = 6
        
        usph_vecs = gen_usph_vecs_mp(n_vecs, df_stn.columns.size, n_cpus)
        idx_list = []
    
        for _st in stn_near:
            if nbr_idx < 9:
                try:
                    df_st = data_hdf5.get_pandas_dataframe_between_dates(
                        _st,
                        event_start=df_neighbors.index[0],
                        event_end=df_neighbors.index[-1]).dropna()
                        

                    cmn_idx = df_neighbors.index.intersection(
                        df_st.index)
    
                    if cmn_idx.size > 0.9 * df_stn.index.size:
                        # print(df_st.head())
    
                        df_neighbors.loc[cmn_idx, _st] = df_st.loc[cmn_idx, :].values.ravel()
                        nbr_idx += 1
                        print(nbr_idx)
                except Exception as msg:
                    print(msg)
                    continue
                
        df_stn_near_period = df_neighbors.dropna(how='any', axis=1)
    
        
        (xnear, ynear) = (df_coords.loc[df_stn_near_period.columns, 'X'],
            df_coords.loc[df_stn_near_period.columns, 'Y'])
        if len(df_stn_near_period.columns) > 3:
            points = np.array([(x, y) for x, y in zip(xnear, ynear)])
            hull = ConvexHull(points)
    
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0],  points[simplex, 1], 'grey',
                         alpha=0.25,  linestyle='--',  linewidth=1)
    
            plt.plot(points[:, 0], points[:, 1], '.', color='r')
            # plt.scatter(xdwd, ydwd, marker='X', color='b')
    
            #print(stn_near)
    
    
        df_nearby_resampled_norm = df_stn_near_period.loc[
            df_stn.index.intersection(df_stn_near_period.index), :].dropna(axis=0, how='any')
        # daily
        for _col in df_nearby_resampled_norm.columns:
        
            df_nearby_resampled_norm.loc[df_nearby_resampled_norm[_col] == 0, _col
            ] = np.random.random() * np.random.uniform(0.02, 0.1, len(df_nearby_resampled_norm.loc[df_nearby_resampled_norm[_col] == 0]))
        # df_nearby_resampled_norm
        df_pos = df_nearby_resampled_norm
        if len(df_pos.index) > 10:
        
            tot_refr_var_arr = df_pos.values.astype(float).copy('c')
            # tot_refr_var_arr.T.shape
            usph_vecs = gen_usph_vecs_mp(n_vecs, df_pos.columns.size, n_cpus)
            depths2 = depth_ftn_mp(tot_refr_var_arr, tot_refr_var_arr, usph_vecs, n_cpus, 1)
            print('done calculating depth')
        
            df_pcp_depth = pd.DataFrame(index=df_pos.index, data=depths2,
                                        columns=['d'])
        
            df_low_d = df_pcp_depth[(df_pcp_depth > 1) & (
                df_pcp_depth <= 4)].dropna()
            # df_pcp_depth[df_pcp_depth.values == 1].dropna()
            # df_low_d.shape[0] / df_pcp_depth.shape[0]
            de = df_pos.loc[df_low_d.index]  # .sum(axis=1)#.max()
            # de.iloc[:10, :]
            de[de < 0.1] = 0
            # de.loc[de.sum(axis=1).sort_values().in
            df_depth_all.loc[de.index, de.columns] += 1
            idx_list.append(de.index)
            
                # break
    plt.grid(alpha=0.25)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir, 'hulls_%s_22a.png' % catch_id),
        bbox_inches='tight')
    plt.close()

    print('done')

    df_results = df_depth_all[df_depth_all.sum(axis=1) > 0]
    # df_results.sum(axis=1).max()

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
        # try:
        ids_event = df_results.columns[
                np.where(df_results.loc[_idx, :] > 0)[0].astype(int)]
        # if len(ids_event) > 7:
        # print('dd')
        df_depth_low = data_hdf5.get_pandas_dataframe_for_date(
                ids_event, event_date=_idx)
        pcp_data_events.loc[_idx, ids_event] = df_depth_low.values
        # except Exception as msg:
            # print(msg)
            # continue
    # pcp_data_events.dropna(how='all', axis=1, inplace=True)
    # plot()
    # pcp_data_events.read_csv(os.path.join(out_save_dir, 'df_unusuals.csv'), sep=';', index_col=0,
                             # parse_dates=True, infer_datetime_format=True)
    # pcp_data_events.sum(axis=1).plot(legend=False)
    max_Events = pcp_data_events.sum(axis=1).sort_values()

    max_event = pcp_data_events.loc[
        max_Events.index[-10:], :].dropna(axis=1, how='all').sort_index()  # .plot()

    for _idx in max_event.index:
        data_event = max_event.loc[_idx, :].dropna()
        xcoords = df_coords.loc[data_event.index, 'X']
        ycoords = df_coords.loc[data_event.index, 'Y']

        df_ = data_hdf5.get_pandas_dataframe_for_date(
            catch_ids, event_date=str(_idx))

        plt.ioff()
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
        pcp = ax1.scatter(xcoords, ycoords, c=data_event.values, marker='X', s=data_event.values,
                          vmin=0.5, cmap=plt.get_cmap('jet_r'))
        ax1.scatter(df_coords.loc[df_.columns, 'X'], df_coords.loc[df_.columns, 'Y'], c=df_.values,
                    marker='X', s=50, cmap=plt.get_cmap('jet_r'), vmin=0.5)

        fig.colorbar(pcp, shrink=0.75, label='mm/hr')
        ax1.grid(alpha=0.25)
        ax1.set_title('%s' % (_idx))
        plt.tight_layout()
        plt.savefig(os.path.join(out_save_dir, 'max_event_sum_%s_dwd.png'
                         % str(_idx).replace(':', '_').replace(' ', '_')))

        plt.close()
        break#




    #===========================================================================
    # In this study, we normalized the data
    # depth between 0 and 1 by dividing the depth by half the total
    # number of points in the convex hull.
    #===========================================================================
    
    
    pass


# print('done')

if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    