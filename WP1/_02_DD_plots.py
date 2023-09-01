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
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp)

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5


def main():
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    out_save_dir = data_path / r"Results\01_DD_plots"
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    
    var_to_test = 'discharge_vol'
    
    path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)

    #===========================================================================
    # Depth func parameters
    n_vecs = int(1e4)
    n_cpus = 3
    
    neighbor_to_chose = 2
    
    
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
    
    time_steps_shift = [1, 2, 3, 4]
    
    for catch_id in tqdm.tqdm(catch_ids[1:10]):
        print(catch_id)
        # break
        df_stn = data_hdf5.get_pandas_dataframe_between_dates(
            catch_id, event_start=beg_date, event_end=end_date)
        df_stn = df_stn.dropna(how='all')
        # normalize by median
        
        df_stn_norm_orig = df_stn / df_stn.median()
        # df_stn[df_stn > 90] = np.nan

        (xdwd, ydwd) = (df_coords.loc[catch_id, 'X'], df_coords.loc[catch_id, 'Y'])

        distances, indices = catch_points_tree.query(np.array([xdwd, ydwd]), k=neighbor_to_chose + 100)

        # stn_near = list(np.array(dwd_ids)[indices[:7]])
        
        indices = indices[distances > 0]
        distances = distances[distances > 0]
        stn_near = list(np.array(catch_ids)[indices])
        assert len(stn_near) == len(distances)
        
        
        df_neighbors = pd.DataFrame(index=df_stn_norm_orig.index, columns=stn_near)
        
        df_orig_shift = pd.DataFrame(index=df_stn_norm_orig.index, columns=time_steps_shift)
        
        # nbr_idx = 0
        
        usph_vecs = gen_usph_vecs_mp(n_vecs, df_stn_norm_orig.columns.size, n_cpus)

        for sep_dist, _st in tqdm.tqdm(zip(distances, stn_near)):
            
            df_st = data_hdf5.get_pandas_dataframe_between_dates(_st, event_start=df_stn_norm_orig.index[0],
                    event_end=df_stn_norm_orig.index[-1]).dropna()
            df_st_norm = df_st / df_st.median()
            
            df_ngbr_shift = pd.DataFrame(index=df_stn_norm_orig.index, columns=time_steps_shift)

            # if cmn_idx.size > 0.9 * df_st_norm.index.size:
                # df_stn_norm_orig = df_stn_norm_orig.loc[cmn_idx]
                # df_st_norm = df_st_norm.loc[cmn_idx]
                
                # refr_var_arr_orig = df_stn_norm_orig.values.astype(float).copy('c')
                # refr_var_arr_ngbr = df_st_norm.values.astype(float).copy('c')
                
                # depths_da_da = depth_ftn_mp(refr_var_arr_orig, refr_var_arr_orig,
                #                             usph_vecs, n_cpus, 1)
                #
                # depths_da_db = depth_ftn_mp(refr_var_arr_orig, refr_var_arr_ngbr,
                #                             usph_vecs, n_cpus, 1)
                # depths_db_da = depth_ftn_mp(refr_var_arr_ngbr, refr_var_arr_orig,
                #                             usph_vecs, n_cpus, 1)
                
                # norm_fact = cmn_idx.size / 2
                # plt.ioff()                
                # fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
                
                # ax.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.5)
                # ax.scatter(depths_da_da / norm_fact , depths_da_db / norm_fact, s=5,
                           # marker='o', edgecolor='k', facecolor='gray', alpha=0.2)
                
                # plt.show()
            for tshift in time_steps_shift:
                df_stn_norm_orig_shifted = df_stn_norm_orig.shift(tshift)
                df_st_norm_shifted = df_st_norm.shift(tshift)
                
                df_orig_shift.loc[df_stn_norm_orig_shifted.index, tshift] = df_stn_norm_orig_shifted.values.ravel()
                df_ngbr_shift.loc[df_st_norm_shifted.index, tshift] = df_st_norm_shifted.values.ravel()
                      

            df_orig_shift_nonan = df_orig_shift.dropna(axis=0)
            df_ngbr_shift_nonan = df_ngbr_shift.dropna(axis=0)
            cmn_idx = df_orig_shift_nonan.index.intersection(df_ngbr_shift_nonan.index)
            
            norm_fact_4d = cmn_idx.size / 2
                # break
            usph_vecs = gen_usph_vecs_mp(n_vecs, df_ngbr_shift_nonan.columns.size, n_cpus)
            shifted_var_arr_orig = df_orig_shift_nonan.loc[cmn_idx].values.astype(float).copy('c')
            shifted_var_arr_ngbr = df_ngbr_shift_nonan.loc[cmn_idx].values.astype(float).copy('c')

            depths_da_da = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_orig,
                                        usph_vecs, n_cpus, 1)
            
            depths_da_db = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_ngbr,
                                        usph_vecs, n_cpus, 1)
            
            # depths_db_da = depth_ftn_mp(shifted_var_arr_ngbr, shifted_var_arr_ngbr,
                                        # usph_vecs, n_cpus, 1)
            
            points = np.array([(x, y) for x, y in zip(depths_da_da/ norm_fact_4d, depths_da_db/ norm_fact_4d)])
            hull = ConvexHull(points)
                                   
            plt.ioff()                
            fig, ax = plt.subplots(1,1, dpi=300, figsize=(4, 4))
            
            ax.plot([0, 1], [0, 1], c='r', linestyle='--', alpha=0.75)
            ax.scatter(depths_da_da / norm_fact_4d , depths_da_db / norm_fact_4d, s=5,
                       marker='o', edgecolor='k', facecolor='gray', alpha=0.4, label='C-Hull A=%0.2f - n=%d' % (hull.area, cmn_idx.size))
            ax.legend(loc=0, ncols=2)
            ax.set_xlabel(r'$D_{\alpha}(x)$')
            ax.set_ylabel(r'$D_{\beta}(x)$')
            ax.set_title('Catch. %s vs %s - Sep. dist. %d km' 
                         % (catch_id, _st, sep_dist/1e3))
            ax.grid(alpha=0.5)
            plt.savefig(out_save_dir / (r'%s_%s.png' % (catch_id, _st)), bbox_inches='tight')
            plt.close()
            
            # depths2 = depth_ftn_mp(shifted_var_arr_orig, shifted_var_arr_ngbr, usph_vecs, n_cpus, 1)
            print('done calculating depth')


    #===========================================================================
    # In this study, we normalized the data
    # depth between 0 and 1 by dividing the depth by half the total
    # number of points in the convex hull.
    #===========================================================================
    
    
    
    """
    Hence, if the DDplot
    of two catchments is close to the diagonal line, we assume
    the catchments have similar response behaviour. A Q(t) vs Q
    (t − 1) (d = 2) plot for two sample catchments α and β is shown
    in Figure 3(a), while the corresponding Q(t), Q(t − 1) and Q
    (t − 2) (d = 3) plots for the same catchments are shown in Figure
    3(b). A DD-plot where d = 4 is given in Figure 3(c). Figures 3(a)
    and (b) summarize the flow dynamics of the catchment in two
    and three dimensions, where we can see that catchments α and β
    have quite similar behaviour. In Figure 3(c), the DD-plot summarizes
    the four-dimensional relationship between flow
    dynamics of catchments α and β.
    """
    
    pass


# for sep_dist, _st in zip(distances, stn_near):
#
#     if nbr_idx < 7:
#         try:
#             df_st = data_hdf5.get_pandas_dataframe_between_dates(_st, event_start=df_stn_norm.index[0],
#                 event_end=df_stn_norm.index[-1]).dropna()
#             df_st_norm = df_st / df_st.median()
#             cmn_idx = df_neighbors.index.intersection(df_st.index)
#
#             if cmn_idx.size > 0.9 * df_stn.index.size:
#                 # print(df_st.head())
#
#                 df_neighbors.loc[ cmn_idx, _st] = df_st_norm.loc[cmn_idx, :].values.ravel()
#                 nbr_idx += 1
#                 print(nbr_idx)
#         except Exception as msg:
#             print(msg)
#             continue
#
# df_stn_near_period = df_neighbors.dropna(how='all', axis=1)
#
# df_results = df_depth_all[df_depth_all.sum(axis=1) > 0]

# if len(stn_near) > 0:
#     (xnear, ynear) = (df_coords.loc[stn_near, 'X'], df_coords.loc[stn_near, 'Y'])
#     points = np.array([(x, y) for x, y in zip(xnear, ynear)])
#     hull = ConvexHull(points)
#
#     for simplex in hull.simplices:
#         plt.plot(points[simplex, 0], points[simplex, 1], 'grey',
#                  alpha=0.25, linestyle='--', linewidth=1)
#
#     plt.plot(points[:, 0], points[:, 1], '.', color='r')
#     plt.scatter(xdwd, ydwd, marker='X', color='b')
#
#     print(stn_near)
#
# df_nearby_resampled_norm = df_stn_near_period.loc[
#         df_stn_norm.index.intersection(df_stn_near_period.index), :].dropna(axis=0, how='any')
#
# # hourly: 0.1, 0.05,
# # daily
# for _col in df_nearby_resampled_norm.columns:
#
#     df_nearby_resampled_norm.loc[
#         df_nearby_resampled_norm[_col] == 0, _col
#     ] = np.random.random() * np.random.uniform(
#         0.02, 0.1,
#         len(df_nearby_resampled_norm.loc[
#             df_nearby_resampled_norm[_col] == 0]))
#
#
# df_pos = df_nearby_resampled_norm
# if len(df_pos.index) > 10:
#
#     tot_refr_var_arr = df_pos.values.astype(float).copy('c')
#
#     usph_vecs = gen_usph_vecs_mp(
#         n_vecs, df_pos.columns.size, n_cpus)
#     depths2 = depth_ftn_mp(tot_refr_var_arr,
#                            tot_refr_var_arr,
#                            usph_vecs, n_cpus, 1)
#     print('done calculating depth')
#
#     df_pcp_depth = pd.DataFrame(index=df_pos.index,
#                                 data=depths2,
#                                 columns=['d'])
#
#     df_low_d = df_pcp_depth[(df_pcp_depth >= 1) & (
#         df_pcp_depth < 2)].dropna()
#     # df_pcp_depth[df_pcp_depth.values == 1].dropna()
#     df_low_d.shape[0] / df_pcp_depth.shape[0]
#     de = df_pos.loc[df_low_d.index]  # .sum(axis=1)#.max()
#     # de.iloc[:10, :]
#     de[de < 0.1] = 0
#     # de.loc[de.sum(axis=1).sort_values().in
#     df_depth_all.loc[de.index, de.columns] += 1
#     idx_list.append(de.index)
#
#
# plt.grid(alpha=0.25)
# plt.axis('equal')
# plt.tight_layout()
# plt.savefig(os.path.join(
# out_save_dir, 'hulls_%s_22a.png' % catch_id),
# bbox_inches='tight')
# plt.close()
#
# print('done')

if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    