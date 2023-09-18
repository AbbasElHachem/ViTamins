'''
@author: Abbas-Uni-Stuttgart

September, 2023

1:37:50 PM

'''
import os
import sys
import time
import timeit
import glob
import traceback as tb
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull
import tqdm
from depth_funcs import (
    gen_usph_vecs_norm_dist_mp as gen_usph_vecs_mp, depth_ftn_mp, depth_ftn_mp_v2, cmpt_rand_pts_chull_vol)

modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5


def main():
    
    main_dir = Path(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\8344e4f3-d2ea-44f5-8afa-86d2987543a9\data")

    data_dir = main_dir / r'timeseries'
    os.chdir(data_dir)  # _old_recent
    
    df_coords = pd.read_csv((main_dir / r"CAMELS_GB_topographic_attributes.csv"),
                            index_col=0, sep=',')
    
    # def epsg wgs84 and utm32 for coordinates conversion
    df_coords.columns
    x_utm32, y_utm32 = (df_coords['gauge_easting'].values.ravel(),
                         df_coords['gauge_northing'].values.ravel())
    df_coords.loc[:, 'X'] = x_utm32
    df_coords.loc[:, 'Y'] = y_utm32
    
    # 'discharge_spec', 'peti', shortwave_rad
    variables = ['precipitation', 'pet', 'temperature', 'discharge_vol', 
                 'humidity', 'longwave_rad','windspeed']
    
    var_red = ['precipitation', 'discharge_vol'] # temperature
    start_date = '1980'
    # get all .csv file
    df_stns_all = glob.glob('*.csv')
    
    #===========================================================================
    # Depth func parameters
    n_vecs = int(1e4)
    n_cpus = 7
    use_red_var = False
    min_D_Val = 4
    do_plot = False
    #===========================================================================
    
    
    data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')
    # =============================================================
    out_save_dir = data_path / r"Results\02_1catch_all_var"
    
    if not os.path.exists(out_save_dir):
        os.mkdir(out_save_dir)
    
    # var_to_test = 'discharge_vol'
    
    df_all_events = pd.DataFrame(
        index=pd.date_range(start='1970-01-01', end='2015-09-30', freq='D'),
        columns=df_coords.index, data=np.nan)
    
    
    for i_idx, df_stn in tqdm.tqdm(enumerate(df_stns_all)):
        # break
    
        stn = int(df_stn.split('19701001')[0].split('_')[-2])#.split('_')[0]
        out_save_dir_stn = out_save_dir / (r"%s" % stn)
    
        if not os.path.exists(out_save_dir_stn):
            os.mkdir(out_save_dir_stn)
        
        if len(str(stn)) > 0:
            print('{} / {}'.format(i_idx + 1, len(df_stns_all)))
            print(stn, '-', df_stn)
            df_in = pd.read_csv(df_stn, sep=',', index_col=0, engine='c', low_memory=False)
            df_in.index = pd.to_datetime(df_in.index, format='%Y-%m-%d')
            df_in.dropna(how='any', inplace=True)
            df_in_diff = df_in.diff(1).dropna()
            
            df_q = df_in_diff[df_in_diff.loc[:, 'discharge_vol'] > 0].max()*0.25
            
            df_in_diff_pos = df_in_diff[df_in_diff.loc[:, 'discharge_vol'] > df_q.discharge_vol]
            
            df_in_pos = df_in.loc[df_in_diff_pos.index, :]
            if use_red_var:
                df_in = df_in_pos.loc[:, var_red]
            else:
                df_in = df_in_pos.loc[:, variables]
#            
            usph_vecs = gen_usph_vecs_mp(n_vecs, len(df_in_pos.columns), n_cpus)
            
            df_in_vals = df_in_pos.values.astype(float).copy('c')
            depths_da_da = depth_ftn_mp_v2(df_in_vals, df_in_vals, usph_vecs, n_cpus)
            
            
            idx_d1 = np.where(1== depths_da_da)[0]
            idx_low_d = np.where((1 < depths_da_da) & (depths_da_da <= min_D_Val))[0]
            # idx_low_d = np.where(depths_da_da <= min_D_Val)[0]
            
            # plt.scatter(range(len(depths_da_da)), depths_da_da)
            # plt.show()
            df_in_low_d1 = df_in_diff_pos.iloc[idx_d1,:]
            df_in_low_d = df_in_diff_pos.iloc[idx_low_d,:]
            
            df_in_low_d1 = df_in.loc[df_in_low_d1.index,:]
            df_in_low_d = df_in.loc[df_in_low_d.index,:]
            
            # low_depth = depths_da_da[depths_da_da < 10]
            nlow_d = idx_low_d.shape[0]
            nlow_d1 = idx_d1.shape[0]
            
            df_all_events.loc[df_in_low_d.index, stn] = 1
            
            
            print('Computing unit hull volume...')
            scipy_hull_vol = ConvexHull(df_in_vals).volume
            print('scipy unit_hull_vol:', scipy_hull_vol)
            chk_iter = int(1e4)
            max_iters = 10
            vol_tol =0.91
            unit_hull_vol = cmpt_rand_pts_chull_vol(df_in_vals,  usph_vecs, chk_iter, max_iters, n_cpus,  vol_tol)
    
    
            # depths_da_da_norm = depths_da_da / (depths_da_da.shape[0] /2)
            # depths_da_da_norm.max()
            # if use_red_var:
            #
            #     plt.ioff()
            #     # Visualizing 3-D numeric data with Scatter Plots
            #     # length, breadth and depth
            #     fig = plt.figure(figsize=(6, 6), dpi=300)
            #     ax = fig.add_subplot(111) # , projection='3d'
            #
            #     xs = df_in.loc[:, var_red[0]]
            #     ys = df_in.loc[:, var_red[1]]
            #     ax.scatter(xs, ys, s=depths_da_da_norm*1000, alpha=0.76, edgecolors='r')
            #
            #     # zs = df_in.loc[:, var_red[2]]
            #
            #     # ax.scatter(ys, xs, zs, s=depths_da_da_norm*1000, alpha=0.76, edgecolors='r')
            #
            #     ax.set_xlabel(var_red[1])
            #     ax.set_ylabel(var_red[0])
            #     # ax.set_zlabel(var_red[2])
            #
            #     # Update the axis view and title
            #     # ax.view_init(-120, 60)
            #
            #     plt.savefig(out_save_dir_stn / (r'2d_%s_PQT.png' % (stn)), bbox_inches='tight')
            #     plt.close('all')
            if do_plot:
                for _col in df_in.columns:
                    plt.ioff()
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=300)
                    
                    ax.plot(df_in.index, df_in.loc[:,_col].values, color='k', alpha=0.4)
                    ax.scatter(df_in_low_d.index, df_in_low_d.loc[:,_col].values, facecolor='r', edgecolor='darkred', marker='X', label='n=%d' % nlow_d)
                    # ax.scatter(df_in_low_d1.index, df_in_low_d1.loc[:,_col].values, facecolor='b', edgecolor='darkblue', marker='o', label='n=%d' % nlow_d1)
                    
                    ax.set_title('%s' % _col)
                    ax.set_xlabel('%s - Time' % _col)
                    ax.set_ylabel('Daily values')
                    ax.grid(alpha=0.5)
                    ax.legend(loc=0)
                    plt.savefig(out_save_dir_stn / (r'_%s_%s_pos_dQ5.png' % (stn, _col)), bbox_inches='tight')
                    plt.close('all')
                
                
                
                # disch vs pcp
                # disch vs pet
                # disch vs wind
                # disch vs temp
                # disch vs hum
            print('Plotting')
            plt.ioff()
            fig, ((ax1, ax2, ax3),
                  (ax5, ax6, ax7)) = plt.subplots(nrows=2, ncols=3, figsize=(12, 8), dpi=200, sharex=True)
                          
            ax1.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'precipitation'].values, facecolor='gray', edgecolor='k', alpha=0.2, label='n=%d' % df_in.index.size)
            
            
            ax1.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'precipitation'].values, facecolor='r', edgecolor='darkred', label='n=%d' % df_in_low_d.index.size)
            # ax1.scatter(df_in_low_d1.loc[:,'discharge_vol'].values,
                        # df_in_low_d1.loc[:, 'precipitation'].values, marker='o', facecolor='b', edgecolor='darkred', label='n=%d' % df_in_low_d.index.size)
            
            
            ax1.set_xlabel('Q vol')
            ax1.set_ylabel('Pcp')
            
            ax1.grid(alpha=0.5)
            ax1.legend(loc=0)
            ax2.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'pet'].values, facecolor='gray', edgecolor='k', alpha=0.2)
            ax2.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'pet'].values, facecolor='b', edgecolor='darkblue', marker='o')
            # ax2.scatter(df_in_low_d1.loc[:,'discharge_vol'].values,
                        # df_in_low_d1.loc[:, 'pet'].values, marker='o', facecolor='g', edgecolor='darkred', label='n=%d' % df_in_low_d.index.size)
            
            ax2.set_xlabel('Q vol')
            ax2.set_ylabel('Pet')
            
            ax2.grid(alpha=0.5)
            
            ax3.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'temperature'].values, facecolor='gray', edgecolor='k', alpha=0.2)
            ax3.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'temperature'].values, facecolor='g', edgecolor='darkgreen', marker='D')
            
            ax3.set_xlabel('Q vol')
            ax3.set_ylabel('temperature')
            ax3.grid(alpha=0.5)
            
            
            ax5.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'humidity'].values, facecolor='gray', edgecolor='k', alpha=0.2)
            ax5.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'humidity'].values, facecolor='orange', edgecolor='darkorange', marker='X')
            
            ax5.set_xlabel('Q vol')
            ax5.set_ylabel('humidity')
            ax5.grid(alpha=0.5)
            
            ax6.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'longwave_rad'].values, facecolor='gray', edgecolor='k', alpha=0.2)
            ax6.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'longwave_rad'].values, facecolor='g', edgecolor='darkgreen', marker='D')
            
            ax6.set_xlabel('Q vol')
            ax6.set_ylabel('longwave_rad')
            ax6.grid(alpha=0.5)
            
            ax7.scatter(df_in.loc[:,'discharge_vol'].values,
                       df_in.loc[:, 'windspeed'].values, facecolor='gray', edgecolor='k', alpha=0.2)
            ax7.scatter(df_in_low_d.loc[:,'discharge_vol'].values,
                       df_in_low_d.loc[:, 'windspeed'].values, facecolor='m', edgecolor='pink')
            
            ax7.set_xlabel('Q vol')
            ax7.set_ylabel('windspeed')
            ax7.grid(alpha=0.5)
            
            plt.tight_layout()
            
            plt.savefig(out_save_dir_stn / (r'low_d_evt_%s22_posdQ.png' % (stn)), bbox_inches='tight')
            plt.close('all')
                
                
                
            print('done calculating depth')
        
        
        
        df_all_events.dropna(axis=1, how='all').dropna(axis=0, how='all').to_csv(
            out_save_dir / r'all_events_stn_dQ.csv', sep=';')
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


if __name__ == '__main__':
    _save_log_ = False

    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    main()

    STOP = timeit.default_timer()  # Ending time
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
    