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
    
    variables = ['precipitation', 'pet', 'temperature', 'discharge_vol', 
                 'humidity', 'longwave_rad','windspeed', 'discharge_spec']
    for var_to_test in variables:
    
        path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)
    
        #===========================================================================
        
        #===========================================================================
        data_hdf5 = HDF5(infile=path_to_data)
        catch_ids = data_hdf5.get_all_names()
        
        catch_coords = data_hdf5.get_coordinates(ids=catch_ids)
    
        df_coords = pd.DataFrame(index=catch_ids,
            data=catch_coords['easting'], columns=['X'])
        y_dwd_coords = catch_coords['northing']
        df_coords.loc[:, 'Y'] = y_dwd_coords
    
        
        plt.ioff()
        plt.figure(figsize=(6, 6), dpi=300)
        for catch_id in tqdm.tqdm(catch_ids):
            # print(catch_id)
            # break
            df_stn = data_hdf5.get_pandas_dataframe(catch_id)
            df_stn = df_stn.dropna(how='all')
            
            vals_q99 = df_stn.quantile(0.99)[0]
            
            df_stn_max = df_stn[df_stn > vals_q99].dropna()
            plt.scatter(df_stn_max.index, df_stn_max.values, alpha=0.1, facecolor='gray',edgecolor='darkred', s=15)
        
        plt.grid(alpha=0.5)
        plt.ylabel('%s' % var_to_test, fontsize=10)
        
        plt.xlabel('Time index')
        
        # plt.legend(loc=0)
        plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\00_Data_quality\scatter_%s.png" % (var_to_test), bbox_inches='tight')
        plt.close()
        # normalize by median
        
        # df_stn_norm_orig = df_stn# / df_stn.median()
        # df_stn[df_stn > 90] = np.nan

 




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
    