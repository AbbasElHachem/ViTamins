'''
Created on 4 Sep 2023

@author: hachem
'''
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
modulepath = r'X:\staff\elhachem\GitHub\ClimXtreme\WP2\a_analyse_dwd'
sys.path.append(modulepath)

from _a_01_read_hdf5 import HDF5

data_path = Path(r'X:\staff\elhachem\2023_09_01_ViTaMins')

main_dir = Path(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\8344e4f3-d2ea-44f5-8afa-86d2987543a9\data")


in_df = pd.read_csv(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\all_events_stn.csv",
                    sep=';', index_col=0, parse_dates=True)
sum_tstep = in_df.sum(axis=1)


plt.ioff()
fig = plt.figure(dpi=300, figsize=(6, 5))

plt.plot(sum_tstep.index, sum_tstep.values, linewidth=0.5, c='b')
plt.grid(alpha=0.5)
plt.ylabel('N days (1<d<=4) all catchments - 7 variables', fontsize=10)
fig.autofmt_xdate()
plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\sum_all_events_stn.png", bbox_inches='tight')
plt.close()

df_coords = pd.read_csv((main_dir / r"CAMELS_GB_topographic_attributes.csv"),
                        index_col=0, sep=',')

# def epsg wgs84 and utm32 for coordinates conversion
df_coords.columns
x_utm32, y_utm32 = (df_coords['gauge_easting'].values.ravel(),
                     df_coords['gauge_northing'].values.ravel())
df_coords.loc[:, 'X'] = x_utm32
df_coords.loc[:, 'Y'] = y_utm32


sum_tstep_high = sum_tstep[sum_tstep > 300]


in_df.loc[sum_tstep_high.index].dropna(axis=0, how='all')


variables = ['precipitation', 'pet', 'temperature', 'discharge_vol', 
                 'humidity', 'longwave_rad','windspeed']
for var_to_test in variables:
    
    path_to_data = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % var_to_test)  
    
    #===========================================================================
    data_hdf5 = HDF5(infile=path_to_data)
    catch_ids = data_hdf5.get_all_names()
    
    for ii, _idx in enumerate(sum_tstep_high.index):    
        _idx_b4 = _idx- pd.Timedelta(days=1)
        vals_b4 = data_hdf5.get_pandas_dataframe_for_date(catch_ids, _idx_b4)
        vals_ = data_hdf5.get_pandas_dataframe_for_date(catch_ids, _idx)
        # break
        plt.ioff()
    
        fig = plt.figure(dpi=300, figsize=(6, 5))
        plt.plot(range(len(vals_b4.columns)), vals_b4.values.ravel(), linewidth=0.5, c='r', label='t-1')
        plt.plot(range(len(vals_.columns)), vals_.values.ravel(), linewidth=0.5, c='b', label='t')
        plt.grid(alpha=0.5)
        plt.ylabel('%s' % var_to_test, fontsize=10)
        plt.legend(loc=0)
        fig.autofmt_xdate()
        plt.title('%s' % _idx)
        plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\time_%s_%d.png" % (var_to_test, ii), bbox_inches='tight')
        plt.close()
        ###
        fig = plt.figure(dpi=300, figsize=(4, 4))
    
        im=plt.scatter(df_coords.loc[vals_.columns.astype(int), 'X'],
                    df_coords.loc[vals_.columns.astype(int), 'Y'], s=vals_.values.ravel()+1, c=vals_.values.ravel(), cmap=plt.get_cmap('jet'), marker='o')
        plt.colorbar(im, shrink=0.75)
        plt.grid(alpha=0.5)
        plt.ylabel('%s' % var_to_test, fontsize=10)
        # fig.autofmt_xdate()
        plt.title('%s' % _idx)
        plt.axis('equal')
        plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\space_%s_%d.png" % (var_to_test, ii), bbox_inches='tight')
        plt.close()
    