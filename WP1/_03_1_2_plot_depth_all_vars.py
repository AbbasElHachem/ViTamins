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


in_df = pd.read_csv(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\all_events_stn_dQ.csv",
                    sep=';', index_col=0, parse_dates=True)
sum_tstep = in_df.sum(axis=1)


plt.ioff()
fig = plt.figure(dpi=300, figsize=(6, 5))

plt.plot(sum_tstep.index, sum_tstep.values, linewidth=0.5, c='b')
plt.grid(alpha=0.5)
plt.ylabel('N days (1<d<=4) all catchments - 7 variables', fontsize=10)
fig.autofmt_xdate()
plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\sum_all_events_stn_dQ.png", bbox_inches='tight')
plt.close()

df_coords = pd.read_csv((main_dir / r"CAMELS_GB_topographic_attributes.csv"),
                        index_col=0, sep=',')

# def epsg wgs84 and utm32 for coordinates conversion
df_coords.columns
x_utm32, y_utm32 = (df_coords['gauge_easting'].values.ravel(),
                     df_coords['gauge_northing'].values.ravel())
df_coords.loc[:, 'X'] = x_utm32
df_coords.loc[:, 'Y'] = y_utm32


sum_tstep_high = sum_tstep.sort_values()[-10:]


in_df.loc[sum_tstep_high.index].dropna(axis=0, how='all')


variables = ['precipitation', 'pet', 'temperature', 'discharge_vol', 'discharge_speci',
                 'humidity', 'longwave_rad','windspeed']

# variables = ['precipitation', 'discharge_spec']
# for var_to_test in variables:
    
path_to_data_pcp = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'precipitation')  
path_to_data_q = data_path / (r'Data/CAMELS_GB_1440min_1970_2015_%s.h5' % 'discharge_spec')  
    
#===========================================================================
data_hdf5_pcp = HDF5(infile=path_to_data_pcp)
catch_ids_pcp = data_hdf5_pcp.get_all_names()

data_hdf5_q = HDF5(infile=path_to_data_q)
catch_ids_q = data_hdf5_q.get_all_names()

for ii, _idx in enumerate(sum_tstep_high.index):    
    _idx_b4 = _idx- pd.Timedelta(days=1)
    
    vals_b4_p = data_hdf5_pcp.get_pandas_dataframe_for_date(catch_ids_pcp, _idx_b4)
    vals_p = data_hdf5_pcp.get_pandas_dataframe_for_date(catch_ids_pcp, _idx)
    
    vals_b4_q = data_hdf5_q.get_pandas_dataframe_for_date(catch_ids_pcp, _idx_b4)
    vals_q = data_hdf5_q.get_pandas_dataframe_for_date(catch_ids_pcp, _idx)
    
    # break
    plt.ioff()

    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(8, 4))
    ax1.plot(range(len(vals_b4_p.columns)), vals_b4_p.values.ravel(), linewidth=0.5, c='r', label='t-1')
    ax1.plot(range(len(vals_p.columns)), vals_p.values.ravel(), linewidth=0.5, c='b', label='t')
    ax1.grid(alpha=0.5)
    ax1.set_ylabel('%s' % 'Precipitation', fontsize=10)
    ax1.legend(loc=0)
    
    ax2.plot(range(len(vals_b4_q.columns)), vals_b4_q.values.ravel(), linewidth=0.5, c='r', label='t-1')
    ax2.plot(range(len(vals_q.columns)), vals_q.values.ravel(), linewidth=0.5, c='b', label='t')
    ax2.grid(alpha=0.5)
    ax2.set_ylabel('%s' % 'Q specific mm/d', fontsize=10)
    # ax2.legend(loc=0)
    
    fig.autofmt_xdate()
    ax1.set_title('%s' % _idx)
    plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\time_%s_%d.png" % ('pcp', ii), bbox_inches='tight')
    plt.close()
    ###
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=300, figsize=(8, 4))

    im=ax1.scatter(df_coords.loc[vals_p.columns.astype(int), 'X'],
                df_coords.loc[vals_p.columns.astype(int), 'Y'], s=vals_p.values.ravel()+1, c=vals_p.values.ravel(), cmap=plt.get_cmap('viridis'), marker='X')
    fig.colorbar(im, ax=ax1, shrink=0.75)
    ax1.grid(alpha=0.5)
    ax1.set_ylabel('%s' % 'Precipitation', fontsize=10)
    # fig.autofmt_xdate()
    ax1.set_title('%s' % _idx)
    
    im=ax2.scatter(df_coords.loc[vals_q.columns.astype(int), 'X'],
                df_coords.loc[vals_q.columns.astype(int), 'Y'], s=vals_q.values.ravel()+1, c=vals_q.values.ravel(), cmap=plt.get_cmap('viridis'), marker='X')
    fig.colorbar(im, ax=ax2, shrink=0.75)
    ax2.grid(alpha=0.5)
    ax2.set_ylabel('%s' % 'Q specific mm/d', fontsize=10)
    # fig.autofmt_xdate()
    ax2.set_title('%s' % _idx)
    
    
    plt.axis('equal')
    plt.savefig(r"X:\staff\elhachem\2023_09_01_ViTaMins\Results\02_1catch_all_var\space_%d.png" % (ii), bbox_inches='tight')
    plt.close()
    