# !/usr/bin/env python.
# -*- coding: utf-8 -*-

# TODO: ADD DOCS

"""
Name:    create HDF5 from DWD data
Purpose: create a single hdf5 data from ppt data

Created on: 2020-04-15

Parameters
----------
Direcotry with downloaded data and corresponding metadata

Returns
-------
A single HDF5 with all data combined
"""

__author__ = "Abbas El Hachem"
__institution__ = ('Institute for Modelling Hydraulic and Environmental '
                   'Systems (IWS), University of Stuttgart')
__copyright__ = ('Attribution 4.0 International (CC BY 4.0); see more '
                 'https://creativecommons.org/licenses/by/4.0/')
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"
__version__ = 0.1
__last_update__ = '15.04.2020'

# =============================================================================

import os
import glob
import sys

import pandas as pd
import numpy as np

import tables
import matplotlib.pyplot as plt
from pathlib import Path


# =============================================================================
in_freq = '1440min'


main_dir = Path(r"X:\staff\elhachem\2023_09_01_ViTaMins\Data\8344e4f3-d2ea-44f5-8afa-86d2987543a9\data")

data_dir = main_dir / r'timeseries'
os.chdir(data_dir)  # _old_recent

df_coords = pd.read_csv((main_dir / r"CAMELS_GB_topographic_attributes.csv"),
                        index_col=0, sep=',')

# def epsg wgs84 and utm32 for coordinates conversion

x_utm32, y_utm32 = (df_coords['gauge_easting'].values.ravel(),
                     df_coords['gauge_nortihng'].values.ravel())
df_coords.loc[:, 'X'] = x_utm32
df_coords.loc[:, 'Y'] = y_utm32


variables = ['precipitation', 'pet', 'temperature', 'discharge_spec', 'discharge_vol', 'peti',
             'humidity','shortwave_rad','longwave_rad','windspeed']



# get all .csv file
df_stns_all = glob.glob('*.csv')
# df_stns_all[0]
# craete out dates


start_dt = '1970-01-01'  # % start_year
end_dt = '2015-09-30'  # % end_year

dates = pd.date_range(start=start_dt, end=end_dt, freq=in_freq)

stns = [df.split('1970')[0].split('_')[-1] for df in df_stns_all if len(df.split('_')[0]) > 0]
cmn_stns = df_coords.index.intersection(stns)

# if data_radar_area:
#     cmn_stns = dwd_stns_in_radar

df_stns = [df for df in df_stns_all if df.split('_')[0] in cmn_stns]

df_coords = df_coords.loc[cmn_stns, :]
nstats = len(cmn_stns)



for _var in variables:
    
    blank_df = pd.DataFrame(index=dates,
                        data=np.full(shape=dates.shape, fill_value=-9))

    hdf5_path = os.path.join(r'X:\staff\elhachem\2022_PWS_NRW\dwd',
                             'CAmels_UK_%s_%s_%s_%s.h5'
                             % (in_freq, '1970', '2015', _var))

    

    if not os.path.isfile(hdf5_path):
        print('creating Hdf5 ', nstats)
        
        # number of maximum timesteps
        nts_max = dates.shape[0]

    hf = tables.open_file(hdf5_path, 'w', filters=tables.Filters(complevel=6))

    # timestamps
    hf.create_group(where=hf.root,
                    name='timestamps',
                    title='Timestamps of respective Aggregation as Start Time')

    hf.create_carray(where=hf.root.timestamps,
                     name='isoformat',
                     atom=tables.StringAtom(19),
                     shape=(nts_max,),
                     chunkshape=(10000,),
                     title='Strings of Timestamps in Isoformat')

    hf.create_carray(where=hf.root.timestamps,
                     name='year',
                     atom=tables.IntAtom(),
                     shape=(nts_max,),
                     chunkshape=(10000,),
                     title='Yearly Timestamps')

    hf.create_carray(where=hf.root.timestamps,
                     name='month',
                     atom=tables.IntAtom(),
                     shape=(nts_max,),
                     chunkshape=(10000,),
                     title='Monthly Timestamps')

    hf.create_carray(where=hf.root.timestamps,
                     name='day',
                     atom=tables.IntAtom(),
                     shape=(nts_max,),
                     chunkshape=(10000,),
                     title='Daily Timestamps')

    hf.create_carray(where=hf.root.timestamps,
                     name='yday',
                     atom=tables.IntAtom(),
                     shape=(nts_max,),
                     chunkshape=(10000,),
                     title='Yearday Timestamps')

    hf.create_carray(where=hf.root.timestamps,
                     name='start_idx',
                     atom=tables.Time64Atom(),
                     shape=(nstats,),
                     title='First Index of the Timeseries')
    hf.create_carray(where=hf.root.timestamps,
                     name='end_idx',
                     atom=tables.Time64Atom(),
                     shape=(nstats,),
                     title='Last Index of the Timeseries')

    # data
    hf.create_carray(where=hf.root,
                     name='data',
                     atom=tables.FloatAtom(dflt=np.nan),
                     shape=(nts_max, nstats),
                     chunkshape=(10000, 1),
                     title='Camels UK %s' % in_freq)

    # coordinates
    hf.create_group(where=hf.root,
                    name='coord',
                    title='WGS84-UTM32')
    hf.create_carray(where=hf.root.coord,
                     name='lon',
                     atom=tables.FloatAtom(dflt=np.nan),
                     shape=(nstats,),
                     title='Longitude')
    hf.create_carray(where=hf.root.coord,
                     name='lat',
                     atom=tables.FloatAtom(dflt=np.nan),
                     shape=(nstats,),
                     title='Latitude')

    hf.create_carray(where=hf.root.coord,
                     name='easting',
                     atom=tables.FloatAtom(dflt=np.nan),
                     shape=(nstats,),
                     title='Easting')
    hf.create_carray(where=hf.root.coord,
                     name='northing',
                     atom=tables.FloatAtom(dflt=np.nan),
                     shape=(nstats,),
                     title='Northing')

    # metadata
    hf.create_carray(where=hf.root,
                     name='name',
                     atom=tables.StringAtom(50),
                     shape=(nstats,),
                     title='Name of Gauge')

    hf.create_carray(where=hf.root,
                     name='id',
                     atom=tables.StringAtom(10),
                     shape=(nstats,),
                     title='Identification Number of Gauge')

    # convert timestamp to isoformat
    ts_iso = []
    ts_year = []
    ts_month = []
    ts_day = []
    ts_yday = []

    for ii in range(dates.shape[0]):
        ts = dates[ii]
        ts_iso.append(ts.isoformat())
        # get timestamp years
        ts_year.append(ts.year)
        # get timestamp months
        ts_month.append(ts.month)
        # get timestamp days
        ts_day.append(ts.day)
        # get timestamp year days
        ts_yday.append(ts.timetuple().tm_yday)
        # hours
    # fill hf5 with predefined stamps
    hf.root.timestamps.isoformat[:] = ts_iso[:]
    hf.root.timestamps.year[:] = ts_year[:]
    hf.root.timestamps.month[:] = ts_month[:]
    hf.root.timestamps.day[:] = ts_day[:]
    hf.root.timestamps.yday[:] = ts_yday[:]

    print('Done Creating hdf5')
    hf.close()

#==============================================================================
#
#==============================================================================
print('write station data')
# write station data
hf = tables.open_file(hdf5_path, 'r+')

stns = [_f.split('_')[0] for _f in df_stns]

for i_idx, df_stn in enumerate(df_stns):
    # break

    stn = df_stn.split('_')[0]

    if len(stn) > 0:
        print('{} / {}'.format(i_idx + 1, len(df_stns)))
        print(stn, '-', df_stn)
        df_in = pd.read_csv(df_stn, sep=';', index_col=0,
                            engine='c', low_memory=False)
        df_in.index = pd.to_datetime(df_in.index,
                                     format='%Y-%m-%d %H:%M:%S')

        df_in = df_in[~df_in.index.duplicated()]

        # df_in = select_df_within_period(
        # df=df_in, start=dates[0], end=dates[-1])
        # if df_for_season:
        # df_in = select_season(df_in, summer_month)

        if df_in.size > 0:

            blank_df = pd.DataFrame(index=dates,
                                    data=np.full(
                                        shape=dates.shape, fill_value=-9),
                                    columns=[stn])
            print(df_in)
            print('Dataframe has %d data ' %
                  df_in.size, df_in.sum())
            print(df_in.index[0], df_in.index[-1])
            try:
                cmn_idx = blank_df.index.intersection(df_in.index)
                # index.intersection(df_in.index).fillna(-9)
                blank_df.loc[cmn_idx, stn] = df_in.values
                hf.root.data[:, i_idx] = blank_df.values.flatten()

            except Exception as msg:
                print(i_idx, msg)
        else:
            print('Not enough data')
            df_in = pd.DataFrame(index=blank_df.index,
                                 data=np.full(
                                     shape=blank_df.index.shape, fill_value=-9))
        try:
            start_idx = df_in.index[0]
        except Exception as msg:
            print('Error with start index', msg)
        # plt.ioff()
        # plt.figure()
        #
        # plt.plot(blank_df.index,
        #          blank_df.values)
        # plt.savefig(os.path.join(data_dir, '%s.png' % stn))
        # plt.close()
        end_idx = df_in.index[-1]
        hf.root.coord.lon[i_idx] = df_coords['geoLaenge'].loc[stn]
        hf.root.coord.lat[i_idx] = df_coords['geoBreite'].loc[stn]
        # df_coords['Stationshoehe'].loc[stn]
        hf.root.coord.z[i_idx] = np.nan
        # x_utm32, y_utm32 = convert_coords_fr_wgs84_to_utm32_(
        #     epgs_initial_str=wgs84, epsg_final_str=utm32,  # utm32,
        #     first_coord=df_coords['geoLaenge'].loc[stn],
        #     second_coord=df_coords['geoBreite'].loc[stn])

        # df_coords['X'].loc[stn]
        hf.root.coord.easting[i_idx] = df_coords['X'].loc[stn]
        hf.root.coord.northing[i_idx] = df_coords['Y'].loc[stn]

        hf.root.timestamps.start_idx[i_idx] = np.datetime64(start_idx)
        hf.root.timestamps.end_idx[i_idx] = np.datetime64(end_idx)
        hf.root.name[i_idx] = np.string_(stn)

    else:
        print('check stn')
print('Done filling hdf5')
hf.close()
