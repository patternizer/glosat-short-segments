#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: short-segment-analysis.py
#-----------------------------------------------------------------------
# Version 0.1
# 23 April, 2021
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#-----------------------------------------------------------------------

# Dataframe libraries:
import numpy as np
import pandas as pd
import xarray as xr

# Datetime libraries:
from datetime import datetime
import nc_time_axis
import cftime

# Plotting libraries:
import matplotlib
#matplotlib.use('agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

# Maths libraries:
from scipy.interpolate import griddata
from scipy import spatial
from math import radians, cos, sin, asin, sqrt
import scipy
import scipy.stats as stats    

# OS libraries:
import os, sys
from  optparse import OptionParser
#import argparse

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    
#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 16

#------------------------------------------------------------------------------
# LOAD: station absolute temperature timeseries
#------------------------------------------------------------------------------

ds = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')        
station_code = '716121'
target = ds[ds['stationcode']==station_code].reset_index(drop=True)
ts_station = np.array(target.groupby('year').mean().iloc[:,0:12]).ravel() 
t_station = pd.date_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='M')          

#------------------------------------------------------------------------------
# LOAD: station anomaly temperature timeseries
#------------------------------------------------------------------------------

da = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')        
station_code = '716121'
target_anomaly = da[da['stationcode']==station_code].reset_index(drop=True)
ts_station_anomaly = np.array(target_anomaly.groupby('year').mean().iloc[:,0:12]).ravel() 
t_station_anomaly = pd.date_range(start=str(target_anomaly['year'].iloc[0]), periods=len(ts_station_anomaly), freq='M')          

#------------------------------------------------------------------------------
# LOAD: 20CRv3 absolute temperature timeseries
#------------------------------------------------------------------------------

dr = 'DATA/absolutes_lon_286.4_lat_45.5_st_lawrence_valley/ensemble_mean_lon_286.4_lat_45.5.csv'

nheader = 1
f = open(dr, encoding="utf8", errors='ignore')
lines = f.readlines()
t_20CRv3 = []
ts_20CRv3 = []
for i in range(nheader,len(lines)):
    words = lines[i].split(',')   
    t = words[1]
    ts = float(words[2])
    t_20CRv3.append(t)
    ts_20CRv3.append(ts)
f.close()    
ts_20CRv3 = np.array(ts_20CRv3)
t_20CRv3 = pd.date_range(start=t_20CRv3[0], periods=len(ts_20CRv3), freq='M')          

#------------------------------------------------------------------------------
# LOAD: 20CRv3 anomaly timeseries
#------------------------------------------------------------------------------

dr = 'DATA/anomalies_lon_286.4_lat_45.5_st_lawrence_valley/ensemble_mean_lon_286.4_lat_45.5.csv'

nheader = 1
f = open(dr, encoding="utf8", errors='ignore')
lines = f.readlines()
t_20CRv3_anomaly = []
ts_20CRv3_anomaly = []
for i in range(nheader,len(lines)):
    words = lines[i].split(',')   
    t = words[1]
    ts = float(words[2])
    t_20CRv3_anomaly.append(t)
    ts_20CRv3_anomaly.append(ts)
f.close()    
ts_20CRv3_anomaly = np.array(ts_20CRv3_anomaly)
t_20CRv3_anomaly = pd.date_range(start=t_20CRv3_anomaly[0], periods=len(ts_20CRv3_anomaly), freq='M')          

#------------------------------------------------------------------------------
# CALCULATE: 20CRv3 normal
#------------------------------------------------------------------------------

# StationNorm = TimeMeanStation(period1) + TimeMean20CRv3(1961-1990) - TimeMean20CRv3(period1)

baseline_mask = (t_20CRv3>=pd.to_datetime('1961-01-01')) & (t_20CRv3<pd.to_datetime('1991-01-01'))
t_baseline_20CRv3 = t_20CRv3[baseline_mask]
ts_baseline_20CRv3 = ts_20CRv3[baseline_mask]

normal_20CRv3 = np.mean(ts_baseline_20CRv3)
normal_20CRv3_monthly = []
for i in range(12):
    normal = np.mean( ts_baseline_20CRv3[t_baseline_20CRv3.month==i+1] )
    normal_20CRv3_monthly.append(normal)
#    normal_20CRv3_monthly.append(normal_20CRv3)
# check: np.mean(normal_20CRv3_monthly) = normal_20CRv3 # --> True

#------------------------------------------------------------------------------
# ALIGN: time axes
#------------------------------------------------------------------------------
    
if t_station[0]<=t_20CRv3[0]: 
    tmin = t_station[0]
else: 
    tmin = t_20CRv3[0]
if t_station[-1]>=t_20CRv3[-1]:
    tmax = t_station[-1]
else: 
    tmax = t_20CRv3[-1]
timerange = pd.date_range(start=tmin, end=tmax, freq='M')

#------------------------------------------------------------------------------
# SELECT: segment
#------------------------------------------------------------------------------

N1 = 0
N2 = 2400

short_segment = pd.date_range(start=t_station[N1], periods=(N2-N1), freq='M')          
short_segment_mask = (t_20CRv3>=pd.to_datetime(short_segment[0])) & (t_20CRv3<=pd.to_datetime(short_segment[-1]))
short_segment_station = ts_station[N1:N2]
short_segment_station_anomaly = ts_station_anomaly[N1:N2]
short_segment_20CRv3 = ts_20CRv3[short_segment_mask]
short_segment_20CRv3_anomaly = ts_20CRv3_anomaly[short_segment_mask]

# STORE: in dataframe to use time indexing

df0 = pd.DataFrame({},index=timerange)
d1 = pd.DataFrame({'short_segment_station':short_segment_station}, index=t_station[N1:N2])
d2 = pd.DataFrame({'short_segment_station_anomaly':short_segment_station_anomaly}, index=t_station[N1:N2])
d3 = pd.DataFrame({'short_segment_20CRv3':short_segment_20CRv3}, index=t_20CRv3[short_segment_mask])
d4 = pd.DataFrame({'short_segment_20CRv3_anomaly':short_segment_20CRv3_anomaly}, index=t_20CRv3_anomaly[short_segment_mask])
df1 = df0.join(d1, how='outer')
df2 = df1.join(d2, how='outer')
df3 = df2.join(d3, how='outer')
df4 = df3.join(d4, how='outer')
df = df4.copy()

#------------------------------------------------------------------------------
# CALCULATE: station normal
#------------------------------------------------------------------------------

# StationNorm = TimeMeanStation(period1) - TimeMean20CRv3(period1) + TimeMean20CRv3(1961-1990) 
# Add: difference between raw data term: TimeMean20CRv3 - TimeMeanStation(20CRv3)
# --> 
# StationNorm = TimeMeanStation(period1) - TimeMean20CRv3(period1) + TimeMean20CRv3(1961-1990) + TimeMean20CRv3 - TimeMeanStation(20CRv3)

#normal_station =  np.nanmean(df['short_segment_station']) - np.nanmean(df['short_segment_20CRv3']) + normal_20CRv3
#normal_station =  np.nanmean(df['short_segment_station']) - np.nanmean(df['short_segment_20CRv3']) + normal_20CRv3 + np.nanmean(df['short_segment_station']-df['short_segment_20CRv3'])
normal_station_monthly = []
for i in range(12):
    normal = normal_20CRv3_monthly[i] + np.nanmean( df['short_segment_station'][df.index.month==i+1] ) - np.nanmean( df['short_segment_20CRv3'][df.index.month==i+1] )
    normal_station_monthly.append(normal)
#    normal_station_monthly.append(normal_station) # no use of monthly normals
# check: np.mean(normal_20CRv3_monthly) = normal_20CRv3 # --> True

normal_vector = []
for i in range(len(df)):
    for j in range(12):
        if df.index[i].month==j+1:
            normal_vector.append(normal_station_monthly[j])
df['normal_station_monthly'] = normal_vector
#df['normal_station_monthly'] = normal_vector + np.nanmean(df['short_segment_station']-df['short_segment_20CRv3'])         

#------------------------------------------------------------------------------
# CALCULATE: station temperature anomaly
#------------------------------------------------------------------------------

df['short_segment_anomaly'] = df['short_segment_station'] - df['normal_station_monthly']

#------------------------------------------------------------------------------
# PLOT: comparison
#------------------------------------------------------------------------------

figstr = 'raw_absolutes_station_20CRv3.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(t_station, ts_station, color='blue', alpha=0.1)
plt.scatter(t_20CRv3, ts_20CRv3, color='red', alpha=0.1)
plt.plot(t_station, pd.Series(ts_station).rolling(24).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: 24m MA')
plt.plot(t_20CRv3, pd.Series(ts_20CRv3).rolling(24).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'segmemt_absolutes_station_20CRv3.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(df.index, df['short_segment_station'], color='blue', alpha=0.1)
plt.scatter(df.index, df['short_segment_20CRv3'], color='red', alpha=0.1)
plt.plot(df.index, df['short_segment_station'].rolling(24).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: 24m MA')
plt.plot(df.index, df['short_segment_20CRv3'].rolling(24).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'segment_absolutes_station_minus_20CRv3.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(df.index, df['short_segment_station']-df['short_segment_20CRv3'], color='black', alpha=0.1)
plt.plot(df.index, df['short_segment_station'].rolling(24).mean()-df['short_segment_20CRv3'].rolling(24).mean(), color='black', lw=3, alpha=1.0, label='GloSATp03-20CRv3: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'raw_anomalies_station_20CRv3.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(t_station_anomaly, ts_station_anomaly, color='blue', alpha=0.1)
plt.scatter(t_20CRv3_anomaly, ts_20CRv3_anomaly, color='red', alpha=0.1)
plt.plot(t_station_anomaly, pd.Series(ts_station_anomaly).rolling(24).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: 24m MA')
plt.plot(t_20CRv3_anomaly, pd.Series(ts_20CRv3_anomaly).rolling(24).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'segment_anomalies_station_20CRv3_estimate.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(df.index, df['short_segment_20CRv3_anomaly'], color='red', alpha=0.1)
plt.scatter(df.index, df['short_segment_station_anomaly'], color='blue', alpha=0.1)
plt.scatter(df.index, df['short_segment_anomaly'], color='green', ls='-', lw=0.5, alpha=0.1)
plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(24).mean(), color='red', lw=3, alpha=1.0, label='reanalysis (20CRv3): 24m MA')
plt.plot(df.index, df['short_segment_station_anomaly'].rolling(24).mean(), color='blue', lw=3, alpha=1.0, label='truth (GloSATp03): 24m MA')
plt.plot(df.index, df['short_segment_anomaly'].rolling(24).mean(), color='green', lw=3, alpha=1.0, label='short segment model: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

def gamma_fit(y):
    
    mask = np.isfinite(y)
    z = y[mask] 
    n = len(z) 
    z_map = np.linspace(z.min(), z.max(), n)
    disttype = 'gamma' 
    dist = getattr(scipy.stats, disttype)
    param = dist.fit(z) 
    gamma_pdf = dist.pdf(z_map, param[0], param[1], param[2])	
    
    return z_map, gamma_pdf, param

y1 = df['short_segment_station_anomaly'] - df['short_segment_anomaly']
y2 = df['short_segment_station_anomaly'] - df['short_segment_20CRv3_anomaly']
z_map1, gamma_pdf1, param1 = gamma_fit(y1)
z_map2, gamma_pdf2, param2 = gamma_fit(y2)
bias1 = np.nanmean(y1)
bias2 = np.nanmean(y2)

figstr = 'segment_anomalies_station_minus_estimate.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
#plt.plot(z_map1, gamma_pdf1, color='green', lw=3, label=r'$\Gamma$:'+
#         r' ($\alpha=$'+str(np.round(param1[0],2))+','+
#         r' loc='+str(np.round(param1[1],2))+','+
#         r' scale='+str(np.round(param1[2],2))+'): GloSATp03-model')
#plt.plot(z_map2, gamma_pdf2, color='red', lw=3, label=r'$\Gamma$:'+
#         r' ($\alpha=$'+str(np.round(param2[0],2))+','+
#         r' loc='+str(np.round(param2[1],2))+','+
#         r' scale='+str(np.round(param2[2],2))+'): GloSATp03-20CRv3')
plt.plot(z_map1, gamma_pdf1, color='green', lw=3, label=r'$\Gamma$: GloSATp03-model')
plt.plot(z_map2, gamma_pdf2, color='red', lw=3, label=r'$\Gamma$: GloSATp03-20CRv3')
plt.axvline(x=bias1, lw=1, ls='dashed', color='green', label='bias='+str(np.round(bias1,2)).zfill(2))
plt.axvline(x=bias2, lw=1, ls='dashed', color='red', label='bias='+str(np.round(bias2,2)).zfill(2))
axes = plt.gca()
[ymin,ymax] = axes.get_ylim()
ax.set_ylim(ymin,ymax)
ax.set_xlim(-5.0,5.0)
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.ylabel("KDE", fontsize=fontsize)
plt.xlabel("Anomaly difference, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'segment_anomalies_station_minus_20CRv3.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(df.index, df['short_segment_station_anomaly'] - df['short_segment_20CRv3_anomaly'], color='black', alpha=0.1)
plt.plot(df.index, df['short_segment_station_anomaly'].rolling(24).mean() - df['short_segment_20CRv3_anomaly'].rolling(24).mean(), color='black', lw=3, alpha=1.0, label='GloSATp03-20CRv3: 24m MA')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

figstr = 'segment_anomalies_20CRv3_minus_estimate.png'
titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

fig,ax = plt.subplots(figsize=(15,10))
plt.scatter(df.index, df['short_segment_20CRv3_anomaly'] - df['short_segment_anomaly'], color='black', alpha=0.1)
plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(24).mean() - df['short_segment_anomaly'].rolling(24).mean(), color='black', lw=3, alpha=1.0, label='20CRv3-model')
plt.tick_params(labelsize=fontsize)
ax.xaxis.grid(True, which='major')        
ax.yaxis.grid(True, which='major')        
plt.legend(loc=2, ncol=1, fontsize=fontsize)
plt.xlabel("Year", fontsize=fontsize)
plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
plt.title(titlestr, fontsize=fontsize)
plt.savefig(figstr)
plt.close('all')

#------------------------------------------------------------------------------
print('** END')

