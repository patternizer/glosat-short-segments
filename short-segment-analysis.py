#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: short-segment-analysis.py
#-----------------------------------------------------------------------
# Version 0.3
# 27 April, 2021
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
# METHODS
#------------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):

    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6371* c # Radius of earth is 6371 km
    return km

def find_nearest(lat, lon, df):

    # Find nearest cell in array
    distances = df.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
    return df.loc[distances.idxmin(),:]
    
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
    
#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------

fontsize = 16
#station_code = '716121' # St Lawrence Valley
#station_code = '946720' # Adelaide
#station_code = '450050' # Hong Kong
station_code = '071560' # Paris/Montsouris

ma_smooth = 24
startyear_20CRv3 = '1806'
plot_schematic = False
plot_comparisons = True

#------------------------------------------------------------------------------
# LOAD & EXTRACT: GloSAT.p03 station absolute temperature timeseries
#------------------------------------------------------------------------------

ds = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')        
target = ds[ds['stationcode']==station_code].reset_index(drop=True)
target_lat = target['stationlat'].unique()[0]
target_lon = target['stationlon'].unique()[0]
if target_lon < 0: target_lon += 360.0
ts_station = np.array(target.groupby('year').mean().iloc[:,0:12]).ravel() 
if target['year'].iloc[0] > 1678:
    t_station = pd.date_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='M')          
else:
    t_station_xr = xr.cftime_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='A', calendar="gregorian")     
    t_station = [float(t_station_xr[i].year) for i in range(len(t_station_xr))]
    ts_station = pd.Series(ts_station, index=t_station)            
station_name = ds[ds['stationcode']==station_code]['stationname'].unique()[0].lower().replace(" ", "_")

#------------------------------------------------------------------------------
# LOAD & EXTRACT: GloSAT.p03 station temperature anomaly timeseries ('truth')
#------------------------------------------------------------------------------

da = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')        
target_anomaly = da[da['stationcode']==station_code].reset_index(drop=True)
ts_station_anomaly = np.array(target_anomaly.groupby('year').mean().iloc[:,0:12]).ravel() 
if target['year'].iloc[0] > 1678:
    t_station_anomaly = pd.date_range(start=str(target_anomaly['year'].iloc[0]), periods=len(ts_station_anomaly), freq='M')          
else:
    t_station_xr = xr.cftime_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='A', calendar="gregorian")     
    t_station = [float(t_station_xr[i].year) for i in range(len(t_station_xr))]
    ts_station_anomaly = pd.Series(ts_station_anomaly, index=t_station)            

#-----------------------------------------------------------------------------
# LOAD: 20CRv3 coords and unpack
#-----------------------------------------------------------------------------

de_absolute = xr.open_dataset('DATA/ensemble_mean_absolute.nc', decode_cf=True)
de_anomaly = xr.open_dataset('DATA/ensemble_mean_anomaly.nc', decode_cf=True)
lat = de_absolute.lat
lon = de_absolute.lon
X,Y = np.meshgrid(lon,lat)
N = len(lat)*len(lon)
x = X.reshape(N)
y = Y.reshape(N)
dlatlon = pd.DataFrame({'lon':x, 'lat':y}, index=range(N))    

#-----------------------------------------------------------------------------
# FIND: closest grid-cell (using Haversine distance) + calc distance to centre
#-----------------------------------------------------------------------------

pt = [target_lat,target_lon]  
query = find_nearest(pt[0],pt[1],dlatlon)
nearest_lat = query.lat
nearest_lon = query.lon
nearest_cell = query.name
nearest_lat_idx = np.where(de_absolute.lat==nearest_lat)[0][0] # --> 20CRv3 grid lat idx
nearest_lon_idx = np.where(de_absolute.lon==nearest_lon)[0][0] # --> 20CRv3 grid lon idx
delta_lon = (x[1]-x[0])/2
delta_lat = (y[1]-y[0])/2
distance_to_gridcell_centre = haversine(target_lat, target_lon, nearest_lat+delta_lat, nearest_lon+delta_lon)
    
#------------------------------------------------------------------------------
# LOAD: 20CRv3 absolute temperature timeseries
#------------------------------------------------------------------------------

#dr = 'DATA/absolutes_lon_286.4_lat_45.5_st_lawrence_valley/ensemble_mean_lon_286.4_lat_45.5.csv'

#nheader = 1
#f = open(dr, encoding="utf8", errors='ignore')
#lines = f.readlines()
#t_20CRv3 = []
#ts_20CRv3 = []
#for i in range(nheader,len(lines)):
#    words = lines[i].split(',')   
#    t = words[1]
#    ts = float(words[2])
#    t_20CRv3.append(t)
#    ts_20CRv3.append(ts)
#f.close()    
#ts_20CRv3 = np.array(ts_20CRv3)
#t_20CRv3 = pd.date_range(start=t_20CRv3[0], periods=len(ts_20CRv3), freq='M')          

#------------------------------------------------------------------------------
# LOAD: 20CRv3 anomaly timeseries
#------------------------------------------------------------------------------

#dr = 'DATA/anomalies_lon_286.4_lat_45.5_st_lawrence_valley/ensemble_mean_lon_286.4_lat_45.5.csv'

#nheader = 1
#f = open(dr, encoding="utf8", errors='ignore')
#lines = f.readlines()
#t_20CRv3_anomaly = []
#ts_20CRv3_anomaly = []
#for i in range(nheader,len(lines)):
#    words = lines[i].split(',')   
#    t = words[1]
#    ts = float(words[2])
#    t_20CRv3_anomaly.append(t)
#    ts_20CRv3_anomaly.append(ts)
#f.close()    
#ts_20CRv3_anomaly = np.array(ts_20CRv3_anomaly)
#t_20CRv3_anomaly = pd.date_range(start=t_20CRv3_anomaly[0], periods=len(ts_20CRv3_anomaly), freq='M')          

#-----------------------------------------------------------------------------
# LOAD: 20CRv3 absolute temperature timeseries
#-----------------------------------------------------------------------------

ts_20CRv3 = de_absolute.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].values.ravel()- 273.15
t = de_absolute.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].time
t_20CRv3 = pd.date_range(start=startyear_20CRv3, periods=len(t), freq='M')   

#------------------------------------------------------------------------------
# LOAD: 20CRv3 anomaly timeseries
#------------------------------------------------------------------------------

ts_20CRv3_anomaly = de_anomaly.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].values.ravel()
t = de_anomaly.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].time
t_20CRv3_anomaly = pd.date_range(start=startyear_20CRv3, periods=len(t), freq='M')   

#------------------------------------------------------------------------------
# CALCULATE: 20CRv3 normal
#------------------------------------------------------------------------------

baseline_mask = (t_20CRv3>=pd.to_datetime('1961-01-01')) & (t_20CRv3<pd.to_datetime('1991-01-01'))
t_baseline_20CRv3 = t_20CRv3[baseline_mask]
ts_baseline_20CRv3 = ts_20CRv3[baseline_mask]

normal_20CRv3 = np.mean(ts_baseline_20CRv3)
normal_20CRv3_monthly = []
for i in range(12):
    normal = np.mean( ts_baseline_20CRv3[t_baseline_20CRv3.month==i+1] )
    normal_20CRv3_monthly.append(normal)
#    normal_20CRv3_monthly.append(normal_20CRv3) # without monthly normals
# CHECK: np.mean(normal_20CRv3_monthly) = normal_20CRv3 # --> True

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
#N2 = 2400
N2 = len(t_station)
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
# CALCULATE: station normal & station anomaly
#------------------------------------------------------------------------------
# StationNorm = TimeMeanStation(period1) - TimeMean20CRv3(period1) + TimeMean20CRv3(1961-1990) 
#------------------------------------------------------------------------------

normal_station =  np.nanmean(df['short_segment_station']) - np.nanmean(df['short_segment_20CRv3']) + normal_20CRv3
normal_station_monthly = []
for i in range(12):
    normal = normal_20CRv3_monthly[i] + np.nanmean( df['short_segment_station'][df.index.month==i+1] ) - np.nanmean( df['short_segment_20CRv3'][df.index.month==i+1] )
    normal_station_monthly.append(normal)
#   normal_station_monthly.append(normal_station) # without monthly normals
# CHECK: np.mean(normal_station_monthly) = normal_station # --> True

# CONSTRUCT: normal vector (seasonal)

normal_vector = []
for i in range(len(df)):
    for j in range(12):
        if df.index[i].month==j+1:
            normal_vector.append(normal_station_monthly[j])
df['normal_station_monthly'] = normal_vector
df['short_segment_anomaly'] = df['short_segment_station'] - df['normal_station_monthly']

#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_schematic == True:

	#------------------------------------------------------------------------------
	# PLOT: short-segment method schematic
	#------------------------------------------------------------------------------

	figstr = 'short-segment-method.png'
	titlestr = 'Schematic of the short-segment method'

	fig,ax = plt.subplots(figsize=(15,10))
	#plt.plot(df.index, df['short_segment_station'].rolling(ma_smooth).mean()-3, color='green', lw=2, alpha=0.2)
	#plt.plot(df.index, df['short_segment_20CRv3'].rolling(ma_smooth).mean()-3, color='red', lw=2, alpha=0.2)
	plt.plot(short_segment[756:], np.tile(np.nanmean(df['short_segment_20CRv3']-3),len(short_segment[756:])), color='red', lw=3, ls='dotted', alpha=1.0, label='20CRv3: segment mean')
	plt.plot(t_baseline_20CRv3, np.tile(np.mean(normal_20CRv3_monthly),len(t_baseline_20CRv3)), color='red', lw=5, ls='-', alpha=1.0, label='20CRv3: 1961-1990 baseline')
	plt.plot(short_segment, np.tile(np.nanmean(df['short_segment_station']-3),len(short_segment)), color='green', lw=3, ls='dotted', alpha=1.0, label='GloSAT.p03: segment mean')
	plt.plot(t_baseline_20CRv3, np.tile(np.mean(normal_station_monthly),len(t_baseline_20CRv3)), color='green', lw=5, ls='-', alpha=1.0, label='GloSAT.p03: 1961-1990 baseline estimate')
	plt.plot(short_segment[int(N2/2)], np.nanmean(df['short_segment_20CRv3'])-3, color='black', marker='o', markerfacecolor='none', markersize=5, lw=1, ls='-', alpha=1.0)
	plt.plot(short_segment[int(N2/2)], np.nanmean(df['short_segment_station'])-3, color='black', marker='o', markerfacecolor='none', markersize=5, lw=1, ls='-', alpha=1.0)
	plt.plot(t_baseline_20CRv3[180], np.mean(normal_20CRv3_monthly), color='black', marker='o', markerfacecolor='none', markersize=5, lw=1, ls='-', alpha=1.0)
	plt.plot(t_baseline_20CRv3[180], np.mean(normal_station_monthly), color='black', marker='o', markerfacecolor='none', markersize=5, lw=1, ls='-', alpha=1.0)
	plt.plot( (short_segment[int(N2/2)], t_baseline_20CRv3[180]), (np.nanmean(df['short_segment_20CRv3'])-3, np.mean(normal_20CRv3_monthly)), color='black', lw=1, ls='--', alpha=1.0)
	plt.plot( (short_segment[int(N2/2)], t_baseline_20CRv3[180]), (np.nanmean(df['short_segment_station'])-3, np.mean(normal_station_monthly)), color='black', lw=1, ls='--', alpha=1.0)
	plt.annotate(s='', xy=(t_baseline_20CRv3[180], np.mean(normal_20CRv3_monthly)), xytext=(t_baseline_20CRv3[180], np.mean(normal_station_monthly)), arrowprops=dict(arrowstyle='<|-|>', ls='-', lw=2, color='blue'))
	plt.annotate(s='', xy=(short_segment[int(N2/2)], np.nanmean(df['short_segment_20CRv3'])-3), xytext=(short_segment[int(N2/2)], np.nanmean(df['short_segment_station'])-3), arrowprops=dict(arrowstyle='<|-|>', ls='-', lw=2, color='blue'))
	for i in range(12):        
	    if i<6:
    		if i==0:
    		    plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5, label='20CRv3: 1961-1990 baseline (seasonal)')
    		    plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5, label='GloSAT.p03: 1961-1990 estimate (seasonal)')
    		else:
    		    plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5)
    		    plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5)
    		plt.text(t_baseline_20CRv3[0], normal_20CRv3_monthly[i], str(i+1), color='red')
    		plt.text(t_baseline_20CRv3[0], normal_station_monthly[i], str(i+1), color='green')
	    else:
    		plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5)
    		plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5)
    		plt.text(t_baseline_20CRv3[-1], normal_20CRv3_monthly[i], str(i+1), color='red')
    		plt.text(t_baseline_20CRv3[-1], normal_station_monthly[i], str(i+1), color='green')
	plt.tick_params(labelsize=fontsize)   
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr, dpi=300)
	plt.close('all')

if plot_comparisons == True:

	# PLOT: raw absolutes: (GloSAT v 20CRv3)

	figstr = station_code + '-' + 'raw_absolutes_station_20CRv3.png'
	titlestr = station_code + ': ' + station_name
	#titlestr = 'NCDC/NOAA St Lawrence Valley: #716121'

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(t_station, ts_station, color='blue', alpha=0.1)
	plt.scatter(t_20CRv3, ts_20CRv3, color='red', alpha=0.1)
	plt.plot(t_station, pd.Series(ts_station).rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: '+str(ma_smooth)+'m MA')
	plt.plot(t_20CRv3, pd.Series(ts_20CRv3).rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: segment absolutes: (GloSAT v 20CRv3)

	figstr = station_code + '-' + 'segmemt_absolutes_station_20CRv3.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(df.index, df['short_segment_station'], color='blue', alpha=0.1)
	plt.scatter(df.index, df['short_segment_20CRv3'], color='red', alpha=0.1)
	plt.plot(df.index, df['short_segment_station'].rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: '+str(ma_smooth)+'m MA')
	plt.plot(df.index, df['short_segment_20CRv3'].rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: segment absolute difference: (GloSAT - 20CRv3)

	figstr = station_code + '-' + 'segment_absolutes_station_minus_20CRv3.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(df.index, df['short_segment_station']-df['short_segment_20CRv3'], color='black', alpha=0.1)
	plt.plot(df.index, df['short_segment_station'].rolling(ma_smooth).mean()-df['short_segment_20CRv3'].rolling(ma_smooth).mean(), color='black', lw=3, alpha=1.0, label='GloSATp03-20CRv3: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: raw anomalies: (GloSAT v 20CRv3)

	figstr = station_code + '-' + 'raw_anomalies_station_20CRv3.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(t_station_anomaly, ts_station_anomaly, color='blue', alpha=0.1)
	plt.scatter(t_20CRv3_anomaly, ts_20CRv3_anomaly, color='red', alpha=0.1)
	plt.plot(t_station_anomaly, pd.Series(ts_station_anomaly).rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSATp03: '+str(ma_smooth)+'m MA')
	plt.plot(t_20CRv3_anomaly, pd.Series(ts_20CRv3_anomaly).rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: segment anomalies: (GloSAT v 20CRv3 v estimate)

	figstr = station_code + '-' + 'segment_anomalies_station_20CRv3_estimate.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(df.index, df['short_segment_20CRv3_anomaly'], color='red', alpha=0.1)
	plt.scatter(df.index, df['short_segment_station_anomaly'], color='blue', alpha=0.1)
	plt.scatter(df.index, df['short_segment_anomaly'], color='green', ls='-', lw=0.5, alpha=0.1)
	plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='reanalysis (20CRv3): '+str(ma_smooth)+'m MA')
	plt.plot(df.index, df['short_segment_station_anomaly'].rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='truth (GloSATp03): '+str(ma_smooth)+'m MA')
	plt.plot(df.index, df['short_segment_anomaly'].rolling(ma_smooth).mean(), color='green', lw=3, alpha=1.0, label='short segment model: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: Gamma fit to historgrams of anomaly differences: (GloSAT - estimate) & (GloSAT - 20CRv3)

	y1 = df['short_segment_station_anomaly'] - df['short_segment_anomaly']
	y2 = df['short_segment_station_anomaly'] - df['short_segment_20CRv3_anomaly']
	z_map1, gamma_pdf1, param1 = gamma_fit(y1)
	z_map2, gamma_pdf2, param2 = gamma_fit(y2)
	bias1 = np.nanmean(y1)
	bias2 = np.nanmean(y2)

	figstr = station_code + '-' + 'segment_anomalies_station_minus_estimate.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	#plt.plot(z_map1, gamma_pdf1, color='green', lw=3, label=r'$\Gamma$ fit:'+
	#         r' ($\alpha=$'+str(np.round(param1[0],2))+','+
	#         r' loc='+str(np.round(param1[1],2))+','+
	#         r' scale='+str(np.round(param1[2],2))+'): GloSATp03-model')
	#plt.plot(z_map2, gamma_pdf2, color='red', lw=3, label=r'$\Gamma$ fit:'+
	#         r' ($\alpha=$'+str(np.round(param2[0],2))+','+
	#         r' loc='+str(np.round(param2[1],2))+','+
	#         r' scale='+str(np.round(param2[2],2))+'): GloSATp03-20CRv3')
	plt.plot(z_map1, gamma_pdf1, color='green', lw=3, label=r'$\Gamma$ fit: GloSATp03-model')
	plt.plot(z_map2, gamma_pdf2, color='red', lw=3, label=r'$\Gamma$ fit: GloSATp03-20CRv3')
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

	# PLOT: segment anomaly difference: (GloSAT - 20CRv3)

	figstr = station_code + '-' + 'segment_anomalies_station_minus_20CRv3.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(df.index, df['short_segment_station_anomaly'] - df['short_segment_20CRv3_anomaly'], color='black', alpha=0.1)
	plt.plot(df.index, df['short_segment_station_anomaly'].rolling(ma_smooth).mean() - df['short_segment_20CRv3_anomaly'].rolling(ma_smooth).mean(), color='black', lw=3, alpha=1.0, label='GloSATp03-20CRv3: '+str(ma_smooth)+'m MA')
	plt.tick_params(labelsize=fontsize)
	ax.xaxis.grid(True, which='major')        
	ax.yaxis.grid(True, which='major')        
	plt.legend(loc=2, ncol=1, fontsize=fontsize)
	plt.xlabel("Year", fontsize=fontsize)
	plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
	plt.title(titlestr, fontsize=fontsize)
	plt.savefig(figstr)
	plt.close('all')

	# PLOT: segment anomaly difference: (20CRv3 - estimate)

	figstr = station_code + '-' + 'segment_anomalies_20CRv3_minus_estimate.png'
	titlestr = station_code + ': ' + station_name

	fig,ax = plt.subplots(figsize=(15,10))
	plt.scatter(df.index, df['short_segment_20CRv3_anomaly'] - df['short_segment_anomaly'], color='black', alpha=0.1)
	plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(ma_smooth).mean() - df['short_segment_anomaly'].rolling(ma_smooth).mean(), color='black', lw=3, alpha=1.0, label='20CRv3-model: '+str(ma_smooth)+'m MA')
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

