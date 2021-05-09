#!/usr/bin/env python

#-----------------------------------------------------------------------
# PROGRAM: short-segment-analysis-africa.py
#-----------------------------------------------------------------------
# Version 0.2
# 30 April, 2021
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

fontsize = 18
ma_smooth = 12
startyear_20CRv3 = '1806'
plot_schematic = True
plot_comparisons = True
plot_continent_mean = True
plot_gamma_fit = True
use_k_start = False

#------------------------------------------------------------------------------
# LOAD: lists of continent stationcodes
#------------------------------------------------------------------------------

dt_continent_file = 'dt_africa.csv'
da_continent_file = 'da_africa.csv'
ds_continent_file = 'ds_africa.csv'
dt_continent_stationcodes = pd.read_csv(dt_continent_file).stationcode.astype(str)
da_continent_stationcodes = pd.read_csv(da_continent_file).stationcode.astype(str)
ds_continent_stationcodes = pd.read_csv(ds_continent_file).stationcode.astype(str)

dt_continent_stationcodes = pd.Series(['039172']) # Armagh Observatory
#dt_continent_stationcodes = pd.Series(['450050']) # Hong Kong
#dt_continent_stationcodes = pd.Series(['716121']) # St Lawrence Valley
#dt_continent_stationcodes = pd.Series(['946720']) # Adelaide

print('N(stations)='+str(len(dt_continent_stationcodes)))
print('M(normals)='+str(dt_continent_stationcodes.isin(da_continent_stationcodes).sum()))
print('M(short-segments)='+str(dt_continent_stationcodes.isin(ds_continent_stationcodes).sum()))

df_diff_20CRv3 = pd.DataFrame()    
df_diff_GloSAT = pd.DataFrame()    
df_diff_model = pd.DataFrame()    
df_20CRv3 = pd.DataFrame()
df_GloSAT = pd.DataFrame()
df_model = pd.DataFrame()

#------------------------------------------------------------------------------
# LOAD: GloSAT absolute temperaturs and anomalies dataframes
#------------------------------------------------------------------------------
    
dt = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')        
da = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')        

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

#------------------------------------------------------------------------------
# LOAD: 20CRv3 ensemble stats
#------------------------------------------------------------------------------

de_mean = xr.open_dataset('DATA/ensemble_anomalies_africa_mean.nc', decode_cf=True)
de_pctl_05 = xr.open_dataset('DATA/ensemble_anomalies_africa_pctl_05.nc', decode_cf=True)
de_pctl_50 = xr.open_dataset('DATA/ensemble_anomalies_africa_pctl_50.nc', decode_cf=True)
de_pctl_95 = xr.open_dataset('DATA/ensemble_anomalies_africa_pctl_95.nc', decode_cf=True)

ts_mean = de_mean.TMP2m[:,0,0,0]
t = de_mean.TMP2m[:,0,0,0].time
t_mean = pd.date_range(start=str(t[0].values)[0:4], periods=len(t), freq='M')   
de_mean = pd.DataFrame({'t':t_mean,'ts':ts_mean})

ts_pctl_05 = de_pctl_05.TMP2m[:,0,0,0]
t = de_pctl_05.TMP2m[:,0,0,0].time
t_pctl_05 = pd.date_range(start=str(t[0].values)[0:4], periods=len(t), freq='M')   
de_pctl_05 = pd.DataFrame({'t':t_pctl_05,'ts':ts_pctl_05})

ts_pctl_50 = de_pctl_50.TMP2m[:,0,0,0]
t = de_pctl_50.TMP2m[:,0,0,0].time
t_pctl_50 = pd.date_range(start=str(t[0].values)[0:4], periods=len(t), freq='M')   
de_pctl_50 = pd.DataFrame({'t':t_pctl_50,'ts':ts_pctl_50})

ts_pctl_95 = de_pctl_95.TMP2m[:,0,0,0]
t = de_pctl_95.TMP2m[:,0,0,0].time
t_pctl_95 = pd.date_range(start=str(t[0].values)[0:4], periods=len(t), freq='M')   
de_pctl_95 = pd.DataFrame({'t':t_pctl_95,'ts':ts_pctl_95})    

#------------------------------------------------------------------------------
# LOOP: over all stations in dt_continent_stationcodes
#------------------------------------------------------------------------------

n_stations = len(dt_continent_stationcodes)
if use_k_start == True:
    k_start = 740
else:
    k_start = 0    
for k in range(k_start, n_stations):

    station_code = dt_continent_stationcodes[k]
    if n_stations > 1:
        if (da_continent_stationcodes.isin([station_code]).any()) | (ds_continent_stationcodes.isin([station_code]).any()):
            if (da_continent_stationcodes.isin([station_code]).any()):
                print('k='+str(k)+': '+station_code+': normal')            
            else:
                print('k='+str(k)+': '+station_code+': short-segment')
        else:
            continue
        
    #------------------------------------------------------------------------------
    # EXTRACT: GloSAT.p03 station absolute temperature timeseries
    #------------------------------------------------------------------------------
    
    station_name = dt[dt['stationcode']==station_code]['stationname'].unique()[0].lower().replace(" ", "_")            
    target = dt[dt['stationcode']==station_code].reset_index(drop=True)    
    target_lat = target['stationlat'].unique()[0]
    target_lon = target['stationlon'].unique()[0]
    if target_lon < 0: 
        target_lon += 360.0
    ts_station = np.array(target.groupby('year').mean().iloc[:,0:12]).ravel() 
    if target['year'].iloc[0] > 1678:
        t_station = pd.date_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='M')          
    else:
        t_station_xr = xr.cftime_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='M', calendar="gregorian")     
        ts_station = pd.Series(ts_station, index=t_station_xr)         
        
        year = [t_station_xr[i].year for i in range(len(t_station_xr))]
        year_frac = []
        for i in range(len(t_station_xr)):
            if i%12 == 0:
                istart = i
                iend = istart+11   
                frac = np.cumsum([t_station_xr[istart+j].day for j in range(12)])
                year_frac += list(frac/frac[-1])
            else:
                i += 1
        year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
        t_station = np.array(year_decimal)
                
    #------------------------------------------------------------------------------
    # EXTRACT: GloSAT.p03 station temperature anomaly timeseries ('truth')
    #------------------------------------------------------------------------------
    
    target_anomaly = da[da['stationcode']==station_code].reset_index(drop=True)
    ts_station_anomaly = np.array(target_anomaly.groupby('year').mean().iloc[:,0:12]).ravel() 
    if target['year'].iloc[0] > 1678:
        t_station_anomaly = pd.date_range(start=str(target_anomaly['year'].iloc[0]), periods=len(ts_station_anomaly), freq='M')          
    else:
        t_station_anomaly_xr = xr.cftime_range(start=str(target['year'].iloc[0]), periods=len(ts_station), freq='M', calendar="gregorian")     
        ts_station_anomaly = pd.Series(ts_station_anomaly, index=t_station_anomaly_xr)       
         
        year = [t_station_anomaly_xr[i].year for i in range(len(t_station_anomaly_xr))]
        year_frac = []
        for i in range(len(t_station_anomaly_xr)):
            if i%12 == 0:
                istart = i
                iend = istart+11   
                frac = np.cumsum([t_station_anomaly_xr[istart+j].day for j in range(12)])
                year_frac += list(frac/frac[-1])
            else:
                i += 1
        year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
        t_station_anomaly = np.array(year_decimal)
        
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
        
    #-----------------------------------------------------------------------------
    # XARRAY: in-built function to do the same thing!
    # nearest_latlon = de_absolute["TMP2m"].sel(lon=target_lon, lat=target_lat, method="nearest")
    # nearest_lat = nearest_latlon.lat.values
    # nearest_lon = nearest_latlon.lon.values
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # LOAD: 20CRv3 absolute temperature timeseries
    #-----------------------------------------------------------------------------
    
    ts_20CRv3 = de_absolute.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].values.ravel() - 273.15
    t = de_absolute.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].time
    
    if target['year'].iloc[0] > 1678:
        t_20CRv3 = pd.date_range(start=startyear_20CRv3, periods=len(t), freq='M')   
    else:
        year = [t_20CRv3[i].year for i in range(len(t_20CRv3))]
        year_frac = []
        for i in range(len(t_20CRv3)):
            if i%12 == 0:
                istart = i
                iend = istart+11   
                frac = np.cumsum([t_20CRv3[istart+j].day for j in range(12)])
                year_frac += list(frac/frac[-1])
            else:
                i += 1
        year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
        t_20CRv3 = np.array(year_decimal)
        
    #------------------------------------------------------------------------------
    # LOAD: 20CRv3 anomaly timeseries
    #------------------------------------------------------------------------------
    
    ts_20CRv3_anomaly = de_anomaly.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].values.ravel()
    t = de_anomaly.TMP2m[:,0,nearest_lat_idx,nearest_lon_idx].time
    
    if target['year'].iloc[0] > 1678:
        t_20CRv3_anomaly = pd.date_range(start=startyear_20CRv3, periods=len(t), freq='M')   
    else:
        year = [t_20CRv3_anomaly[i].year for i in range(len(t_20CRv3_anomaly))]
        year_frac = []
        for i in range(len(t_20CRv3_anomaly)):
            if i%12 == 0:
                istart = i
                iend = istart+11   
                frac = np.cumsum([t_20CRv3_anomaly[istart+j].day for j in range(12)])
                year_frac += list(frac/frac[-1])
            else:
                i += 1
        year_decimal = [float(year[i])+year_frac[i] for i in range(len(year))]    
        t_20CRv3_anomaly = np.array(year_decimal)
    
    #------------------------------------------------------------------------------
    # CALCULATE: 20CRv3 normal
    #------------------------------------------------------------------------------
    
    if target['year'].iloc[0] > 1678:
        baseline_mask = (t_20CRv3>=pd.to_datetime('1961-01-01')) & (t_20CRv3<pd.to_datetime('1991-01-01'))
    else:
        baseline_mask = (t_20CRv3>=1961) & (t_20CRv3<1991)
    t_baseline_20CRv3 = t_20CRv3[baseline_mask]
    ts_baseline_20CRv3 = ts_20CRv3[baseline_mask]
    
    normal_20CRv3 = np.mean(ts_baseline_20CRv3)
    normal_20CRv3_monthly = []
    for i in range(12):
        if target['year'].iloc[0] > 1678:
            normal = np.mean( ts_baseline_20CRv3[t_baseline_20CRv3.month==i+1] )
        else:
            normal = np.mean( ts_baseline_20CRv3[t_baseline_20CRv3_xr.month==i+1] )
        normal_20CRv3_monthly.append(normal)
    #   normal_20CRv3_monthly.append(normal_20CRv3) # without monthly normals
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
    
    short_segment = pd.date_range(start=t_station[0], periods=len(t_station), freq='M')          
    short_segment_mask = (t_20CRv3>=pd.to_datetime(short_segment[0])) & (t_20CRv3<=pd.to_datetime(short_segment[-1]))
    short_segment_station = ts_station
    short_segment_station_anomaly = ts_station_anomaly
    short_segment_20CRv3 = ts_20CRv3[short_segment_mask]
    short_segment_20CRv3_anomaly = ts_20CRv3_anomaly[short_segment_mask]
    
    # STORE: in dataframe to use time indexing
    
    df0 = pd.DataFrame({},index=timerange)
    d1 = pd.DataFrame({'short_segment_station':short_segment_station}, index=t_station)
    d2 = pd.DataFrame({'short_segment_station_anomaly':short_segment_station_anomaly}, index=t_station)
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
    # SAVE: result
    #------------------------------------------------------------------------------

    if use_k_start == True:
        
        k == k_start
        df_diff_20CRv3 = pd.read_csv('df_diff_20CRv3.csv', index_col=0)
        df_diff_GloSAT = pd.read_csv('df_diff_GloSAT.csv', index_col=0)
        df_diff_model = pd.read_csv('df_diff_model.csv', index_col=0)
        df_20CRv3 = pd.read_csv('df_20CRv3.csv', index_col=0)
        df_GloSAT = pd.read_csv('df_GloSAT.csv', index_col=0)
        df_model = pd.read_csv('df_model.csv', index_col=0)               
        df_diff_20CRv3.index = pd.to_datetime(df_diff_20CRv3.index)
        df_diff_GloSAT.index = pd.to_datetime(df_diff_GloSAT.index)
        df_diff_model.index = pd.to_datetime(df_diff_model.index)
        df_20CRv3.index = pd.to_datetime(df_20CRv3.index)
        df_GloSAT.index = pd.to_datetime(df_GloSAT.index)
        df_model.index = pd.to_datetime(df_model.index)
    
    df_diff_20CRv3[station_code] = df['short_segment_20CRv3_anomaly'] - df['short_segment_station_anomaly']    
    df_diff_GloSAT[station_code] = df['short_segment_anomaly'] - df['short_segment_station_anomaly'] 
    df_diff_model[station_code] = df['short_segment_20CRv3_anomaly'] - df['short_segment_anomaly']
    df_20CRv3[station_code] = pd.Series(ts_20CRv3_anomaly, index=t_20CRv3_anomaly)
    df_GloSAT[station_code] = df['short_segment_station_anomaly']
    df_model[station_code] = df['short_segment_anomaly']
            
    if (k>9) & (k%10 == 0):    
        
        df_diff_20CRv3.to_csv('df_diff_20CRv3.csv')
        df_diff_GloSAT.to_csv('df_diff_GloSAT.csv')
        df_diff_model.to_csv('df_diff_model.csv')
        df_20CRv3.to_csv('df_20CRv3.csv')
        df_GloSAT.to_csv('df_GloSAT.csv')
        df_model.to_csv('df_model.csv')
        
        if plot_gamma_fit == True:
                    
            #------------------------------------------------------------------------------
            # PLOT: Gamma fit to historgrams of anomaly differences: (GloSAT - estimate) & (GloSAT - 20CRv3)
            #------------------------------------------------------------------------------
            
            diff_20CRv3_all = []
            diff_GloSAT_all = []
            diff_model_all = []
            for i in range(df_diff_20CRv3.shape[1]):
                diff_20CRv3 = df_diff_20CRv3.iloc[:,i]
                diff_GloSAT = df_diff_GloSAT.iloc[:,i]
                diff_model = df_diff_model.iloc[:,i]
                diff_20CRv3_all.append(diff_20CRv3)    
                diff_GloSAT_all.append(diff_GloSAT)    
                diff_model_all.append(diff_model)    
            diff_20CRv3_all = np.array(diff_20CRv3_all).flatten()
            diff_GloSAT_all = np.array(diff_GloSAT_all).flatten()        
            diff_model_all = np.array(diff_model_all).flatten()        
            mask_20CRv3 = np.isfinite(diff_20CRv3_all) 
            mask_GloSAT = np.isfinite(diff_GloSAT_all) 
            mask_model = np.isfinite(diff_model_all) 
            diff_20CRv3_all = diff_20CRv3_all[mask_20CRv3]
            diff_GloSAT_all = diff_GloSAT_all[mask_GloSAT]
            diff_model_all = diff_model_all[mask_model]
                        
            bias1 = np.nanmean(diff_20CRv3_all)       
            bias2 = np.nanmean(diff_GloSAT_all)        
            bias3 = np.nanmean(diff_model_all)        
            z_map1, gamma_pdf1, param1 = gamma_fit(diff_20CRv3_all)
            z_map2, gamma_pdf2, param2 = gamma_fit(diff_GloSAT_all)
            z_map3, gamma_pdf3, param3 = gamma_fit(diff_model_all)
                
            figstr = 'africa_segment_anomalies_model_bias'+'_'+str(k).zfill(3)+'.png'
            titlestr = 'Africa: 20CRv3 v GloSAT v short-segment model biases'    
                
            fig,ax = plt.subplots(figsize=(15,10))
            plt.plot(z_map1, gamma_pdf1, color='red', lw=3, label=r'$\Gamma$ fit: 20CRv3-GloSAT')
            plt.plot(z_map2, gamma_pdf2, color='blue', lw=3, label=r'$\Gamma$ fit: model-GloSAT')
            plt.plot(z_map3, gamma_pdf3, color='green', lw=3, label=r'$\Gamma$ fit: 20CRv3-model')
            plt.axvline(x=bias1, lw=1, ls='dashed', color='red', label='bias='+str(np.round(bias1,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_20CRv3_all.shape[0]))
            plt.axvline(x=bias2, lw=1, ls='dashed', color='blue', label='bias='+str(np.round(bias2,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_GloSAT_all.shape[0]))
            plt.axvline(x=bias3, lw=1, ls='dashed', color='green', label='bias='+str(np.round(bias3,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_model_all.shape[0]))
        #    plt.axvline(x=0)
            axes = plt.gca()
            [ymin,ymax] = axes.get_ylim()
            ax.set_ylim(ymin,ymax)
            plt.tick_params(labelsize=fontsize)
            ax.set_xlim(-8,8)
        #    ax.xaxis.grid(True, which='major')
        #    ax.yaxis.grid(True, which='major')
            plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
            plt.ylabel("KDE", fontsize=fontsize)
            plt.xlabel("Anomaly difference, $\mathrm{\degree}C$", fontsize=fontsize)
            plt.title(titlestr, fontsize=fontsize)
            plt.savefig(figstr)
            plt.close('all')
            
        if plot_continent_mean == True:
        
            ts_20CRv3_continent = df_20CRv3.mean(axis=1)
            t_20CRv3_continent = ts_20CRv3_continent.index
            ts_model_continent = df_model.mean(axis=1)
            t_model_continent = ts_model_continent.index
            ts_GloSAT_A_continent = df_GloSAT.mean(axis=1)
            t_GloSAT_A_continent = ts_GloSAT_A_continent.index
            n_GloSAT = np.sum(np.sum(np.isfinite(df_GloSAT))>0)
            n_model = np.sum(np.sum(np.isfinite(df_model))>0)
            
            figstr = 'africa_20crv3_model'+'_'+str(k).zfill(3)+'.png'
            titlestr = 'Africa: 20CRv3 v GloSAT v short-segment model estimate: N(additional stations)='+str(n_model-n_GloSAT)
            
            fig,ax = plt.subplots(figsize=(15,10))
            plt.plot(t_20CRv3_continent, ts_20CRv3_continent.rolling(60).mean(), color='red', lw=3, alpha=1.0, zorder=1, label='20CRv3: '+str(60)+'m MA')
            plt.plot(t_GloSAT_A_continent, ts_GloSAT_A_continent.rolling(60).mean(), color='blue', lw=3, alpha=1.0, zorder=1, label='GloSAT: '+str(60)+'m MA: N(stations)='+str(n_GloSAT))
            plt.plot(t_model_continent, ts_model_continent.rolling(60).mean(), color='green', lw=3, alpha=1.0, zorder=1, label='model: '+str(60)+'m MA: N(stations)='+str(n_model))
#            plt.plot(t_20CRv3_continent, ts_20CRv3_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='red', lw=3, alpha=1.0, zorder=1, label='20CRv3: '+str(60)+'m MA')
#            plt.plot(t_GloSAT_A_continent, ts_GloSAT_A_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='blue', lw=3, alpha=1.0, zorder=1, label='GloSAT: '+str(60)+'m MA: N(stations)='+str(n_GloSAT))
#            plt.plot(t_model_continent, ts_model_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='green', lw=3, alpha=1.0, zorder=1, label='model: '+str(60)+'m MA: N(stations)='+str(n_model))
            plt.step(de_pctl_50.t, de_pctl_50.ts.rolling(60).mean(), color='grey', lw=1, ls='--', label='20CRv3 (continent) median: '+str(60)+'m MA')
            plt.fill_between(de_pctl_05.t, de_pctl_05.ts.rolling(60).mean(), de_pctl_95.ts.rolling(60).mean(), alpha=0.1, color='black', lw=1, label='20CRv3 (continent) 5-95% c.i.: '+str(60)+'m MA')
            plt.tick_params(labelsize=fontsize)   
            plt.tick_params(labelsize=fontsize)        
            plt.axhline(y=0)
            plt.ylim(-2,2)
            plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
            fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)            
            plt.xlabel("Year", fontsize=fontsize)
            plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
            plt.title(titlestr, fontsize=fontsize)
            plt.savefig(figstr, dpi=300)
            plt.close('all')        
        
    #------------------------------------------------------------------------------
    # PLOTS
    #------------------------------------------------------------------------------
    
    if plot_schematic == True:
    
    	#------------------------------------------------------------------------------
    	# PLOT: short-segment method schematic
    	#------------------------------------------------------------------------------
    
    	figstr = station_code + '-' + 'short_segment_method.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(df.index, df['short_segment_20CRv3'].rolling(ma_smooth).mean(), color='red', ls='-', lw=2, alpha=0.2, label='20CRv3: segment '+str(ma_smooth)+'m MA')
    	plt.plot(short_segment, np.tile(np.nanmean(df['short_segment_20CRv3']),len(short_segment)), color='red', lw=3, ls='--', alpha=1.0, label='20CRv3: segment mean')
    	plt.plot(t_baseline_20CRv3, np.tile(np.mean(normal_20CRv3_monthly),len(t_baseline_20CRv3)), color='red', lw=5, ls='-', alpha=1.0, label='20CRv3: 1961-1990 baseline')
    	for i in range(12):        
    	    if i<6:
        		if i==0:
        		    plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5, label='20CRv3: 1961-1990 baseline (seasonal)')
        		else:
        		    plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5)
        		plt.text(t_baseline_20CRv3[0], normal_20CRv3_monthly[i], str(i+1), color='red')
    	    else:
        		plt.plot(t_baseline_20CRv3, np.tile(normal_20CRv3_monthly[i],len(t_baseline_20CRv3)), color='red', lw=1, alpha=0.5)
        		plt.text(t_baseline_20CRv3[-1], normal_20CRv3_monthly[i], str(i+1), color='red')
    
    	plt.plot(df.index, df['short_segment_station'].rolling(ma_smooth).mean(), color='blue', ls='-', lw=2, alpha=0.2, label='GloSAT: segment '+str(ma_smooth)+'m MA')    
    	plt.plot(short_segment, np.tile(np.nanmean(df['short_segment_station']),len(short_segment)), color='blue', lw=3, ls='--', alpha=1.0, label='GloSAT: segment mean')
    	plt.plot(t_baseline_20CRv3, np.tile(np.mean(normal_station_monthly),len(t_baseline_20CRv3)), color='green', lw=5, ls='-', alpha=1.0, label='GloSAT: 1961-1990 baseline estimate')
    	for i in range(12):        
    	    if i<6:
        		if i==0:
        		    plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5, label='GloSAT: 1961-1990 estimate (seasonal)')
        		else:
        		    plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5)
        		plt.text(t_baseline_20CRv3[0], normal_station_monthly[i], str(i+1), color='green')
    	    else:
        		plt.plot(t_baseline_20CRv3, np.tile(normal_station_monthly[i],len(t_baseline_20CRv3)), color='green', lw=1, alpha=0.5)
        		plt.text(t_baseline_20CRv3[-1], normal_station_monthly[i], str(i+1), color='green')
    
    	plt.plot(short_segment[  int(len(short_segment)/2) ], np.nanmean(df['short_segment_20CRv3']), color='black', marker='o', markerfacecolor='none', markersize=10, lw=1, ls='-', alpha=1.0)
    	plt.plot(short_segment[ int(len(short_segment)/2) ], np.nanmean(df['short_segment_station']), color='black', marker='o', markerfacecolor='none', markersize=10, lw=1, ls='-', alpha=1.0)
    	plt.plot(t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ], np.mean(normal_20CRv3_monthly), color='black', marker='o', markerfacecolor='none', markersize=10, lw=1, ls='-', alpha=1.0)
    	plt.plot(t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ], np.mean(normal_station_monthly), color='black', marker='o', markerfacecolor='none', markersize=10, lw=1, ls='-', alpha=1.0)
    	plt.plot( (short_segment[ int(len(short_segment)/2) ], t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ]), (np.nanmean(df['short_segment_20CRv3']), np.mean(normal_20CRv3_monthly)), color='black', lw=1, ls='--', alpha=1.0)
    	plt.plot( (short_segment[ int(len(short_segment)/2) ], t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ]), (np.nanmean(df['short_segment_station']), np.mean(normal_station_monthly)), color='black', lw=1, ls='--', alpha=1.0)
    #	plt.annotate(s='', xy=(t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ], np.mean(normal_20CRv3_monthly)), xytext=(t_baseline_20CRv3[ int(len(t_baseline_20CRv3)/2) ], np.mean(normal_station_monthly)), arrowprops=dict(arrowstyle='<|-|>', ls='-', lw=2, color='blue'))
    #	plt.annotate(s='', xy=(short_segment[ int(len(short_segment)/2) ], np.nanmean(df['short_segment_20CRv3'])), xytext=(short_segment[ int(len(short_segment)/2) ], np.nanmean(df['short_segment_station'])), arrowprops=dict(arrowstyle='<|-|>', ls='-', lw=2, color='blue'))
    
    	plt.tick_params(labelsize=fontsize)   
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)            
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr, dpi=300)
    	plt.close('all')
    
    if plot_comparisons == True:
    
    	# PLOT: raw absolutes: (GloSAT v 20CRv3)
    
    	figstr = station_code + '-' + 'raw_absolutes_station_20CRv3.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(t_20CRv3, ts_20CRv3, color='red', lw=1, ls='-', alpha=0.1)
    	plt.plot(t_station, ts_station, color='blue', lw=1, ls='-', alpha=0.1)
    	plt.plot(t_20CRv3, pd.Series(ts_20CRv3).rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
    	plt.plot(t_station, pd.Series(ts_station).rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSAT: '+str(ma_smooth)+'m MA')
    	plt.tick_params(labelsize=fontsize)
#    	plt.axhline(y=0)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)             
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')

    	# PLOT: raw anomalies: (GloSAT v 20CRv3)
    
    	figstr = station_code + '-' + 'raw_anomalies_station_20CRv3.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(t_20CRv3_anomaly, ts_20CRv3_anomaly, color='red', lw=1, ls='-', alpha=0.1)
    	plt.plot(t_20CRv3_anomaly, pd.Series(ts_20CRv3_anomaly).rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
    	if np.isfinite(ts_station_anomaly).sum() > 0:
            plt.plot(t_station_anomaly, ts_station_anomaly, color='blue', lw=1, ls='-', alpha=0.1)
            plt.plot(t_station_anomaly, pd.Series(ts_station_anomaly).rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSAT: '+str(ma_smooth)+'m MA')
    	plt.axhline(y=0)
    	plt.tick_params(labelsize=fontsize)
    	plt.ylim(-8,8)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')
        
    	# PLOT: segment absolutes: (GloSAT v 20CRv3)
    
    	figstr = station_code + '-' + 'segmemt_absolutes_station_20CRv3.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(df.index, df['short_segment_20CRv3'], color='red', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_station'], color='blue', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_20CRv3'].rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
    	plt.plot(df.index, df['short_segment_station'].rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSAT: '+str(ma_smooth)+'m MA')
    	plt.tick_params(labelsize=fontsize)
#    	plt.axhline(y=0)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')
    
    	# PLOT: segment absolute difference: (GloSAT - 20CRv3)
    
    	figstr = station_code + '-' + 'segment_absolutes_20CRv3_minus_station.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(df.index, df['short_segment_20CRv3']-df['short_segment_station'], color='black', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_20CRv3'].rolling(ma_smooth).mean()-df['short_segment_station'].rolling(ma_smooth).mean(), color='black', lw=3, alpha=1.0, label='20CRv3-GloSAT: '+str(ma_smooth)+'m MA')
    	plt.axhline(y=0)
    	plt.tick_params(labelsize=fontsize)
    	plt.ylim(-8,8)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Difference in absolute temperature, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')
        
    	# PLOT: segment anomalies: (GloSAT v 20CRv3 v estimate)
    
    	figstr = station_code + '-' + 'segment_anomalies_station_20CRv3_estimate.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(df.index, df['short_segment_20CRv3_anomaly'], color='red', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(ma_smooth).mean(), color='red', lw=3, alpha=1.0, label='20CRv3: '+str(ma_smooth)+'m MA')
    	if np.isfinite(df['short_segment_station_anomaly']).sum() > 0:
            plt.plot(df.index, df['short_segment_station_anomaly'], color='blue', lw=1, ls='-', alpha=0.1)
            plt.plot(df.index, df['short_segment_station_anomaly'].rolling(ma_smooth).mean(), color='blue', lw=3, alpha=1.0, label='GloSAT: '+str(ma_smooth)+'m MA')
    	plt.plot(df.index, df['short_segment_anomaly'], color='green', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_anomaly'].rolling(ma_smooth).mean(), color='green', lw=3, alpha=1.0, label='model: '+str(ma_smooth)+'m MA')
    	plt.axhline(y=0)
    	plt.tick_params(labelsize=fontsize)
    	plt.ylim(-8,8)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')
    
    	# PLOT: segment anomaly difference: (20CRv3 - estimate)
    
    	figstr = station_code + '-' + 'segment_anomalies_20CRv3_minus_estimate.png'
    	titlestr = station_code + ': ' + station_name
    
    	fig,ax = plt.subplots(figsize=(15,10))
    	plt.plot(df.index, df['short_segment_20CRv3_anomaly'] - df['short_segment_anomaly'], color='black', lw=1, ls='-', alpha=0.1)
    	plt.plot(df.index, df['short_segment_20CRv3_anomaly'].rolling(ma_smooth).mean() - df['short_segment_anomaly'].rolling(ma_smooth).mean(), color='black', lw=3, alpha=1.0, label='20CRv3-model: '+str(ma_smooth)+'m MA')
    	plt.axhline(y=0)
    	plt.tick_params(labelsize=fontsize)
    	plt.ylim(-8,8)
#    	ax.xaxis.grid(True, which='major')        
#    	ax.yaxis.grid(True, which='major')        
    	plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=1, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    	fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    	plt.xlabel("Year", fontsize=fontsize)
    	plt.ylabel("Difference in temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
    	plt.title(titlestr, fontsize=fontsize)
    	plt.savefig(figstr)
    	plt.close('all')    
        
if plot_gamma_fit == True:
            
    #------------------------------------------------------------------------------
    # PLOT: Gamma fit to historgrams of anomaly differences: (GloSAT - estimate) & (GloSAT - 20CRv3)
    #------------------------------------------------------------------------------
    
    diff_20CRv3_all = []
    diff_GloSAT_all = []
    diff_model_all = []
    for i in range(df_diff_20CRv3.shape[1]):
        diff_20CRv3 = df_diff_20CRv3.iloc[:,i]
        diff_GloSAT = df_diff_GloSAT.iloc[:,i]
        diff_model = df_diff_model.iloc[:,i]
        diff_20CRv3_all.append(diff_20CRv3)    
        diff_GloSAT_all.append(diff_GloSAT)    
        diff_model_all.append(diff_model)    
    diff_20CRv3_all = np.array(diff_20CRv3_all).flatten()
    diff_GloSAT_all = np.array(diff_GloSAT_all).flatten()        
    diff_model_all = np.array(diff_model_all).flatten()        
    mask_20CRv3 = np.isfinite(diff_20CRv3_all) 
    mask_GloSAT = np.isfinite(diff_GloSAT_all) 
    mask_model = np.isfinite(diff_model_all) 
    diff_20CRv3_all = diff_20CRv3_all[mask_20CRv3]
    diff_GloSAT_all = diff_GloSAT_all[mask_GloSAT]
    diff_model_all = diff_model_all[mask_model]
                
    bias1 = np.nanmean(diff_20CRv3_all)       
    bias2 = np.nanmean(diff_GloSAT_all)        
    bias3 = np.nanmean(diff_model_all)        
    z_map1, gamma_pdf1, param1 = gamma_fit(diff_20CRv3_all)
    z_map2, gamma_pdf2, param2 = gamma_fit(diff_GloSAT_all)
    z_map3, gamma_pdf3, param3 = gamma_fit(diff_model_all)
    
    if n_stations > 1:    
        figstr = 'africa_segment_anomalies_model_bias.png'
        titlestr = 'Africa: 20CRv3 v GloSAT v short-segment model biases'    
    else:
        figstr = station_code + '_' + 'segment_anomalies_model_bias.png'
        titlestr = station_code+': 20CRv3 v GloSAT v short-segment model biases'    
            
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot(z_map1, gamma_pdf1, color='red', lw=3, label=r'$\Gamma$ fit: 20CRv3-GloSAT')
    plt.plot(z_map2, gamma_pdf2, color='blue', lw=3, label=r'$\Gamma$ fit: model-GloSAT')
    plt.plot(z_map3, gamma_pdf3, color='green', lw=3, label=r'$\Gamma$ fit: 20CRv3-model')
    plt.axvline(x=bias1, lw=1, ls='dashed', color='red', label='bias='+str(np.round(bias1,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_20CRv3_all.shape[0]))
    plt.axvline(x=bias2, lw=1, ls='dashed', color='blue', label='bias='+str(np.round(bias2,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_GloSAT_all.shape[0]))
    plt.axvline(x=bias3, lw=1, ls='dashed', color='green', label='bias='+str(np.round(bias3,2)).zfill(2)+'$\mathrm{\degree}C$'+': N(pairs)='+str(diff_model_all.shape[0]))
#    plt.axvline(x=0)
    axes = plt.gca()
    [ymin,ymax] = axes.get_ylim()
    ax.set_ylim(ymin,ymax)
    plt.tick_params(labelsize=fontsize)
    ax.set_xlim(-8,8)
#    ax.xaxis.grid(True, which='major')
#    ax.yaxis.grid(True, which='major')
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)     
    plt.ylabel("KDE", fontsize=fontsize)
    plt.xlabel("Anomaly difference, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr)
    plt.close('all')
    
if plot_continent_mean == True:

    ts_20CRv3_continent = df_20CRv3.mean(axis=1)
    t_20CRv3_continent = ts_20CRv3_continent.index
    ts_model_continent = df_model.mean(axis=1)
    t_model_continent = ts_model_continent.index
    ts_GloSAT_A_continent = df_GloSAT.mean(axis=1)
    t_GloSAT_A_continent = ts_GloSAT_A_continent.index
    n_GloSAT = np.sum(np.sum(np.isfinite(df_GloSAT))>0)
    n_model = np.sum(np.sum(np.isfinite(df_model))>0)
    
    if n_stations > 1:        
        figstr = 'africa_20crv3_model.png'
        titlestr = 'Africa: 20CRv3 v GloSAT v short-segment model estimate: N(additional stations)='+str(n_model-n_GloSAT)
    else:
        figstr = station_code + '_' + '20crv3_model.png'
        titlestr = station_code + ': 20CRv3 v GloSAT v short-segment model estimate: N(additional stations)='+str(n_model-n_GloSAT)
    
    fig,ax = plt.subplots(figsize=(15,10))
    plt.plot(t_20CRv3_continent, ts_20CRv3_continent.rolling(60).mean(), color='red', lw=3, alpha=1.0, zorder=1, label='20CRv3: '+str(60)+'m MA')
    plt.plot(t_GloSAT_A_continent, ts_GloSAT_A_continent.rolling(60).mean(), color='blue', lw=3, alpha=1.0, zorder=1, label='GloSAT: '+str(60)+'m MA: N(stations)='+str(n_GloSAT))
    plt.plot(t_model_continent, ts_model_continent.rolling(60).mean(), color='green', lw=3, alpha=1.0, zorder=1, label='model: '+str(60)+'m MA: N(stations)='+str(n_model))
#    plt.plot(t_20CRv3_continent, ts_20CRv3_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='red', lw=3, alpha=1.0, zorder=1, label='20CRv3: '+str(60)+'m MA')
#    plt.plot(t_GloSAT_A_continent, ts_GloSAT_A_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='blue', lw=3, alpha=1.0, zorder=1, label='GloSAT: '+str(60)+'m MA: N(stations)='+str(n_GloSAT))
#    plt.plot(t_model_continent, ts_model_continent.rolling(60, win_type='triang').mean().rolling(60).mean(), color='green', lw=3, alpha=1.0, zorder=1, label='model: '+str(60)+'m MA: N(stations)='+str(n_model))
    plt.step(de_pctl_50.t, de_pctl_50.ts.rolling(60).mean(), color='grey', lw=1, ls='--', label='20CRv3 (continent) median: '+str(60)+'m MA')
    plt.fill_between(de_pctl_05.t, de_pctl_05.ts.rolling(60).mean(), de_pctl_95.ts.rolling(60).mean(), alpha=0.1, color='black', lw=1, label='20CRv3 (continent) 5-95% c.i.: '+str(60)+'m MA')    
    plt.tick_params(labelsize=fontsize)   
    plt.tick_params(labelsize=fontsize)        
    plt.axhline(y=0)
    plt.ylim(-2,2)
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.4), markerscale=1, ncol=2, facecolor='lightgrey', framealpha=1, fontsize=fontsize)    
    fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)            
    plt.xlabel("Year", fontsize=fontsize)
    plt.ylabel("Temperature anomaly, $\mathrm{\degree}C$", fontsize=fontsize)
    plt.title(titlestr, fontsize=fontsize)
    plt.savefig(figstr, dpi=300)
    plt.close('all')
        
#------------------------------------------------------------------------------
print('** END')

