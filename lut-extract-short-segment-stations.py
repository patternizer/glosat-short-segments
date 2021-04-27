#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: lut-extract-short-segment-stations.py
#------------------------------------------------------------------------------
# Version 0.2
# 25 April, 2021
# Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
# michael DOT a DOT taylor AT uea DOT ac DOT uk
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
# Dataframe libraries:
import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
from scipy import spatial
import itertools
import pandas as pd
import xarray as xr
#import klib
import pickle
from datetime import datetime
import nc_time_axis
import cftime
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt; plt.close('all')
import matplotlib.cm as cm
from matplotlib import colors as mcol
from matplotlib.cm import ScalarMappable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.collections import PolyCollection
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import cmocean
# Mapping libraries:
import cartopy
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cf
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# GeoPandas libraries:
import geopandas
from shapely.geometry import *
# OS libraries:
import os
import os.path
from pathlib import Path
import sys
import subprocess
from subprocess import Popen
import time
# Maths libraries:
from scipy.interpolate import griddata
from scipy import spatial
from math import radians, cos, sin, asin, sqrt

# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS: 
#------------------------------------------------------------------------------

fontsize = 20
plot_station_locations = False
plot_climgen_categories = False
plot_continent_stations = False
extract_stations = True
shade_continent = False
show_gridlines = True

# Map settings

# get natural earth names (http://www.naturalearthdata.com/)

projection = 'robinson'        
cmap = 'magma'
resolution = '10m'
category = 'cultural'
name = 'admin_0_countries'
shapefilename = shapereader.natural_earth(resolution, category, name)   

# GeoPandas Continents

continent = 'Africa'
# continent = 'Antarctica'
# continent = 'Asia'
# continent = 'Europe'
# continent = 'North America'
# continent = 'Oceania'
# continent = 'South America'

# cnt.spec Continents

category_keys = {
    'Antarctica':0,
    'Europe':1, 
    'exUSSR':2, 
    'Middle East':3,
    'Asia':4,
    'Africa':5,
    'North America':6,
    'Central America':7,
    'South America':8,
    'Oceania':9,
    'Sea':10}

# category_keys[continent]

#-----------------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):

    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    # Radius of earth is 6371 km
    km = 6371* c 
    return km

def find_nearest(lat, lon, df):

    # Find nearest cell in array
    distances = df.apply(lambda row: haversine(lat, lon, row['lat'], row['lon']), axis=1)
    return df.loc[distances.idxmin(),:]

def point_inside_shape(point, shape):
        
    # point: type Point
    # shape: type Polygon  
    pt = geopandas.GeoDataFrame(geometry=[point], index=['A'])    
    return(pt.within(shape).iloc[0])

#------------------------------------------------------------------------------
# LOAD: absolute temperatures and normals
#------------------------------------------------------------------------------

df_temp_in = pd.read_pickle('DATA/df_temp.pkl', compression='bz2')
df_anom_in = pd.read_pickle('DATA/df_anom.pkl', compression='bz2')
df_normals = pd.read_pickle('DATA/df_normals.pkl', compression='bz2')
#df_temp = df_temp_in[df_temp_in['stationcode'].isin(df_normals[df_normals['sourcecode']>1]['stationcode']) == False]
df_temp = df_temp_in.copy()
df_anom = df_anom_in[df_anom_in['stationcode'].isin(df_normals[df_normals['sourcecode']>1]['stationcode'])]

#------------------------------------------------------------------------------
# LOAD: continent specification file from ClimGen and extract lat,lon,cat vectors
#------------------------------------------------------------------------------

climgen_file = 'DATA/cnt.spec'
nheader = 363
nfooter = 9
f = open(climgen_file)
lines = f.readlines()
categories = []
for i in range(nheader,len(lines)-nfooter):    
    words = lines[i].split()   
    categories.append(words)        
f.close()    

category_vector = np.array([int(categories[i][0]) for i in range(len(categories))])
#category_array = category_vector.reshape(720, 360)

dc_lon = []
dc_lat = []
dc_cat = []
lon_step = 0.25
lat_step = 0.25
for i in range(720):    
    for j in range(360):
        dc_lon.append(lon_step+i*0.5-180)
        dc_lat.append(lat_step+j*0.5-90.0)
        cat = category_vector[i*360+j]
        if cat == -999:
            dc_cat.append(10)        
        else:
            dc_cat.append(cat)        
        
N = len(dc_cat)        
dc = pd.DataFrame({'lon':dc_lon, 'lat':dc_lat, 'cat':dc_cat}, index=range(N))
# dc[dc['lat']<-60.0] = 0

#-----------------------------------------------------------------------------
# CONSTRUCT: Pandas latloncat dataframe for selected continent
#-----------------------------------------------------------------------------

dc_lon = dc[dc['cat']==category_keys[continent]]['lon'].reset_index(drop=True)
dc_lat = dc[dc['cat']==category_keys[continent]]['lat'].reset_index(drop=True)
dc_cat = dc[dc['cat']==category_keys[continent]]['cat'].reset_index(drop=True)

N = len(dc_cat)
dlatloncat = pd.DataFrame({'lon':dc_lon, 'lat':dc_lat}, index=range(N))

#-----------------------------------------------------------------------------
# FIND: CLOSEST GRIDCELL (using Haversine distance) + assign cat for stations 
#-----------------------------------------------------------------------------
    
dt = df_temp.copy() # all stations
da = df_anom.copy() # anomalies

dt_lon = dt.dropna().groupby('stationcode').mean()['stationlon']
dt_lat = dt.dropna().groupby('stationcode').mean()['stationlat']
da_lon = da.dropna().groupby('stationcode').mean()['stationlon']
da_lat = da.dropna().groupby('stationcode').mean()['stationlat']    
ds_lon = da.dropna().groupby('stationcode').mean()['stationlon']
ds_lat = da.dropna().groupby('stationcode').mean()['stationlat']
dt_lonlat = pd.DataFrame({'lon':dt_lon,'lat':dt_lat})
da_lonlat = pd.DataFrame({'lon':da_lon,'lat':da_lat})
ds_lonlat = pd.concat([dt_lonlat,da_lonlat]).drop_duplicates(keep=False)

#dt_query = [ find_nearest(dt_lat[i], dt_lon[i], dlatloncat) for i in range(len(dt_lon)) ]
#dt_nearest_lon = [ query[i][0] for i in range(len(query)) ]
#dt_nearest_lat = [ query[i][1] for i in range(len(query)) ]
#dt_nearest_cat = [ dc_cat.unique().astype(int)[0] for i in range(len(query)) ]

if extract_stations == True:

    #------------------------------------------------------------------------------
    # EXTRACT: stations in continent and write absolutes, anomalies and short-segment station lists to CSV
    #------------------------------------------------------------------------------

    print('extract_stations ...')
                
    # Use GeoPandas to extract continent polygon
     
    gp = geopandas.read_file(shapefilename) # gp.columns --> list available filtering options    
    #g = gp.loc[gp['ADMIN'] == 'Greenland']['geometry'].values # --> county level extract
    g = gp.loc[gp['CONTINENT'] == continent]['geometry']

    # TEST: point_inside_shape(p,g)     
    #p = Point(0.0, 0.0) # --> not in Africa
    #p = Point(3.0, 16.46) # --> Timbuktu is in Africa

    # EXTRACT: short-segment series in continent    
    
    decimalplaces = 1  
    
    ds_stationcode_in_continent = []
    ds_lat_in_continent = []
    ds_lon_in_continent = []    
    for i in range(len(ds_lonlat)):
        p = Point(ds_lonlat.iloc[i,0],ds_lonlat.iloc[i,1])                  
        if g.contains(p).any() == True:
            ds_stationcode_in_continent.append(ds_lonlat.index[i])
            ds_lat_in_continent.append(ds_lonlat.lat[i])
            ds_lon_in_continent.append(ds_lonlat.lon[i])            
    ds_lat_in_continent = pd.Series(ds_lat_in_continent).apply(lambda x: round(x, decimalplaces))
    ds_lon_in_continent = pd.Series(ds_lon_in_continent).apply(lambda x: round(x, decimalplaces))
    ds_in_continent = pd.DataFrame({'stationcode':ds_stationcode_in_continent, 'stationlat':ds_lat_in_continent, 'stationlon':ds_lon_in_continent}).reset_index(drop=True)
    ds_filestr = 'ds'+'_'+ continent.lower().replace(' ','_') +'.csv'
    ds_in_continent.to_csv(ds_filestr)

    # EXTRACT: series with normals in continent    
    
    da_stationcode_in_continent = []
    da_lat_in_continent = []
    da_lon_in_continent = []    
    for i in range(len(da_lonlat)):
        p = Point(da_lonlat.iloc[i,0],da_lonlat.iloc[i,1])                  
        if g.contains(p).any() == True:
            da_stationcode_in_continent.append(da_lonlat.index[i])
            da_lat_in_continent.append(da_lonlat.lat[i])
            da_lon_in_continent.append(da_lonlat.lon[i])
    da_lat_in_continent = pd.Series(da_lat_in_continent).apply(lambda x: round(x, decimalplaces))
    da_lon_in_continent = pd.Series(da_lon_in_continent).apply(lambda x: round(x, decimalplaces))
    da_in_continent = pd.DataFrame({'stationcode':da_stationcode_in_continent, 'stationlat':da_lat_in_continent, 'stationlon':da_lon_in_continent}).reset_index(drop=True)
    da_filestr = 'da'+'_'+ continent.lower().replace(' ','_') +'.csv'
    da_in_continent.to_csv(da_filestr)

    # EXTRACT: all series in continent    
    
    dt_stationcode_in_continent = []
    dt_lat_in_continent = []
    dt_lon_in_continent = []    
    for i in range(len(dt_lonlat)):
        p = Point(dt_lonlat.iloc[i,0],dt_lonlat.iloc[i,1])                  
        if g.contains(p).any() == True:
            dt_stationcode_in_continent.append(dt_lonlat.index[i])
            dt_lat_in_continent.append(dt_lonlat.lat[i])
            dt_lon_in_continent.append(dt_lonlat.lon[i])
    dt_lat_in_continent = pd.Series(dt_lat_in_continent).apply(lambda x: round(x, decimalplaces))
    dt_lon_in_continent = pd.Series(dt_lon_in_continent).apply(lambda x: round(x, decimalplaces))
    dt_in_continent = pd.DataFrame({'stationcode':dt_stationcode_in_continent, 'stationlat':dt_lat_in_continent, 'stationlon':dt_lon_in_continent}).reset_index(drop=True)
    dt_filestr = 'dt'+'_'+ continent.lower().replace(' ','_') +'.csv'
    dt_in_continent.to_csv(dt_filestr)
                
#------------------------------------------------------------------------------
# PLOTS
#------------------------------------------------------------------------------

if plot_station_locations == True:
    
    #------------------------------------------------------------------------------
    # PLOT: stations on world map
    #------------------------------------------------------------------------------

    print('plot_station_locations ...')
        
    figstr = 'short-v-long-segment-stations-all.png'
    titlestr = 'GloSAT.p03: N(stations)=' + str(len(dt['stationcode'].unique()))
     
    fig  = plt.figure(figsize=(15,10))
    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
    if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)    
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, edgecolor=None)    
    if show_gridlines == True:
        ax.gridlines()     
        if projection == 'platecarree':
            ax.set_extent([-180, 180, -90, 90], crs=p)    
            gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
            gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': fontsize}
            gl.ylabel_style = {'size': fontsize}           
    plt.scatter(x=dt_lon, y=dt_lat, 
                color="red", s=1, alpha=0.5,
                transform=ccrs.PlateCarree(), label='Global: N(short-segment stations)='+str(len(dt['stationcode'].unique()) - len(da['stationcode'].unique()) )) 
    plt.scatter(x=da_lon, y=da_lat, 
                color="blue", s=1, alpha=0.5,
                transform=ccrs.PlateCarree(), label='Global: N(stations with 1961-1990 normals)='+str(len(da['stationcode'].unique())) ) 
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.25), markerscale=6, facecolor='lightgrey', framealpha=1, fontsize=14)    
    plt.title(titlestr, fontsize=fontsize, pad=0)
    plt.savefig(figstr, dpi=300)
    plt.close('all')

if plot_climgen_categories == True:

    #------------------------------------------------------------------------------
    # PLOT: ClimGen categories on world map
    #------------------------------------------------------------------------------

    print('plot_ClimGen_categories ...')
        
    figstr = 'climgen-categories.png'
    titlestr = 'ClimGen Categories'
     
    categorystr = '0:Antarctica, 1:Europe, 2:ex-USSR, 3:Middle-East, 4:Asia, 5:Africa, 6:N America, 7:Central America, 8:S America, 9:Oceania, 10:Sea'
    
    fig  = plt.figure(figsize=(15,10))
    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
    if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=1, edgecolor=None)
    if show_gridlines == True:
        ax.gridlines()            
        if projection == 'platecarree':
            ax.set_extent([-180, 180, -90, 90], crs=p)    
            gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
            gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': fontsize}
            gl.ylabel_style = {'size': fontsize}               
    plt.scatter(x=dc.lon, y=dc.lat, c=dc.cat,  
                s=1, alpha=1.0, transform=ccrs.PlateCarree(), cmap=cmap, label=categorystr)      
#    cb = plt.colorbar(orientation="horizontal", shrink=0.5, pad=0.05)    
#    cb.ax.tick_params(labelsize=fontsize)    
#    cb.ax.set_xticks(np.arange(len(dc.cat.unique()))) 
##   cb.ax.set_xticklabels(np.arange(len(dc.cat.unique())))         
##   cb.set_label('Category number', labelpad=0, fontsize=fontsize)
    plt.legend(loc='lower left', bbox_to_anchor=(-0.07, -0.15), handlelength=0, handletextpad=0, facecolor='lightgrey', framealpha=1, fontsize=14)    
    plt.title(titlestr, fontsize=fontsize, pad=0)
    plt.savefig(figstr, dpi=300)
    plt.close('all')
        
if plot_continent_stations == True:

    #------------------------------------------------------------------------------
    # PLOT: stations in continent
    #------------------------------------------------------------------------------

    print('plot_continent_stations ...')

    figstr = 'short-v-long-segment-stations' + '-' + continent.lower().replace(' ','-') + '.png'
    titlestr = 'GloSAT.p03: N(stations)=' + str(len(dt['stationcode'].unique()))
         
    fig  = plt.figure(figsize=(15,10))
    if projection == 'platecarree': p = ccrs.PlateCarree(central_longitude=0); threshold = 0
    if projection == 'mollweide': p = ccrs.Mollweide(central_longitude=0); threshold = 1e6
    if projection == 'robinson': p = ccrs.Robinson(central_longitude=0); threshold = 0
    if projection == 'equalearth': p = ccrs.EqualEarth(central_longitude=0); threshold = 0
    if projection == 'geostationary': p = ccrs.Geostationary(central_longitude=0); threshold = 0
    if projection == 'goodehomolosine': p = ccrs.InterruptedGoodeHomolosine(central_longitude=0); threshold = 0
    if projection == 'europp': p = ccrs.EuroPP(); threshold = 0
    if projection == 'northpolarstereo': p = ccrs.NorthPolarStereo(); threshold = 0
    if projection == 'southpolarstereo': p = ccrs.SouthPolarStereo(); threshold = 0
    if projection == 'lambertconformal': p = ccrs.LambertConformal(central_longitude=0); threshold = 0
    if projection == 'winkeltripel': p = ccrs.WinkelTripel(central_longitude=0); threshold = 0
    ax = plt.axes(projection=p)        
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, edgecolor=None)
    if shade_continent == True:
        projection = 'platecaree'
        p = ccrs.PlateCarree(central_longitude=0); threshold = 0 # --> needed by GeoPandas
        ax.add_geometries(g, crs=p, facecolor='green', zorder=0, alpha=1.0, edgecolor='k')    
    if show_gridlines == True:                      
        ax.gridlines()            
        if projection == 'platecarree':
            ax.set_extent([-180, 180, -90, 90], crs=p)    
            gl = ax.gridlines(crs=p, draw_labels=False, linewidth=1, color='k', alpha=1.0, linestyle='-')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlines = True
            gl.ylines = True
            gl.xlocator = mticker.FixedLocator([-180,-120,-60,0,60,120,180])
            gl.ylocator = mticker.FixedLocator([-90,-60,-30,0,30,60,90])
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': fontsize}
            gl.ylabel_style = {'size': fontsize}               
    plt.scatter(x=dt_lon_in_continent, y=dt_lat_in_continent, 
                color="red", s=1, alpha=0.5,
                transform=ccrs.PlateCarree(), label=continent+': N(short-segment stations)='+str(len(ds_stationcode_in_continent)) ) 
    plt.scatter(x=da_lon_in_continent, y=da_lat_in_continent, 
                color="blue", s=1, alpha=0.5,
                transform=ccrs.PlateCarree(), label=continent+': N(stations with 1961-1990 normals)='+str(len(da_stationcode_in_continent)) ) 
    plt.legend(loc='lower left', bbox_to_anchor=(0, -0.25), markerscale=6, facecolor='lightgrey', framealpha=1, fontsize=14)        
    plt.title(titlestr, fontsize=fontsize, pad=0)
    plt.savefig(figstr, dpi=300)
    plt.close('all')

#------------------------------------------------------------------------------
print('** END')

