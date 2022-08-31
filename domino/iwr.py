import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
import scipy.signal as ssignal

from util import lon_shift,standardise
from deseasonaliser import Agg_Deseasonaliser
from plotting import contourf_grid


def area_restrict(X,lats,lons,lat_coord,lon_coord):
    X=X.isel({lat_coord:(X[lat_coord]>=lats[0])&(X[lat_coord]<=lats[1])})
    X=X.isel({lon_coord:(X[lon_coord]>=lons[0])&(X[lon_coord]<=lons[1])})
    return X

def time_restrict(X,ts,t_coord='time'):
    t1=pd.to_datetime(ts[0])
    t2=pd.to_datetime(ts[1])
    X=X.isel({t_coord:(X[t_coord]>=t1)&(X[t_coord]<=t2)})
    return X        

#Assumes time is first axis
def compute_smooth_anomaly(X,smooth=91,filter_cutoff=10,butter_order=3):
    
    #Compute day_of_year average and smooth it with rolling average smoothing
    dsnlsr=Agg_Deseasonaliser()
    dsnlsr.fit_cycle(X)
    cycle=dsnlsr.evaluate_cycle(X.time,smooth=smooth)
    
    #Compute anom:
    X_anom=X.data-cycle.data
    #Lowpass filter with bilinear butterworth filter
    lowpass_filter=ssignal.butter(butter_order,1/filter_cutoff,btype='low',fs=1,output='sos')
    filtered_X=ssignal.sosfiltfilt(lowpass_filter,X_anom,axis=0)
    return X.copy(data=filtered_X)

def get_day_range(n,w):
    return np.roll(np.arange(1,366),-n+w//2)[:w]

def get_windowed_std(X,w=31):
    
    stds=[X[np.isin(X['time.dayofyear'],get_day_range(n,31))].std(dim='time') for n in np.arange(1,367)]
    
    weights=np.cos(np.deg2rad(X.lat.values))
    stds=np.array(stds)*weights[None,:,None]
    return np.mean(stds,axis=(1,2))

def apply_std_weighting(X,s_norm):
    weights=np.array([s_norm[v-1] for v in X['time.dayofyear'].values])
    return X/weights[:,None,None]

def standardise_proj_vs_background(proj,proj_background):
    
    mean_background=proj_background.mean('time')
    proj_anom=proj-mean_background
    normalisation=np.sqrt(np.sum((proj_anom)**2))/np.sqrt(len(proj_anom)-1)
    return proj_anom/normalisation

def IWR_proj(X,X_regmean,lats=[30,90],lons=[-80,40],lat_coord='lat',lon_coord='lon'):
    
    X=area_restrict(X,lats,lons,lat_coord,lon_coord)
    X_regmean=area_restrict(X_regmean,lats,lons,lat_coord,lon_coord)
        
    lats=X_regmean[lat_coord].values
    lat_weights=np.cos(np.deg2rad(lats))
    proj=np.sum(X*X_regmean*lat_weights[None,:,None],axis=(1,2))
    proj=proj/np.sum(lat_weights)
    return proj

def compute_IWR(X,X_regmean,\
    ref_ts=None,lats=[30,90],lons=[-80,40],lat_coord='lat',lon_coord='lon',ref=None):
    
    if ref_ts is None:
        ref_ts=[dt.datetime(1979,1,1),dt.datetime(2015,12,31,23)]
    fixed_args=[lats,lons,lat_coord,lon_coord]
    IWR=IWR_proj(X,X_regmean,*fixed_args)
    
    if ref is None:
        IWR_ref=IWR_proj(time_restrict(X,ref_ts),X_regmean,*fixed_args)
    else:
        IWR_ref=ref
    return standardise_proj_vs_background(IWR,IWR_ref)

#A wrapper func
def compute_Grams_IWR(X,states,lats=[30,90],lons=[-80,40],ts=None,lat_coord='lat',lon_coord='lon'):
    
    X_ref,X_anom=preprocess_X(X,lats,lons,ts,lat_coord,lon_coord)
    IWR=get_all_regime_iwr(X_ref,X_anom,states)
    return IWR

#Apply all filtering and define a reference time series
def preprocess_X(X,lats=[30,90],lons=[-80,40],ts=None,lat_coord='lat',lon_coord='lon'):
    if ts is None:
        ts=[dt.datetime(1979,1,1),dt.datetime(2019,12,31,23)]
        
    X_anom=area_restrict(lon_shift(X),[30,90],[-80,40],lat_coord,lon_coord)
    X_anom=compute_smooth_anomaly(X_anom)
    X_ref=time_restrict(X_anom,ts)
    
    seasonal_normalisation=get_windowed_std(X_ref)
    X_ref=apply_std_weighting(X_ref,seasonal_normalisation)
    X_anom=apply_std_weighting(X_anom,seasonal_normalisation)
    return X_ref,X_anom

#Actually compute the IWR
def get_all_regime_iwr(X_ref,X_anom,states):
    iwr=[]
    
    X_ref,states=xr.align(X_ref,states)

    for s in np.unique(states.values):
        
        X_reg=X_ref[states.values==s].mean('time')
        ref_proj=IWR_proj(X_ref,X_reg)
        IWR_index=compute_IWR(X_anom,X_reg,ref=ref_proj)
        iwr.append(IWR_index)
    return xr.concat(iwr,'state').T