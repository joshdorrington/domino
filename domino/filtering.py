import numpy as np
from scipy.ndimage import label
import itertools as it
import xarray as xr

def latlonarea(lat0,lat1,lon0,lon1,r=6731):
    theta=np.deg2rad((lat1+lat0)/2)
    r2=r**2
    dphi=lon1-lon0
    dtheta=lat1-lat0
    A=r2*np.abs(np.cos(theta))*dtheta*dphi #in km**2
    return A

def grid_area(da,lat_coord='lat',lon_coord='lon',r=6371):
    lat=da[lat_coord].values
    lon=da[lon_coord].values
    dlon=(lon[1:]-lon[:-1])/2
    dlat=(lat[1:]-lat[:-1])/2
    dlon=[np.median(dlon),*dlon]
    dlat=[np.median(dlat),*dlat]
    areas=np.array([[\
        latlonarea(la-dla,la+dla,lo-dlo,lo+dlo,r=r)\
        for la,dla in zip(lat,dlat)]\
        for lo,dlo in zip(lon,dlon)])
    area=xr.DataArray(data=areas.T,\
        coords={lat_coord:lat,lon_coord:lon})
    return np.abs(area)


def apply_2d_func_to_da(da,func,*args,dims=None):
    da=da.copy(deep=True)
    da_dim=None    
    if dims is not None:
        #if dims is specified, ignore da's without all dims
        if not set(dims).issubset(set(da.dims)):
            return da
        #Put dims at the front
        else:
            da_dim=list(da.dims)
            da=da.transpose(*dims,...)
            
        #Since funcs are 2D,
        #we must add a dummy dim if we only want 1 dim
        if len(dims)==1:
            da=da.expand_dims('dummy_dim',0).copy(deep=True)
            
    #We now want to loop over all dims except the first two
    #and apply our function
    indexers=[np.arange(i) for i in da.shape[2:]]
    for i in it.product(*indexers):
        da[(...,*i)]=func(da[(...,*i)].values,*args)
    try:
        da=da.isel(dummy_dim=0)
    except Exception as e:
        pass
    if da_dim is not None:
        da=da.transpose(*da_dim)    
    return da

def get_large_regions(grid,point_thresh):
    connected,ncon=label(grid)
    connected=connected.astype(float)
    ix,n=np.unique(connected,return_counts=True)
    for m in ix[n<point_thresh]:
        connected[connected==m]=0
    return connected>0

def get_area_regions(grid,area_thresh,area):
    connected,ncon=label(grid)
    connected=connected.astype(float)
    ix,n=np.unique(connected,return_counts=True)
    
    connected_da=xr.DataArray(connected,coords=area.coords)
    ix_areas=np.array([area.where(connected_da==i).sum().values\
              for i in ix])
    for m in ix[ix_areas<area_thresh]:
        connected[connected==m]=0
    return connected>0


def da_large_regions(da,n,dims,area_based=False):
    
    if area_based:
        
        if len(dims)!=2:
            raise(ValueError('Area weighted n currently only supported for two dimensions, assumed to be (lat,lon)'))
        lat,lon=dims
        #area in units of solid angle
        area=grid_area(da,lat_coord=lat,lon_coord=lon,r=1)
        
        return apply_2d_func_to_da(da,get_area_regions,n,area,dims=dims)
    else:
        return apply_2d_func_to_da(da,get_large_regions,n,dims=dims)

def ds_large_regions(mask_ds,n,dims,area_based=False):
        ds=mask_ds.copy()
        for var in list(ds.data_vars):
            
            if (dims is not None) and (not np.all([d in mask_ds[var].dims for d in dims])):
                pass #ignore variables with missing dims
            else:
                ds[var]=da_large_regions(ds[var],n,dims,area_based=area_based)
        return ds

def convolve_pad(x,N):
    Y=np.array([np.convolve(y,np.ones(np.min([N,len(y)])),mode='same') for y in x])
    Y=np.array([np.convolve(y,np.ones(np.min([N,len(y)])),mode='same') for y in Y.T]).T
    Y[Y>0]=1
    return Y

def convolve_pad_da(da,n,dims):
    return apply_2d_func_to_da(da,convolve_pad,n,dims=dims)

def convolve_pad_ds(ds,n,dims):
    ds=ds.copy()
    for var in list(ds.data_vars):
        ds[var]=convolve_pad_da(ds[var],n,dims=dims)
    return ds

def std_filter(val_ds,std,frac):
    return np.abs(val_ds)>=frac*std