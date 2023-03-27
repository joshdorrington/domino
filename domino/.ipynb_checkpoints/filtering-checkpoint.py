import numpy as np
from scipy.ndimage import label
import itertools as it

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

def da_large_regions(da,n,dims):
    return apply_2d_func_to_da(da,get_large_regions,n,dims=dims)

def ds_large_regions(mask_ds,n,dims):
        ds=mask_ds.copy()
        for var in list(ds.data_vars):
            ds[var]=da_large_regions(ds[var],n,dims)
        return ds
    

def convolve_pad(x,N):
    Y=np.array([np.convolve(y,np.ones(N),mode='same') for y in x])
    Y=np.array([np.convolve(y,np.ones(N),mode='same') for y in Y.T]).T
    Y[Y>0]=1
    return Y

def convolve_pad_da(da,n,dims):
    return apply_2d_func_to_da(da,convolve_pad,n,dims=dims)

def convolve_pad_ds(ds,n,dims):
    ds=ds.copy()
    for var in list(ds.data_vars):
        ds[var]=convolve_pad_da(ds[var],n,dims=dims)
    return ds

