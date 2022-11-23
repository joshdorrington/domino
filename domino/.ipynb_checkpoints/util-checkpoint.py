import pandas as pd
import datetime as dt
import numpy as np
import xarray as xr
import itertools

def squeeze_da(da):   
    return da.drop([c for c in da.coords if np.size(da[c])==1])

def drop_scalar_coords(ds):
    for v in [v for v in list(ds.coords) if ds[v].size<2]:
        ds=ds.drop(v)
    return ds

def make_all_dims_coords(da):
    return da.assign_coords({v:da[v] for v in da.dims})

#Takes a scalar input!
def is_time_type(x):
    return (isinstance(x,dt.date) or isinstance(x,np.datetime64))

def offset_time_dim(da,offset,offset_unit='days',offset_dim='time',deep=False):
    time_offset=dt.timedelta(**{offset_unit:offset})
    new_dim=pd.to_datetime(da[offset_dim])+time_offset
    new_da=da.copy(deep=deep)
    new_da[offset_dim]=new_dim
    return new_da


#Takes a time axis t_arr, and splits it into
#contiguous subarrays. Alternatively it splits an 
#axis x_arr into subarrays. If no dt is provided
#to define contiguous segments, the minimum difference
#between elements of t_arr is used
#alternatively alternatively, you can set max_t, which overrides
#dt, and considers any gap less than max_t to be contiguous
def split_to_contiguous(t_arr,x_arr=None,dt=None,max_t=None):
    
    if x_arr is None:
        x_arr=t_arr.copy()
        
    #make sure everything is the right size
    t_arr=np.array(t_arr)
    x_arr=np.array(x_arr)
    try:
        assert len(x_arr)==len(t_arr)
    except:
        print(len(x_arr))
        print(len(t_arr))
        raise(AssertionError())
    #Use smallest dt if none provided
    if dt is None:
        dt=np.sort(np.unique(t_arr[1:]-t_arr[:-1]))[0]
        
    #The default contiguous definition    
    is_contiguous = lambda arr,dt,max_t: arr[1:]-arr[:-1]==dt
    #The alternate max_t contiguous definition
    if max_t is not None:
        is_contiguous=lambda arr,dt,max_t: arr[1:]-arr[:-1]<max_t
        
    #Split wherever is_contiguous is false
    return np.split(x_arr[1:],np.atleast_1d(np.squeeze(np.argwhere(np.invert(is_contiguous(t_arr,dt,max_t))))))

#Take a list of pvals from multiple hypothesis tests and an alpha value to test against (i.e. a significance threshold)
#Returns a boolean list saying whether to regard each pval as significant
def holm_bonferroni_correction(pvals,alpha):
    p_order=np.argsort(pvals)
    N=len(pvals)
    
    #Calculate sequentially larger alpha values for each successive test
    alpha_corrected=alpha/(N+1-np.arange(1,N+1))
    
    #Put the pvalues in increasing order
    sorted_p=np.array(pvals)[p_order]
    
    #Get the first pval that exceeds its corrected alpha value
    K=np.argwhere(sorted_p>alpha_corrected)
    if len(K)==0:
        K=N
    else:
        K=np.min(K)
    #Keep all the first K-1 and reject the rest:

    significant=np.array([*np.repeat(True,K),*np.repeat(False,N-K)])
    
    #Undo the ordering
    significant=significant[np.argsort(p_order)]
    
    return significant

def event_from_datetimes(events,d1,d2,subset_dict={}):
    #Input checking, also throws error for ragged lists
    if np.all(np.array([np.ndim(e) for e in events])==0):
        events=[events]
    if not np.all(np.array([np.ndim(e) for e in events])==1):
        raise(ValueError('events must be 1 or 2 dimensional'))

    duplicate_event_dates=np.sum([np.isin(e1,e2).sum()\
        for e1,e2 in itertools.combinations(events,2)])
    if duplicate_event_dates!=0:
        raise(ValueError('2 events on the same day not supported.'))
    
    #Meat of the function
    daterange=pd.date_range(d1,d2)
    for k,x in subset_dict.items():
        daterange=daterange[np.isin(getattr(daterange,k),x)]  
    event_index=np.zeros(len(daterange))
    for i,e in enumerate(events):
        event_index[np.isin(daterange.to_list(),e)]=i+1
    event_index=xr.DataArray(data=event_index,coords={'time':daterange})
    return event_index

holm_bonferroni_correction,
split_to_contiguous,
is_time_type,
make_all_dims_coords,
drop_scalar_coords,
squeeze_da
