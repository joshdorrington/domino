import pandas as pd
import datetime as dt
import numpy as np
import xarray as xr
import itertools

def squeeze_da(da):  
    """ Remove length 1 coords from a DataArray"""
    return da.drop([c for c in da.coords if np.size(da[c])==1])

def drop_scalar_coords(ds):
    """ Remove coords without dimensions from Dataset"""
    for v in [v for v in list(ds.coords) if ds[v].size<2]:
        ds=ds.drop(v)
    return ds

def make_all_dims_coords(da):
    """Convert all dims to coords"""
    return da.assign_coords({v:da[v] for v in da.dims})

#Takes a scalar input!
def is_time_type(x):
    return (isinstance(x,dt.date) or isinstance(x,np.datetime64))

def offset_time_dim(da,offset,offset_unit='days',offset_dim='time',deep=False):#
    """Shifts the time-like *offset_dim* coord of *da* by *offset* *offset_units*.
    
    e.g. offset_time_dim(da,3,'days'), adds three days to the time axis of da."""
    
    time_offset=dt.timedelta(**{offset_unit:offset})
    
    #rewritten to handle older pandas versions that don't play nicely with dataarrays
    offset_dim_vals=pd.to_datetime(da[offset_dim].values)+time_offset
    new_da=da.copy(deep=deep)
    new_da=new_da.assign_coords({offset_dim:offset_dim_vals})
    
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

def xarr_times_to_ints(time_coord):
    conversion=(1000*cftime.UNIT_CONVERSION_FACTORS["day"])
    return time_coord.to_numpy().astype(float)/conversion

def restrict(ds,extent_dict):
    """ Subset the coords of a Dataset *ds*, using *extent_dict*, a dictionary of the form {coord:[lower_bound,upper_bound],...}."""
    ds=ds.copy()
    for key in extent_dict:
        if key in ds.dims:
            xmin,xmax=extent_dict[key]

            in_range=(ds[key].values>=xmin)&(ds[key].values<=xmax)
            ds=ds.isel({key:in_range})
    return ds

def offset_indices(indices,offsets=None,infer_offset=True,attr_kw=None,offset_unit='days',dim='time'):
    """For a Dataset *indices* and either a dictionary of *offsets* ({data_var:offset,...}) or offsets stored in an attribute *attr_kw*, offset each index along the *dim* coord and take their union."""
    if offsets is None and not infer_offset:
        raise(ValueError('offsets must be provided or infer_offset must be True'))
    if attr_kw is None and infer_offset:
        raise(ValueError('attr_kw must be specified for infer_offset'))
    
    if infer_offset:
        offsets={v:indices[v].attrs[attr_kw] for v in indices.data_vars}
    
    da_arr=[]
    for v in indices.data_vars:
        da=indices[v]
        l=offsets[v]
        if type(l)==np.int_:
            l=int(l) #datetime can't handle np.ints, no idea why not.
        da_arr.append(offset_time_dim(da,-l,offset_unit,offset_dim=dim))
    ds=xr.merge(da_arr)
    return ds
