import cftime
import numpy as np
import pickle
import iris
import pandas as pd
import datetime as dt
import numpy as np
import cf_units

def squeeze_da(da):   
    return da.drop([c for c in da.coords if np.size(da[c])==1])

def standardise(da,dim=None):
    da=da-da.mean(dim=dim)
    da=da/da.std(dim=dim)
    return da
def drop_scalar_coords(ds):
    for v in [v for v in list(ds.coords) if ds[v].size<2]:
        ds=ds.drop(v)
    return ds

def make_all_dims_coords(da):
    return da.assign_coords({v:da[v] for v in da.dims})

#Takes a scalar input!
def is_time_type(x):
    return (isinstance(x,dt.date) or isinstance(x,np.datetime64))

def is_two_valued(x,dropnan=True):
    if dropnan:
        x=np.array(x)[~np.isnan(x)]
    return len(np.unique(x))==2

def offset_time_dim(da,offset,offset_unit='days',offset_dim='time',deep=False):
    time_offset=dt.timedelta(**{offset_unit:offset})
    new_dim=pd.to_datetime(da[offset_dim])+time_offset
    new_da=da.copy(deep=deep)
    new_da[offset_dim]=new_dim
    return new_da

def xarr_times_to_ints(time_coord):
    conversion=(1000*cftime.UNIT_CONVERSION_FACTORS["day"])
    return time_coord.to_numpy().astype(float)/conversion

#shifts lon from 0-360 to -180 to 180
def lon_shift(da,name='lon'):
    da=da.copy()
    da.coords[name] = (da.coords[name] + 180) % 360 - 180
    return da.sortby(da[name])

def load_pickle(p):
    with open(p, "rb") as handle: 
        x=pickle.load(handle)
    return x

def make_pickle(p,x):
    with open(p, 'wb') as handle: 
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 1

#Takes in a cube, and N set of season years, and returns a dictionary of cubes
# N long containing those slices. If as_data=True, returns numpy data in the
#dict instead of cubes.
def window_cube(cube,subsets,as_data=True):
    if as_data:
        cube_windowed={i:cube.extract(iris.Constraint(season_year=subset)).data\
                for i,subset in enumerate(subsets)}
    else:
        cube_windowed={i:cube.extract(iris.Constraint(season_year=subset))\
                for i,subset in enumerate(subsets)}

    return cube_windowed

#Takes an array of shape [Time,*Dimensions] and returns a randomised
#array of the same shape with the same Fourier spectrum for each dimension
def randomise_phase(X):
    Xmean=np.mean(X,axis=0)
    X=X-Xmean[None,...]
    
    Y=np.fft.rfft(X,axis=0)
    R=np.abs(Y)
    theta=np.angle(Y)
    random_theta=np.random.uniform(low=-np.pi,high=np.pi,size=theta.size)
    random_theta=random_theta.reshape(theta.shape)
    random_Y=R*np.e**(1j*random_theta)
    random_X=np.fft.irfft(random_Y,axis=0)
    random_X=random_X+Xmean[None,...]
    return random_X

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

#Take a cube, copy everything about it except its data and first axis (assumed time by default).
#This is useful for combining data from different cubes on one time axis without much fuss.
#Particularly good for unravelling ensembles.
def make_cube_with_different_1st_axis(cube,new_t,t_ax=None):
    S=cube.shape
    
    if t_ax is None:
        t_ax=iris.coords.DimCoord(np.arange(0,new_t),"time",units=cf_units.Unit(f"days since {cf_units.EPOCH}"))
        
    new_cube=iris.cube.Cube(data=np.zeros([new_t,*S[1:]]))
    
    new_cube.add_dim_coord(t_ax,0)
    for i,coord in enumerate(cube.dim_coords[1:]):
        new_cube.add_dim_coord(coord,i+1)
        
    new_cube.standard_name=cube.standard_name
    new_cube.long_name=cube.long_name
    new_cube.var_name=cube.var_name
    new_cube.units=cube.units
    new_cube.metadata=cube.metadata
    new_cube.attributes=cube.attributes
    
    try:
        new_cube.attributes["history"]=\
        new_cube.attributes["history"]+" Remade into a different shape by Josh Dorrington."
    except:
        new_cube.attributes["history"]=" Remade into a different shape by Josh Dorrington."
        
    return new_cube


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


def reduce_data_to_intersection(data,offsets=None,tcoord="time",dtype=None):
    
    if offsets is None: 
        offsets=np.repeat(0,len(data))
    t=[d.coord(tcoord).points-o for d,o in zip(data,offsets)]
    
    if dtype is not None:
        
        t=np.array([d.astype(dtype) for d in t])

    keep_t=list(set(t[0]).intersection(*t))
    keep_d=[np.isin(T,keep_t) for T in t]
    r=[d[k] if np.any(k) else None for d,k in zip(data,keep_d) ]
    return r

def get_empirical_cdf(data,x_ax=None):
    
    if x_ax is None:
        x_ax=np.linspace(data.min(),data.max(),len(data))
    
    y=np.array([np.sum(data<x) for x in x_ax])/len(data)
    return y

def KS_test(d1,d2,x_ax=None,return_distance=False):
    
    d1=np.array(d1)
    d2=np.array(d2)
    
    m=len(d1)
    n=len(d2)

    if x_ax is None:
        x_ax=np.linspace(np.min([d1.min(),d2.min()]),np.max([d1.max(),d2.max()]),np.max([m,n]))
        
        
    edf1=get_empirical_cdf(d1,x_ax)
    edf2=get_empirical_cdf(d2,x_ax)
    
    D=np.max(np.abs(edf1-edf2))
    
    #This is the sample size term
    s=(n+m)/(n*m)
    
    alpha_val=2*np.exp(-2*(D**2)/s)
    if return_distance:
        return alpha_val,D
    else:
        return alpha_val
    
#Recursive function that gets the intersection of a sequence
#of data series. Designed for getting compatible time
#series lined up.
def _get_keep_t(t):
    
    if len(t)==1:
        return set(t[0])
    else:
        return set(t[0])&_get_keep_t(t[1:])
    
    raise(ValueError("Something went wrong."))
  

def AR1(length,autocorr=0.9,std=1,init=0):
    
    sigma=np.sqrt((1-autocorr**2))*std
    seq=[]
    seq.append(init)
    for n in range(length):
        seq.append(autocorr*seq[-1]+np.random.randn(1)*sigma)
    return np.array(seq)

def acf(x,nmax=50):
    
    return np.array([1,*[np.corrcoef(x[n:],x[:-n])[1,0] for n in range(1,nmax+1)]])


def regress(x,y, deg=1, prnt=True):
    import numpy as np
    import numpy.ma as ma
    from scipy.stats import pearsonr
    #Check for mask
    if ma.is_masked(x) or ma.is_masked(y):
        fitter = ma.polyfit
    else:
        fitter = np.polyfit
    model = fitter(x,y,deg)
    prediction = np.polyval(model, x)
    residuals = y - prediction
    corr, pval = pearsonr(prediction, y)
    if prnt:
        print("Fitted model: y = %.3f*x + %.3f" % (model[0], model[1]))
        print("Correlation = %.3f (p=%.3f)" % (corr, pval))
        print("Explained variance = %.3f" % (corr**2))
        print("Returning (coefficients, prediction, residuals)")
    return corr, residuals

def combine_time_series(x_dict,t_dict,ix=None):
    
    ts=np.unique([t for m,T in t_dict.items() for t in T])
        
    xs=[]
    
    for m,t in t_dict.items():
        x=x_dict[m]
        if ix is not None:
            x=x[:,ix]
            
        x_arr=np.zeros_like(ts)*np.nan
        in_t=np.isin(ts,t)
        x_arr[in_t]=x
        
        xs.append(x_arr)
    return ts,np.array(xs)

#Func must be of the form f(c1,c2,**kwargs)
def force_2cube_operation(c1,c2,func,**fkwargs):
    
    newc2=c1.copy()
    newc2.data=c2.data
    return func(c1,newc2,**fkwargs)

def mean_cube_arr(arr):
    
    c0=arr[0]
    cmean=c0.copy()
    L=len(arr)
    if L==1:
        return c0
    for c in arr[1:]:
        
        cmean.data+=c.data
    cmean=cmean/L
    return cmean