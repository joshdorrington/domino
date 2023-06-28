import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import itertools as it
from scipy.ndimage import label
import dateutil.relativedelta as durel

from domino import agg
from domino.categorical_analysis import get_transmat, synthetic_states_from_transmat
from domino.util import holm_bonferroni_correction, split_to_contiguous, is_time_type, make_all_dims_coords, drop_scalar_coords, squeeze_da,offset_time_dim
from domino.filtering import ds_large_regions, convolve_pad_ds
from domino.deseasonaliser import Agg_Deseasonaliser

class LaggedAnalyser(object):
    """Computes lagged composites of variables with respect to a categorical categorical event series, with support for bootstrap resampling to provide a non-parametric assessment of composite significance, and for deseasonalisation of variables.
    
    **Arguments:**
        
    *event*
            
    An xarray.DataArray with one dimension taking on categorical values, each defining a class of event (or non-event).
            
    **Optional arguments**
        
    *variables, name, is_categorical*
        
    Arguments for adding variables to the LaggedAnalyser. Identical behaviour to calling *LaggedAnalyser.add_variables* directly.
    """
    
    def __init__(self,event,variables=None,name=None,is_categorical=None):
        """Initialise a new LaggedAnalyser object."""
        
        #: event is a dataarray
        self.event=xr.DataArray(event)#: This is a docstring?
        """@private"""
        
        #variables are stored in a dataset, and can be added later,
        #or passed as a DataArray, a Dataset or as a dict of DataArrays
        self.variables=xr.Dataset(coords=event.coords)
        """@private"""

        if variables is not None:
            self.add_variable(variables,name,is_categorical,False)
            
        #Time lagged versions of the dataset self.variables will be stored here, with a key
        #equal to the lag applied. Designed to be accessed by the self.lagged_variables function
        self._lagged_variables={}
        self.lagged_means=None
        """@private"""

        #variables that are a linear combination of other variables are more efficiently
        #computed after compositing using the self.add_derived_composite method
        self._derived_variables={}
        self._deseasonalisers={}
        
        self.composite_mask=None
        """@private"""

        self.boot_indices=None
        """@private"""

        return
    
    def __repr__(self):
        l1='A LaggedAnalyser object\n'
        l2='event:\n\n'
        da_string=self.event.__str__().split('\n')[0]
        l3='\n\nvariables:\n\n'
        ds_string=self.variables.__str__().split('\n')
        ds_string=ds_string[0]+' '+ds_string[1]
        ds_string2='\n'+self.variables.data_vars.__str__()
        if list(self._lagged_variables.keys())!=[]:
            lag_string=f'\n Lagged variables at time intervals:\n {list(self._lagged_variables.keys())}'
        else:
            lag_string=""
        return "".join([l1,l2,da_string,l3,ds_string,ds_string2,lag_string])
    
    def __str__(self):
        return self.__repr__()

    def add_variable(self,variables,name=None,is_categorical=None,overwrite=False,join_type='outer'):
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        
        *variables* 
        
        An xarray.DataArray, xarray.Dataset or dictionary of xarray.DataArrays, containing data to be composited with respect to *event*. One of the coordinates of *variables* should have the same name as the coordinate of *events*. Stored internally as an xarray.Dataset. If a dictionary is passed, the DataArrays are joined according to the method *join_type* which defaults to 'outer'.
            
        **Optional Arguments**
        
        *name* 
        
        A string. If *variables* is a single xarray.DataArray then *name* will be used as the name of the array in the LaggedAnalyser.variables DataArray. Otherwise ignored.
        
        *is_categorical* 
        
        An integer, if *variables* is an xarray.DataArray, or else a dictionary of integers with keys corresponding to DataArrays in the xarray.Dataset/dictionary. 0 indicates that the variable is continuous, and 1 indicates that it is categorical. Note that continuous and categorical variables are by default composited differently (see LaggedAnalyser.compute_composites). Default assumption is all DataArrays are continuous, unless a DataAarray contains an 'is_categorical' key in its DataArray.attrs, in which case this value is used.
            
        *overwrite*
        
        A boolean. If False then attempts to assign a variable who's name is already in *LaggedAnalyser.variables* will raise a ValueError
        
        *join_type*
        
        A string setting the rules for how differences in the coordinate indices of different variables are handled:
        “outer”: use the union of object indexes
        “inner”: use the intersection of object indexes

        “left”: use indexes from the pre-existing *LaggedAnalyser.variables* with each dimension

        “right”: use indexes from the new *variables* with each dimension

        “exact”: instead of aligning, raise ValueError when indexes to be aligned are not equal

        “override”: if indexes are of same size, rewrite indexes to be those of the pre-existing *LaggedAnalyser.variables*. Indexes for the same dimension must have the same size in all objects.
        """
        if isinstance(variables,dict):
            
            if is_categorical is None:
                is_categorical={v:None for v in variables}
                
            [self._add_variable(da,v,is_categorical[v],overwrite,join_type) for v,da in variables.items()]
            
        elif isinstance(variables,xr.Dataset):
            self.add_variable({v:variables[v] for v in variables.data_vars},None,is_categorical,overwrite,join_type)
            
        else:
            
            self._add_variable(variables,name,is_categorical,overwrite,join_type)            
        return
    
    def _more_mergable(self,ds):
        
        return drop_scalar_coords(make_all_dims_coords(ds))
    
    def _add_variable(self,da,name,is_categorical,overwrite,join_type):
        
        if name is None:
            name=da.name
        if (name in self.variables)&(not overwrite):
            raise(KeyError(f'Key "{name}" is already in variables.'))
        
        try:
            self.variables=self.variables.merge(squeeze_da(da).to_dataset(name=name),join=join_type)
        except:
            #Trying to make the merge work:
            self.variables=self._more_mergable(self.variables).merge(self._more_mergable(squeeze_da(da).to_dataset(name=name)),join=join_type)

        if (is_categorical is None) and (not 'is_categorical' in da.attrs):
            self.variables[name].attrs['is_categorical']=0
        elif is_categorical is not None:
            self.variables[name].attrs['is_categorical']=is_categorical

    def lagged_variables(self,t):
        """A convenience function that retrieves variables at lag *t* from the *LaggedAnalyser*"""
        if t in self._lagged_variables:
            return self._lagged_variables[t]
        elif t==0:
            return self.variables
        else:
            raise(KeyError(f'Lag {t} is not in self._lagged_variables.'))

    def _lag_variables(self,offset,offset_unit='days',offset_dim='time',mode='any',overwrite=False):
        
        if offset==0:
            return
        if (offset in self._lagged_variables)&(not overwrite):
            raise(KeyError(f'Key "{offset}" is already in lagged_variables.'))
            
        #We are really paranoid about mixing up our lags. So we implement this safety check
        self._check_offset_is_valid(offset,mode)
        
        #REPLACED PREVIOUS IMPLEMENTATION WITH EQUIVALENT UTIL IMPORT.
        self._lagged_variables[offset]=offset_time_dim(self.variables,-offset,offset_unit=offset_unit,offset_dim=offset_dim)

        return
    
    #For coords not in a time format
    def _ilag_variables(self,offset,*args,overwrite=False):
        raise(NotImplementedError('Only lagging along timelike dimensions is currently supported.'))
        
    def lag_variables(self,offsets,offset_unit='days',offset_dim='time',mode='any',overwrite=False):
        """Produces time lags of *LaggedAnalyser.variables*, which can be used to produce lagged composites.
        
        **Arguments**
        
        *offsets*
        
        An iterable of integers which represent time lags at which to lag *LaggedAnalyser.variables* in the units specified by *offset_unit*. Positive offsets denote variables *preceding* the event.
            
        **Optional arguments**
        
        *offset_unit*
        
        A string, defining the units of *offsets*. Valid options are weeks, days, hours, minutes, seconds, milliseconds, and microseconds.
            
        *offset_dim*
        
        A string, defining the coordinate of *LaggedAnalyser.variables* along which offsets are to be calculated.
            
        *mode*
        
        One of 'any', 'past', or 'future'. If 'past' or 'future' is used then only positive or negative lags are valid, respectively.
            
        *overwrite*
        
        A boolean. If False, then attempts to produce a lag which already exist will raise a ValueError.
        
        """
        time_type=int(is_time_type(self.variables[offset_dim][0].values))
        self.offset_unit=offset_unit
        lag_funcs=[self._ilag_variables,self._lag_variables]
        offsets=np.atleast_1d(offsets)
        for o in offsets:
            lag_funcs[time_type](int(o),offset_unit,offset_dim,mode,overwrite)
        
    def _check_offset_is_valid(self,offset,mode):
        
        valid_modes=['any','past','future']
        if not mode in valid_modes:
            raise(ValueError(f'mode must be one of {valid_modes}'))
        if offset>0 and mode == 'past':
            raise(ValueError(f'Positive offset {offset} given, but mode is "{mode}"'))
        if offset<0 and mode == 'future':
            raise(ValueError(f'Negative offset {offset} given, but mode is "{mode}"'))
        return
    
    """
        COMPOSITE COMPUTATION FUNCTIONS 
        Composite computation is split over 4 function layers:
        compute_composites(): calls
            _compute_aggregate_over_lags(): calls
                _composite_from_ix(): splits data into cat vars
                and con vars and then calls
                    _aggregate_from_ix(): applies an operation to
                    subsets of ds, where the ix takes unique values
                then merges them together.
             then loops over lags and merges those.
        And then substracts any anomalies and returns the data.


        Example usage of the aggregate funcs:
        i.e. self._aggregate_from_ix(ds,ix,'time',self._mean_ds)
        self._aggregate_from_ix(ds,ix,'time',self._std_ds)
        self._aggregate_from_ix(ds,ix,'time',self._cat_occ_ds,s=reg_ds)
    """
    
    def _aggregate_from_ix(self,ds,ix,dim,agg_func,*agg_args):
        return xr.concat([agg_func(ds.isel({dim:ix==i}),dim,*agg_args) for i in np.unique(ix)],'index_val')
    
    
    #Splits variables into cat and con and then combines the two different kinds of composites.
    #Used with a synthetic 'ix' for bootstrapping by self._compute_bootstraps.
    def _composite_from_ix(self,ix,ds,dim,con_func,cat_func,lag=0):
                
        ix=ix.values #passed in as a da
        cat_vars=[v for v in ds if ds[v].attrs['is_categorical']]
        con_vars=[v for v in ds if not v in cat_vars]
        cat_ds=ds[cat_vars]
        con_ds=ds[con_vars]
        cat_vals=cat_ds.map(np.unique)

        if (con_vars!=[]) and (con_func is not None):
            if (cat_vars!=[]) and (cat_func is not None):
                con_comp=self._aggregate_from_ix(con_ds,ix,dim,con_func)
                cat_comp=self._aggregate_from_ix(cat_ds,ix,dim,cat_func,cat_vals)
                comp=con_comp.merge(cat_comp)
            else:
                comp=self._aggregate_from_ix(con_ds,ix,dim,con_func)
        else:
                comp=self._aggregate_from_ix(cat_ds,ix,dim,cat_func,cat_vals)
        comp.attrs=ds.attrs
        return comp.assign_coords({'lag':[lag]})    
    
    #loops over all lags, calling _composite_from_ix, and assembles composites into a single dataset
    def _compute_aggregate_over_lags(self,da,dim,lag_vals,con_func,cat_func):
            
        if lag_vals=='all':
            lag_vals=list(self._lagged_variables)
                    
        composite=self._composite_from_ix(*xr.align(da,self.variables),dim,con_func,cat_func)
              
        if lag_vals is not None:
            lag_composites=[]
            for t in lag_vals:
                lag_composites.append(self._composite_from_ix(*xr.align(da,self.lagged_variables(t)),dim,con_func,cat_func,lag=t))
            composite=xr.concat([composite,*lag_composites],'lag').sortby('lag')
            
        return composite

    #The top level wrapper for compositing
    def compute_composites(self,dim='time',lag_vals='all',as_anomaly=False,con_func=agg.mean_ds,cat_func=agg.cat_occ_ds,inplace=True):
        
        """
        Partitions *LaggedAnalyser.variables*, and any time-lagged equivalents, into subsets depending on the value of *LaggedAnalyser.event*, and then computes a bulk summary metric for each.

        **Optional arguments**
        
        *dim*
        
        A string, the coordinate along which to compute composites.
            
        *lag_vals*
        
        Either 'All', or a list of integers, denoting the time lags for which composites should be computed.
            
        *as_anomaly*
        
        A Boolean, defining whether composites should be given as absolute values or differences from the unpartitioned value.
            
        *con_func*
        
        The summary metric to use for continuous variables. Defaults to a standard mean average. If None, then continuous variables will be ignored
            
        *cat_func*
        
        The summary metric to use for categorical variables. Defaults to the occurrence probability of each categorical value. If None, then categorical variables will be ignored
            
        *inplace*
    
        A boolean, defining whether the composite should be stored in *LaggedAnalyser.composites*
        
        **returns**
        
        An xarray.Dataset like  *LaggedAnalyser.variables* but summarised according to *con_func* and *cat_func*, and with an additional coordinate *index_val*, which indexes over the values taken by *LaggedAnalyser.event*.
            
        """
        composite=self._compute_aggregate_over_lags(self.event,dim,lag_vals,con_func,cat_func)
        lagged_means=self.aggregate_variables(dim,lag_vals,con_func,cat_func)

        if as_anomaly:
            composite=composite-lagged_means
            
        composite=make_all_dims_coords(composite)
        for v in list(composite.data_vars):
            composite[v].attrs=self.variables[v].attrs
        if inplace:
            self.composites=composite
            self.composite_func=(con_func,cat_func)
            self.composites_are_anomaly=as_anomaly
            self.lagged_means=lagged_means
        return composite

    #Aggregates variables over all time points where event is defined, regardless of its value
    def aggregate_variables(self,dim='time',lag_vals='all',con_func=agg.mean_ds,cat_func=agg.cat_occ_ds):
        
        """Calculates a summary metric from *LaggedAnalyser.variables* at all points where *LaggedAnalyser.event* is defined, regardless of its value.
        
        **Optional arguments**
        
        *dim*
        
        A string, the name of the shared coordinate between *LaggedAnalyser.variables* and *LaggedAnalyser.event*.
        
        *lag_vals*
        
        'all' or a iterable of integers, specifying for which lag values to compute the summary metric.
        
        *con_func*
        
        The summary metric to use for continuous variables. Defaults to a standard mean average. If None, then continuous variables will be ignored
            
        *cat_func*
        
        The summary metric to use for categorical variables. Defaults to the occurrence probability of each categorical value. If None, then continuous variables will be ignored

        **returns**
        
        An xarray.Dataset like  *LaggedAnalyser.variables* but summarised according to *con_func* and *cat_func*.

"""
        fake_event=self.event.copy(data=np.zeros_like(self.event))
        return self._compute_aggregate_over_lags(fake_event,dim,lag_vals,con_func,cat_func).isel(index_val=0)

    def add_derived_composite(self,name,func,composite_vars,as_anomaly=False):
        """Applies *func* to one or multiple composites to calculate composites of derived quantities, and additionally, stores *func* to allow derived bootstrap composites to be calculated. For linear quantities, where Ex[f(x)]==f(Ex[x]), then this can minimise redundant memory use.
        
        **Arguments**
        
        *name*
        
        A string, providing the name of the new variable to add.
            
        *func*
        
         A callable which must take 1 or more xarray.DataArrays as inputs
            
        *composite_vars*
        
        An iterable of strings, of the same length as the number of arguments taken by *func*. Each string must be the name of a variable in *LaggedAnalyser.variables* which will be passed into *func* in order.
        
        **Optional arguments**
        
        *as_anomaly*
        
        A boolean. Whether anomaly composites or full composites should be passed in to func.
        """
        
        if np.ndim(as_anomaly)==1:
            raise(NotImplementedError('variable-specific anomalies not yet implemented'))

        self._derived_variables[name]=(func,composite_vars,as_anomaly)
        self.composites[name]=self._compute_derived_da(self.composites,func,composite_vars,as_anomaly)
        
        if self.lagged_means is not None:
            self.lagged_means[name]=self._compute_derived_da(self.lagged_means,func,composite_vars,as_anomaly)
            
        return

    ### Compute bootstraps ###
    
    #Top level func
    def compute_bootstraps(self,bootnum,dim='time',con_func=agg.mean_ds,cat_func=agg.cat_occ_ds,lag=0,synth_mode='markov',data_vars=None,reuse_ixs=False):
        
        """Computes composites from synthetic event indices, which can be used to assess whether composites are insignificant.
        
        **Arguments**
        
        *bootnum*
        
        An integer, the number of bootstrapped composites to compute
            
        **Optional arguments**
        
        *dim*
        
        A string, the name of the shared coordinate between *LaggedAnalyser.variables* and *LaggedAnalyser.event*.
            
        *con_func*
        
        The summary metric to use for continuous variables. Defaults to a standard mean average. If None, then continuous variables will be ignored
            
        *cat_func*
        
        The summary metric to use for categorical variables. Defaults to the occurrence probability of each categorical value. If None, then continuous variables will be ignored

        *lag*
        
        An integer, specifying which lagged variables to use for the bootstraps. i.e. bootstraps for lag=90 will be from a completely different season than those for lag=0.
            
        *synth_mode*
        
        A string, specifying how synthetic event indices are to be computed. Valid options are:
            
        "random": 
        
        categorical values are randomly chosen with the same probability of occurrence as those found in *LaggedAnalyser.event*, but with no autocorrelation.

        "markov": 
        
        A first order Markov chain is fitted to *LaggedAnalyser.event*, producing some autocorrelation and state dependence in the synthetic series. Generally a better approximation than "random" and so should normally be used.

        "shuffle": 
        
        The values are randomly reordered. This means that each value will occur exactly the same amount of times as in the original index, and so is ideal for particularly rare events or short series.
            
        *data_vars*
        
        An iterable of strings, specifying for which variables bootstraps should be computed.
                
        **returns**
        
        An xarray.Dataset like *LaggedAnalyser.variables* but summarised according to *con_func* and *cat_func*, and with a new coordinate 'bootnum' of length *bootnum*.

        """
        if data_vars==None:
            data_vars=list(self.variables.data_vars)

        boots=self._add_derived_boots(self._compute_bootstraps(bootnum,dim,con_func,cat_func,lag,synth_mode,data_vars,reuse_ixs))
        if self.composites_are_anomaly:
            boots=boots-self.lagged_means.sel(lag=lag)
        return make_all_dims_coords(boots)
    
    
    def _compute_derived_da(self,ds,func,varnames,as_anomaly):
        if as_anomaly:
            input_vars=[ds[v]-self.lagged_means[v] for v in varnames]
        else:
            input_vars=[ds[v] for v in varnames]
        return make_all_dims_coords(func(*input_vars))
    
    
    def _add_derived_boots(self,boots):
        for var in self._derived_variables:
            func,input_vars,as_anomaly=self._derived_variables[var]
            boots[var]=self._compute_derived_da(boots,func,input_vars,as_anomaly)
        return boots

    def _compute_bootstraps(self,bootnum,dim,con_func,cat_func,lag,synth_mode,data_vars,reuse_ixs):

        da,ds=xr.align(self.event,self.lagged_variables(lag))
        ds=ds[data_vars]
        
        if (self.boot_indices is None)|(not reuse_ixs):
            
            ix_vals,ix_probs,L=self._get_bootparams(da)
            ixs=self._get_synth_indices(da.values,bootnum,synth_mode,da,dim,ix_vals,ix_probs,L)
            self.boot_indices=ixs
        else:
            ixs=self.boot_indices
            print('Reusing stored boot_indices, ignoring new boot parameters.')
        
        boots=[make_all_dims_coords(\
                self._composite_from_ix(ix,ds,dim,con_func,cat_func,lag)\
             ) for ix in ixs]
        return xr.concat(boots,'boot_num')
    
    #Gets some necessary variables
    def _get_bootparams(self,da):
        ix_vals,ix_probs=np.unique(da.values,return_counts=True)
        return ix_vals,ix_probs/len(da),len(da)
    
    #compute indices
    def _get_synth_indices(self,index,bootnum,mode,da,dim,ix_vals,ix_probs,L):
        
        ixs=[]
        if mode=='markov':
            xs=split_to_contiguous(da[dim].values,x_arr=da)
            T=get_transmat(xs)
            for n in range(bootnum):
                ixs.append(synthetic_states_from_transmat(T,L-1))
                
        elif mode=='random':
            for n in range(bootnum):
                ixs.append(np.random.choice(ix_vals,size=L,p=list(ix_probs)))
                
        elif mode=='shuffle':
            for n in range(bootnum):
                ixv=index.copy()
                np.random.shuffle(ixv)
                
                ixs.append(ixv)
        else:
            raise(ValueError(f'synth_mode={synth_mode} is not valid.'))
            
        return [xr.DataArray(ix) for ix in ixs]
        
    ### apply significance test ###
    
    def get_significance(self,bootstraps,comp,p,data_vars=None,hb_correction=False):
        
        """Computes whether a composite is significant with respect to a given distribution of bootstrapped composites. 
        
        **Arguments**
        
        *bootstraps*

        An xarray.Dataset with a coordinate 'bootnum', such as produced by *LaggedAnalyser.compute_bootstraps*

        *comp*

        An xarray Dataset of the same shape as *bootstraps* but without a 'bootnum' coordinate. Missing or additional variables are allowed, and are simply ignored.
        *p*

        A float, specifying the p-value of the 2-sided significance test (values in the range 0 to 1). 
            
        **Optional arguments**

        *data_vars*
            
        An iterable of strings, specifying for which variables significance should be computed.
            
        *hb_correction*
        
        A Boolean, specifying whether a Holm-Bonferroni correction should be applied to *p*, in order to reduce the family-wide error rate. Note that this correction is currently only applied to each variable in *comp* independently, and so will have no impact on scalar variables.
        
        **returns**
        
        An xarray.Dataset like *comp* but with boolean data, specifying whether each feature of each variable passed the significance test.
        """
        if data_vars==None:
            data_vars=list(bootstraps.data_vars)

        bootnum=len(bootstraps.boot_num)
        comp=comp[data_vars]
        bootstraps=bootstraps[data_vars]
        frac=(comp<bootstraps).sum('boot_num')/bootnum
        pval_ds=1-2*np.abs(frac-0.5)
        if hb_correction:
            for var in pval_ds:
                corrected_pval=holm_bonferroni_correction(pval_ds[var].values.reshape(-1),p)\
                            .reshape(pval_ds[var].shape)
                pval_ds[var].data=corrected_pval
        else:
            pval_ds=pval_ds<p
            
        self.composite_sigs=pval_ds.assign_coords(lag=comp.lag)
        return self.composite_sigs
    
    def bootstrap_significance(self,bootnum,p,dim='time',synth_mode='markov',reuse_lag0_boots=False,data_vars=None,hb_correction=False):
        
        """A wrapper around *compute_bootstraps* and *get_significance*, that calculates bootstraps and applies a significance test to a number of time lagged composites simulataneously.
        
    **Arguments**

    *bootnum*

    An integer, the number of bootstrapped composites to compute

    *p*

    A float, specifying the p-value of the 2-sided significance test (values in the range 0 to 1). 

    **Optional arguments**

    *dim*

    A string, the name of the shared coordinate between *LaggedAnalyser.variables* and *LaggedAnalyser.event*.

    *synth_mode*

    A string, specifying how synthetic event indices are to be computed. Valid options are:
    "random": categorical values are randomly chosen with the same probability of occurrence as those found in *LaggedAnalyser.event*, but with no autocorrelation.
    'markov': A first order Markov chain is fitted to *LaggedAnalyser.event*, producing some autocorrelation and state dependence in the synthetic series. Generally a better approximation than "random" and so should normally be used.

    *reuse_lag0_boots*
        A Boolean. If True, bootstraps are only computed for lag=0, and then used as a null distribution to assess all lagged composites. For variables which are approximately stationary across the lag timescale, then this is a good approximation and can increase performance. However if used incorrectly, it may lead to 'significant composites' which simply reflect the seasonal cycle. if False, separate bootstraps are computed for all time lags.

    *data_vars*
        An iterable of strings, specifying for which variables significance should be computed.

    *hb_correction*
        A Boolean, specifying whether a Holm-Bonferroni correction should be applied to *p*, in order to reduce the family-wide error rate. Note that this correction is currently only applied to each variable in *comp* independently, and so will have no impact on scalar variables.
        
    **returns**

    An xarray.Dataset like *LaggedAnalyser.variables* but with the *dim* dimension summarised according to *con_func* and *cat_func*, an additional *lag* coordinate, and with boolean data specifying whether each feature of each variable passed the significance test.

        """
        lag_vals=list(self._lagged_variables)
        
        con_func,cat_func=self.composite_func
        
        boots=self.compute_bootstraps(bootnum,dim,con_func,cat_func,0,synth_mode,data_vars)
        
        #reuse_lag0_boots=True can substantially reduce run time!
        if not reuse_lag0_boots:
                    boots=xr.concat([boots,*[self.compute_bootstraps(bootnum,dim,con_func,cat_func,t,synth_mode,data_vars)\
                        for t in lag_vals]],'lag').sortby('lag')
                
        sig_composite=self.get_significance(boots,self.composites,p,data_vars,hb_correction=hb_correction)
        
        self.composite_sigs=sig_composite
        return self.composite_sigs
    
    
    def deseasonalise_variables(self,variable_list=None,dim='time',agg='dayofyear',smooth=1,coeffs=None):
        """Computes a seasonal cycle for each variable in *LaggedAnalyser.variables* and subtracts it inplace, turning *LaggedAnalyser.variables* into deseasonalised anomalies. The seasonal cycle is computed via temporal aggregation of each variable over a given period - by default the calendar day of the year. This cycle can then be smoothed with an n-point rolling average.

                **Optional arguments**

                *variable_list*
                
                A list of variables to deseasonalise. Defaults to all variables in the *LaggedAnalyser.variables*

                *dim*
                
                A string, the name of the shared coordinate between *LaggedAnalyser.variables* and *LaggedAnalyser.event*, along which the seasonal cycle is computed. Currently, only timelike coordinates are supported.
                
                *agg*
                
                A string specifying the datetime-like field to aggregate over. Useful and supported values are 'season', 'month', 'weekofyear', and 'dayofyear'
                    
                *smooth*
                
                An integer, specifying the size of the n-timestep centred rolling mean applied to the aggregated seasonal cycle. By default *smooth*=1 results in no smoothing.

                *coeffs*
                
                A Dataset containing a precomputed seasonal cycle, which, if *LaggedAnalyser.variables* has coordinates (*dim*,[X,Y,...,Z]), has coords (*agg*,[X,Y,...,Z]), and has the same data variables as *LaggedAnalyser.variables*. If *coeffs* is provided, no seasonal cycle is fitted to *LaggedAnalyser.variables*, *coeffs* is used instead.

        """        

        if variable_list is None:
            variable_list=list(self.variables)
        for var in variable_list:
            da=self.variables[var]
            dsnlsr=Agg_Deseasonaliser()
            if coeffs is None:
                dsnlsr.fit_cycle(da,dim=dim,agg=agg)
            else:
                dsnslr.cycle_coeffs=coeffs[var]

            cycle=dsnlsr.evaluate_cycle(data=da[dim],smooth=smooth)
            self.variables[var]=da.copy(data=da.data-cycle.data)
            dsnlsr.data=None #Prevents excess memory storage
            self._deseasonalisers[var]=dsnlsr
        return   
    
    def get_seasonal_cycle_coeffs(self):
        """ Retrieve seasonal cycle coeffs computed with *LaggedAnalyser.deseasonalise_variables*, suitable for passing into *coeffs* in other *LaggedAnalyser.deseasonalise_variables* function calls as a precomputed cycle.
        
        **Returns**
        An xarray.Dataset, as specified in  the *LaggedAnalyser.deseasonalise_variables* *coeff* optional keyword.
        """
        coeffs=xr.Dataset({v:dsnlsr.cycle_coeffs for v,dsnlsr in self._deseasonalisers.items()})
        return coeffs

    #If deseasonalise_variables has been called, then this func can be used to compute the
    #seasonal mean state corresponding to a given composite. This mean state+ the composite
    # produced by self.compute_composites gives the full field composite pattern.
    def get_composite_seasonal_mean(self):
        """
        If *LaggedAnalyser.deseasonalise_variables* has been called, then this function returns the seasonal mean state corresponding to a given composite, given by a sum of the seasonal cycle weighted by the time-varying occurrence of each categorical value in *LaggedAnalyser.events*. This mean state + the deseasonalised anomaly composite
    produced by *LaggedAnalyser.compute_composites* then retrieves the full composite pattern.
    
    **Returns**
        An xarray.Dataset containing the composite seasonal mean values.
        """
        variable_list=list(self._deseasonalisers)
        ts={e:self.event[self.event==e].time for e in np.unique(self.event)}
        lags=np.unique([0,*list(self._lagged_variables)])
        
        mean_states={}
        for var in variable_list:
            dsnlsr=self._deseasonalisers[var]
            agg=dsnlsr.agg
            mean_states[var]=xr.concat([\
                                 xr.concat([\
                                    self._lag_average_cycle(dsnlsr,agg,l,t,i)\
                                for l in lags],'lag')\
                            for i,t in ts.items()],'index_val')
            
        return xr.Dataset(mean_states)
        
    def _lag_average_cycle(self,dsnlsr,agg,l,t,i):
        
        dt=durel.relativedelta(**{self.offset_unit:int(l)})
        tvals=pd.to_datetime([pd.to_datetime(tt)+dt for tt in t.values])
        cycle_eval=dsnlsr.cycle_coeffs.sel({agg:getattr(tvals,agg)})
        cycle_mean=cycle_eval.mean(agg).assign_coords({'lag':l,'index_val':i})
        return cycle_mean
    
class PatternFilter(object):
    """Provides filtering methods to refine n-dimensional boolean masks, and apply them to an underlying dataset.
    
        **Optional arguments:**
        
        *mask_ds*
        
        An xarray boolean Dataset of arbitrary dimensions which provides the initial mask dataset. If *mask_ds*=None  and *analyser*=None, then *mask_ds* will be initialised as a Dataset of the same dimensions and data_vars as *val_ds*, with all values = 1 (i.e. initially unmasked). 
        
        *val_ds*
        
        An xarray Dataset with the same dimensions as *mask_ds* if provided, otherwise arbitrary, consisting of an underlying dataset to which the mask is applied. If *val_ds*=None and *analyser*=None, then *PatternFilter.apply_value_mask* will raise an Error
            
        *analyser*
        
        An instance of a  core.LaggedAnalyser class for which both composites and significance masks have been computed, used to infer the *val_ds* and *mask_ds* arguments respectively. This overrides any values passed explicitly to  *mask_ds* and *val_ds*.
            
    """
    def __init__(self,mask_ds=None,val_ds=None,analyser=None):
        """Initialise a new PatternFilter object"""
        self.mask_ds=mask_ds
        """@private"""
        self.val_ds=val_ds
        """@private"""

        if analyser is not None:
            self._parse_analyser(analyser)
            
        else:
            if mask_ds is None:
                self.mask_ds=self._mask_ds_like_val_ds()
                
    def __repr__(self):
        return 'A PatternFilter object'
        
    def __str__(self):
            return self.__repr__
        
    def _parse_analyser(self,analyser):
        self.mask_ds=analyser.composite_sigs
        self.val_ds=analyser.composites
        
    def _mask_ds_like_val_ds(self):
        if self.val_ds is None:
            raise(ValueError('At least one of "mask_ds", "val_ds" and "analyser" must be provided.'))
        
        x=self.val_ds
        y=x.where(x!=0).fillna(1) #replace nans and 0s with 1
        y=(y/y).astype(int) #make everything 1 via division and assert integer type.
        self.mask_ds=y
        return
    
    def update_mask(self,new_mask,mode):
        """ Update *PatternFilter.mask_ds* with a new mask, either taking their union or intersection, or replacing the current mask with new_mask.
        
        **Arguments**
        
        *new_mask*

        An xarray.Dataset with the same coords and variables as *PatternFilter.mask_ds*.

        *mode*

        A string, one of 'replace','intersection' or 'union', defining how *new_mask* should be used to update the mask.
        """
        new_mask=new_mask.astype(int)
        if mode=='replace':
            self.mask_ds=new_mask
        elif mode=='intersection':
            self.mask_ds=self.mask_ds*new_mask
        elif mode == 'union':
            self.mask_ds=self.mask_ds|new_mask
        else:
            raise(ValueError(f'Invalid mode, {mode}'))
        return
                  
    def apply_value_mask(self,truth_function,*args,mode='intersection'):
        """ Apply a filter to *PatternFilter.mask_ds* based on a user-specified truth function which is applied to *PatternFilter.val_ds. 
        
        **Examples**
        
            #Mask values beneath a threshold:
            def larger_than_thresh(ds,thresh):
                return ds>thresh
            patternfilter.apply_value_mask(is_positive,thresh)

            #Mask values where absolute value is less than a reference field:
            def amp_greater_than_reference(ds,ref_ds):
                return np.abs(ds)>ref_ds
            pattern_filter.apply_value_mask(amp_greater_than_reference,ref_ds)

        **Arguments**

        *truth_function*
        
        A function with inputs (val_ds,*args) that returns a boolean dataset with the same coords and data variables as *PatternFilter.val_ds*.

        **Optional arguments**
        
        *mode*
            
        A string, one of 'replace','intersection' or 'union', defining how the value filter should be used to update the *PatternFilter.mask_ds*.
        """        
        if self.val_ds is None:
            raise(ValueError('val_ds must be provided to apply value mask.'))
        value_mask=truth_function(self.val_ds,*args)
        self.update_mask(value_mask,mode)
        return
    
    def apply_area_mask(self,n,dims=None,mode='intersection',area_type='gridpoint'):
        """ Apply a filter to *PatternFilter.mask_ds* that identifies connected groups of True values within a subspace of the Dataset's dimensions specified by *dims*, and masks out groups which are beneath a threshold size *n*. This is done through the application of *scipy.ndimage.label* using the default structuring element (https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html). 
    
        When *area_type*='gridpoint', *n* specifies the number of connected datapoints within each connected region. For the special case where *dims* consists of a latitude- and longitude-like coordinate, area_type='spherical' applies a cosine-latitude weighting, such that *n* can be interpreted as a measure of area, where a datapoint with lat=0 would have area 1. 
        
        **Examples**
        
            #Keep groups of True values consisting of an area >=30 square equatorial gridpoints
            patternfilter.apply_area_mask(30,dims=('lat','lon'),area_type='spherical')
            
            #Keep groups of True values that are consistent for at least 3 neighbouring time lags
            patternfilter.apply_area_mask(3,dims=('time'))
            
            #Keep groups of true values consisting of >=10 longitudinal values, or >=30 values in longitude and altitude if the variables have an altitude coord:
            patternfilter.apply_area_mask(10,dims=('longitude'))
            patternfilter.apply_area_mask(30,dims=('longitude,altitude'),mode='union')

        **Arguments**

        *n*
            
        A scalar indicating the minimum size of an unmasked group, in terms of number of gridpoints (for *area_type*=gridpoint) or the weighted area (for *area_type*=spherical), beneath which the group will be masked.

        **Optional arguments**
        
        *dims*
            
        An iterable of strings specifying coords in *PatternFilter.mask_ds* which define the subspace in which groups of connected True values are identified. Other dims will be iterated over. DataArrays within *PatternFilter.mask_ds* that do not contain all the *dims* will be ignored. If *dims*=None, all dims in each DataArray will be used.
            
        *mode*

        A string, one of 'replace','intersection' or 'union', defining how the area filter should be used to update the *PatternFilter.mask_ds*.
            
        *area_type*

        A string, one of 'gridpoint' or 'spherical' as specified above. 'spherical' is currently only supported for len-2 *dims* kwargs, with the first assumed to be latitude-like. 
            
        """        
        if area_type=='gridpoint':
            area_based=False
        elif area_type=='spherical':
            area_based=True
        else:
            raise(ValueError(f"Unknown area_type {area_type}. Valid options are 'gridpoint' and 'spherical'"))
        area_mask=ds_large_regions(self.mask_ds,n,dims=dims,area_based=area_based)
        self.update_mask(area_mask,mode)
        return
    
    
    def apply_convolution(self,n,dims,mode='replace'):
        """ Apply a square n-point convolution filter to *PatternFilter.mask_ds* in one or two dimensions specified by *dims*, iterated over remaining dimensions. This has the effect of extending the unmasked regions and smoothing the mask overall.
        
        **Arguments**
        
        *n*
            
        A positive integer specifying the size of the convolution filter. *n*=1 leaves the mask unchanged. Even *n* are asymmetric and shifted right. 

        *dims*

        A length 1 or 2 iterable of strings specifying the dims in which the convolution is applied. Other dims will be iterated over. DataArrays within *PatternFilter.mask_ds* that do not contain all the *dims* will be ignored. 

        *mode*

        A string, one of 'replace','intersection' or 'union', defining how the area filter should be used to update the *PatternFilter.mask_ds*.
        """
        
        if not len(dims) in [1,2]:
            raise(ValueError('Only 1 and 2D dims currently supported'))
            
        convolution=convolve_pad_ds(self.mask_ds,n,dims=dims)
        self.update_mask(convolution,mode)
        return
    
    def get_mask(self):
        """" Retrieve the mask with all filters applied.
        **Returns**
        An xarray.Dataset of boolean values.
        """
        return self.mask_ds
    
    def filter(self,ds=None,drop_empty=True,fill_val=np.nan):
        """ Apply the current mask to *ds* or to *PatternFilter.val_ds* (if *ds* is None), replacing masked gridpoints with *fill_val*.
        **Optional arguments**
        
        *ds*
        
        An xarray.Dataset to apply the mask to. Should have the same coords and data_vars as *PatternFilter.mask_ds*. If None, the mask is applied to *PatternFilter.val_ds*.
        
        *drop_empty*
        
        A boolean value. If True, then completely masked variables are dropped from the returned masked Dataset.
        
        *fill_val*
        
        A scalar that defaults to np.nan. The value with which masked gridpoints in the Dataset are replaced.
        
        **Returns**
        
        A Dataset with masked values replaced by *fill_val*.
        """
        if ds is None:
            ds=self.val_ds.copy(deep=True)
            
        ds=ds.where(self.mask_ds)
        if drop_empty:
            drop_vars=((~np.isnan(ds)).sum()==0).to_array('vars')
            ds=ds.drop_vars(drop_vars[drop_vars].vars.values)
        return ds.fillna(fill_val)
    
def _DEFAULT_RENAME_FUNC(v,d):
    
    for k,x in d.items():
        v=v+f'_{k}{x}'
    return v
    
def _Dataset_to_dict(ds):
    return {v:d['data'] for v,d in ds.to_dict()['data_vars'].items()}

class IndexGenerator(object):
    
    """ Computes dot-products between a Dataset of patterns and a Dataset of variables, reducing them to standardised scalar indices.
    """
    def __init__(self):
        self._means=[]
        self._stds=[]
        self._rename_function=_DEFAULT_RENAME_FUNC
        
    def __repr__(self):
        return 'An IndexGenerator object'
        
    def __str__(self):
            return self.__repr__
    
    
    def centre(self,x,dim='time',ref=None):
        """@private"""

        if ref is None:
            ref=x.mean(dim=dim)
        return x-ref
    
    def normalise(self,x,dim='time',ref=None):
        """@private"""

        if ref is None:
            ref=x.std(dim=dim)
        return x/ref
    
    def standardise(self,x,dim='time',mean_ref=None,std_ref=None):
        """@private"""
        centred_x=self.centre(x,dim,mean_ref)
        standardised_x=self.normalise(centred_x,dim,std_ref)
        return standardised_x
        
    def collapse_index(self,ix,dims):
        """@private"""
        lat_coords=['lat','latitude','grid_latitude']
        if not np.any(np.isin(lat_coords,dims)):
            return ix.sum(dims)
        
        else:
            #assumes only one lat coord: seems safe.
            lat_dim=lat_coords[np.where(np.isin(lat_coords,dims))[0][0]]
            weights=np.cos(np.deg2rad(ix[lat_dim]))
            return ix.weighted(weights).sum(dims)
            
    def generate(self,pattern_ds,series_ds,dim='time',slices=None,ix_means=None,ix_stds=None,drop_blank=False,in_place=True,strict_metadata=False):
        """Compute standardised indices from an xarray.Dataset of patterns and an xarray.Dataset of arbitrary dimension variables.
        
        **Arguments**
        
        *pattern_ds*
        
        An xarray.Dataset of patterns to project onto with arbitrary dimensions.
        
        *series_ds*
        
        An xarray.Dataset of variables to project onto the patterns. Coordinates of *series_ds* once subsetted using *slices* must match the dimensions of *pattern_ds* + the extra coord *dim*.
        
        **Optional arguments**
        
        *dim*:
        
        A string specifying the remaining coord of the scalar indices. Defaults to 'time', which should be the choice for most use cases.
        
        *slices*
        
        A dictionary or iterable of dictionaries, each specifying a subset of *pattern_ds* to take before computing an index, with one index returned for each dictionary and for each variable. Subsetting is based on the *xr.Dataset.sel* method: e.g. *slices*=[dict(lag=0,index_val=1)] will produce 1 set of indices based on pattern_ds.sel(lag=0,index_val=1). If *slices*=None, no subsets are computed.
        
        *ix_means*
        
        If None, the mean of each index is calculated and subtracted, resulting in centred indices. Otherwise, *ix_means* should be a dictionary of index names and predefined mean values which are subtracted instead. Of most use for online computations, updating a precomputed index in a new dataset.
        
        *ix_stds*
        
        If None, the standard deviation of each index is calculated and is divided by, resulting in standardised indices. Otherwise, *ix_stds* should be a dictionary of index names and predefined std values which are divided by instead. Of most use for online computations, updating a precomputed index in a new dataset.

        *drop_blank*
        
        A boolean. If True, drop indices where the corresponding pattern is entirely blank. If False, returns an all np.nan time series.
        *in_place*
        
        *strict_metadata*
        
        If False, indices will be merged into a common dataset regardless of metadata. If True, nonmatching metadata will raise a ValueError.
        
        **Returns
        
        An xarray.Dataset of indices with a single coordinate (*dim*).
        """
        #Parse inputs
        
        if slices is None:
            self.slices=[{}]
        elif type(slices) is dict:
            self.slices=[slices]
        else:
            self.slices=slices
            
        if ix_means is not None or ix_stds is not None:
            self.user_params=True
            self.means=ix_means
            self.stds=ix_stds
        else:
            self.user_params=False
            self.means={}
            self.stds={}
            
        self.indices=None
        
        #Compute indices
        indices=[self._generate_index(pattern_ds,series_ds,dim,sl)\
                for sl in self.slices]
        try:
            indices=xr.merge(indices)
        except Exception as e:
            if strict_metadata:
                print("Merge of indices failed. Consider 'strict_metadata=False'")
                raise e
            else:
                indices=xr.merge(indices,compat='override')
            
        #Optionally remove indices which are all nan    
        if drop_blank:
            drop=(~indices.isnull()).sum()==0
            drop=[k for k,d in drop.to_dict()['data_vars'].items() if d['data']]
            indices=indices.drop_vars(drop)
            _=[(self.means.pop(x),self.stds.pop(x)) for x in drop]
        if in_place:
            self.indices=indices
        return indices
    
    def _generate_index(self,pattern_ds,series_ds,dim,sl):
                
        pattern_ds,series_ds=xr.align(pattern_ds,series_ds)
        pattern_ds=pattern_ds.sel(sl)
        dims=list(pattern_ds.dims)

        index=pattern_ds*series_ds
        #coslat weights lat coords
        index=self.collapse_index(index,dims)
        index=self._rename_index_vars(index,sl)

        if self.user_params:
            mean=self.means
            std=self.stds
        else:
            mean=_Dataset_to_dict(index.mean(dim))
            std=_Dataset_to_dict(index.std(dim))
            for v in mean:
                self.means[v]=mean[v]
            for v in std:
                self.stds[v]=std[v]
                
        index=self.standardise(index,dim,mean_ref=mean,std_ref=std)
        index=self._add_index_attrs(index,sl,mean,std)

        
        self.generated_index=index
        return index
    
    def _add_index_attrs(self,index,sl,mean,std):
        for v in index:
            ix=index[v]
            ix.attrs['mean']=np.array(mean[v])
            ix.attrs['std']=np.array(std[v])
            for k,i in sl.items():
                ix.attrs[k]=i
            index[v]=ix
        return index
    
    def _rename_index_vars(self,index,sl):
        func=self._rename_function
        return index.rename({v:func(v,sl) for v in index.data_vars})
    
    def get_standardisation_params(self,as_dict=False):
        
        """ Retrieve index means and stds for computed indices, for use as future inputs into index_means or index_stds in *IndexGenerator.Generate*
        """
        if as_dict:
            return self.means,self.stds
        else:
            params=[xr.Dataset(self.means),xr.Dataset(self.stds)]
            return xr.concat(params,'param').assign_coords({'param':['mean','std']})
