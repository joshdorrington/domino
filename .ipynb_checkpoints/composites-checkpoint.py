import numpy as np
import datetime as dt
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from util import holm_bonferroni_correction, offset_time_dim, is_two_valued,split_to_contiguous,is_time_type,make_all_dims_coords,drop_scalar_coords
from regime import xr_reg_occurrence,get_transmat,synthetic_states_from_transmat
from scores import BSS
from sklearn.linear_model import LogisticRegression

def aggregate_ds(ds,dim,agg_func,**agg_kwargs):
    return ds.map(agg_func,dim=dim,**agg_kwargs)
def mean_ds(ds,dim):
    return aggregate_ds(ds,dim,lambda da,dim: da.mean(dim=dim))
def std_ds(ds,dim):
    return aggregate_ds(ds,dim,lambda da,dim: da.std(dim=dim))
def cat_occ_ds(ds,dim,reg_ds):
    return aggregate_ds(ds,dim,xr_reg_occurrence,s=reg_ds,coord_name='variable_cat_val')

class LaggedAnalyser(object):
    """Analysis of lagged composites defined with respect to a categorical event series"""
    
    def __init__(self,event,variables=None,name=None,is_categorical=None):
        """Create a LaggedAnalyser object.
        
        **Arguments:**
        *event*
            An xarray.DataArray with one dimension taking on categorical values, each defining a class of event (or non-event).
            
        **Optional arguments**
        *variables*,*name*,*is_categorical*
            Arguments for adding variables to the LaggedAnalyser. Identical behaviour to calling add_variables directly.
        """
        
        #event is a dataarray
        self.event=xr.DataArray(event)
        
        #variables are stored in a dataset, and can be added later,
        #or passed as a DataArray, a Dataset or as a dict of DataArrays
        self.variables=xr.Dataset(coords=event.coords)
        if variables is not None:
            self.add_variable(variables,name,is_categorical,False)
        
        #Time lagged versions of the dataset self.variables will be stored here, with a key
        #equal to the lag applied. Designed to be accessed by the self.lagged_variables function
        self._lagged_variables={}
        
        #variables that are a linear combination of other variables are more efficiently
        #computed after compositing using the self.add_derived_composite method
        self._derived_variables={}
        
        self.composite_mask=None
        
        return
    
    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        l1='A Precursorself object\n'
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
    
    def _more_mergable(self,ds):
        
        return drop_scalar_coords(make_all_dims_coords(ds))
    
    def _add_variable(self,da,name,is_categorical,overwrite,join_type):
        
        if (name in self.variables)&(not overwrite):
            raise(KeyError(f'Key "{name}" is already in variables.'))
        
        try:
            self.variables=self.variables.merge(da.to_dataset(name=name),join=join_type)
        except:
            #Trying to make the merge work:
            self.variables=self._more_mergable(self.variables).merge(self._more_mergable(da.to_dataset(name=name)),join=join_type)

        if (is_categorical is None) and (not 'is_categorical' in da.attrs):
            self.variables[name].attrs['is_categorical']=0
        elif is_categorical is not None:
            self.variables[name].attrs['is_categorical']=is_categorical
       
    
    def add_variable(self,variables,name=None,is_categorical=None,overwrite=False,join_type='outer'):
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        
        *variables* 
            An xarray.DataArray, xarray.Dataset or dictionary of xarray.DataArrays, containing data to be composited with respect to *event*. One of the coordinates of *variables* should have the same name as the coordinate of *events*. Stored internally as an xarray.Dataset. If a dictionary is passed, the DataArrays are joined according to the method 'outer'.
            
        **Optional Arguments**
        
        *name* 
            A string. If *variables* is a single xarray.DataArray then *name* will be used as the name of the array in the LaggedAnalyser.variables DataArray. Otherwise ignored.
        
        *is_categorical* 
            An integer, if *variables* is an xarray.DataArray, or else a dictionary of integers with keys corresponding to DataArrays in the xarray.Dataset/dictionary. 0 indicates that the variable is continuous, and 1 indicates that it is categorical. Note that continuous and categorical variables are by default composited differently (see LaggedAnalyser.compute_composites). Default assumption is all DataArrays are continuous, unless a DataAarray contains an 'is_categorical' key in its DataArray.attrs, in which case this value is used.
            
        *overwrite*
            A boolean. If False then attempts to assign a variable who's name is already in *LaggedAnalyser.variables* will result in a ValueError
        
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
        
        **Examples**
        
        NEED SOME EXAMPLES
        """

        input_composites=[self.composites[var] for var in composite_vars]
        if as_anomaly:
            input_composites=[comp-self.composite_means[var]\
                    for comp,var in zip(input_composites,composite_vars)]
        elif np.ndim(as_anomaly)==1:
            raise(NotImplementedError('variable-specific anomalies not yet implemented'))

        self.composites[name]=func(*input_composites)
        self._derived_variables[name]=(func,composite_vars)
        return
    
    #A convenience wrapper on top of xr.align
    def align(self,**kwargs):
        """A convenience wrapper around xarray.align."""
        
        aligned_variables,aligned_event=xr.align(self.variables,self.event,**kwargs)
        self.variables=aligned_variables
        self.event=aligned_event
        return 
    
    def lagged_variables(self,t):
        """A convenience function that retrieves variables at lag *t* from the *LaggedAnalyser*"""
        if t in self._lagged_variables:
            return self._lagged_variables[t]
        elif t==0:
            return self.variables
        else:
            raise(KeyError(f'Lag {t} is not in self._lagged_variables.'))
            
    def _lag_variables(self,offset,offset_unit='days',offset_dim='time',mode='any',overwrite=False):
        
        if (offset in self._lagged_variables)&(not overwrite):
            raise(KeyError(f'Key "{offset}" is already in lagged_variables.'))
            
        #We are really paranoid about mixing up our lags. So we implement this safety check
        self._check_offset_is_valid(offset,mode)
        
        #The meat of the function
        time_offset=dt.timedelta(**{offset_unit:offset})
        new_dim=pd.to_datetime(self.variables[offset_dim])+time_offset
        self._lagged_variables[offset]=self.variables.copy(deep=False)
        self._lagged_variables[offset][offset_dim]=new_dim
        return
    
    #For coords not in a time format
    def _ilag_variables(self,offset,*args,overwrite=False):
        raise(NotImplementedError('Only timelike dimensions are currently supported.'))
        
    def lag_variables(self,offsets,offset_unit='days',offset_dim='time',mode='any',overwrite=False):
        """Produces time lags of *LaggedAnalyser.variables* which can be used to produce lagged composites.
        
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
        
        lag_funcs=[self._ilag_variables,self._lag_variables]
        offsets=np.atleast_1d(offsets)
        for o in offsets:
            lag_funcs[time_type](int(o),offset_unit,offset_dim,mode,overwrite)
            
    def _check_offset_is_valid(self,offset,mode):
        
        valid_modes=['any','past','future']
        if not mode in valid_modes:
            raise(ValueError(f'mode must be one of {valid_modes}'))
        if offset<0 and mode == 'past':
            raise(ValueError(f'Negative offset {offset} given, but mode is "{mode}"'))
        if offset>0 and mode == 'future':
            raise(ValueError(f'Positive offset {offset} given, but mode is "{mode}"'))
        return
    
    def dropna(self,dim,event=True,variables=False,**kwargs):
        
        """A convenience wrapper to xarray.DataArray.dropna/xarray.Dataset.dropna
        """
        if event:
            self.event=self.event.dropna(dim,**kwargs)
        if variables:
            self.variables=self.variables.dropna(dim,**kwargs)
          
    ### COMPOSITE COMPUTATION FUNCTIONS ###
    #Composite computation is split over 4 function layers:
    #compute_composites(): calls
    #    _compute_aggregate_over_lags(): calls
    #        _composite_from_ix(): splits data into cat vars
    #        and con vars and then calls
    #            _aggregate_from_ix(): applies an operation to
    #            subsets of ds, where the ix takes unique values
    #        then merges them together.
    #     then loops over lags and merges those.
    #And then substracts any anomalies and returns the data.
    
    
    #Example usage of the aggregate funcs:
    #i.e. self._aggregate_from_ix(ds,ix,'time',self._mean_ds)
    #self._aggregate_from_ix(ds,ix,'time',self._std_ds)
    #self._aggregate_from_ix(ds,ix,'time',self._cat_occ_ds,s=reg_ds)
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
    def compute_composites(self,dim='time',lag_vals='all',as_anomaly=False,con_func=mean_ds,cat_func=cat_occ_ds,inplace=True):
        
        """
        Partitions *LaggedAnalyser.variables*, and time-lagged equivalents, into subsets depending on the value of *LaggedAnalyser.event*, and then computes a bulk summary metric for each.

        **Optional arguments**
        
        *dim*
            A string, the coordinate along which to compute composites.
            
        *lag_vals*
            Either 'All', or a list of integers, denoting the time lags for which composites should be computed.
            
        *as_anomaly*
            A Boolean, defining whether composites should be given as absolute values or differences from the unpartitioned value.
            
        *con_func*
            The summary metric to use for continuous variables. Defaults to a standard mean average. If None, then continuous variables will be ignored
            
        cat_func*
            The summary metric to use for categorical variables. Defaults to the occurrence probability of each categorical value. If None, then continuous variables will be ignored
            
        *inplace*
            A boolean, defining whether the composite should be stored in *LaggedAnalyser.composites*
        
        **returns**
        
        *composite*
            An xarray.Dataset like  *LaggedAnalyser.variables* but summarised according to *con_func* and *cat_func*, and with an additional coordinate *index_val*, which indexes over the values taken by *LaggedAnalyser.event*.
            
        """
        composite=self._compute_aggregate_over_lags(self.event,dim,lag_vals,con_func,cat_func)
        
        if as_anomaly:
            lagged_means=self.aggregate_variables(dim,lag_vals,con_func,cat_func)
            composite=composite-lagged_means
            
        composite=make_all_dims_coords(composite)
        
        if inplace:
            self.composites=composite
            self.composites_are_anomaly=as_anomaly
            if as_anomaly:
                self.lagged_means=lagged_means

        return composite

    #Aggregates variables over all time points where event is defined, regardless of its value
    def aggregate_variables(self,dim='time',lag_vals='all',con_func=mean_ds,cat_func=cat_occ_ds):
        
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
        """
        fake_event=self.event.copy(data=np.zeros_like(self.event))
        return self._compute_aggregate_over_lags(fake_event,dim,lag_vals,con_func,cat_func).isel(index_val=0)


    ### Compute bootstraps ###
    
    #Top level func
    def compute_bootstraps(self,bootnum,dim='time',con_func=mean_ds,cat_func=cat_occ_ds,lag=0,synth_mode='markov',data_vars=None):
        
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
        """
        if data_vars==None:
            data_vars=list(self.variables.data_vars)

        boots=self._add_derived_boots(self._compute_bootstraps(bootnum,dim,con_func,cat_func,lag,synth_mode,data_vars))
        if self.composites_are_anomaly:
            boots=boots-self.lagged_means.sel(lag=lag)
        return make_all_dims_coords(boots)
    
    def _add_derived_boots(self,boots):
        for var in self._derived_variables:
            func,input_vars=self._derived_variables[var]
            boots[var]=func(*[boots[v] for v in input_vars])
        return boots

    def _compute_bootstraps(self,bootnum,dim,con_func,cat_func,lag,synth_mode,data_vars):

        da,ds=xr.align(self.event,self.lagged_variables(lag))
        ix_vals,ix_probs,L=self._get_bootparams(da)
        ds=ds[data_vars]
        ixs=self._get_synth_indices(bootnum,synth_mode,da,dim,ix_vals,ix_probs,L)
        boots=[self._composite_from_ix(ix,ds,dim,con_func,cat_func,lag) for ix in ixs]
        return xr.concat(boots,'boot_num')
    
    #Gets some necessary variables
    def _get_bootparams(self,da):
        ix_vals,ix_probs=np.unique(da.values,return_counts=True)
        return ix_vals,ix_probs/len(da),len(da)
    
    #compute indices
    def _get_synth_indices(self,bootnum,mode,da,dim,ix_vals,ix_probs,L):
        
        ixs=[]
        if mode=='markov':
            xs=split_to_contiguous(da[dim].values,x_arr=da)
            T=get_transmat(xs)
            for n in range(bootnum):
                ixs.append(synthetic_states_from_transmat(T,L-1))
                
        elif mode=='random':
            for n in range(bootnum):
                ixs.append(np.random.choice(ix_vals,L,ix_probs))
        else:
            raise(ValueError(f'synth_mode={synth_mode} is not valid.'))
            
        return [xr.DataArray(ix) for ix in ixs]
        
    ### apply significance test ###
    
    def get_significance(self,bootstraps,comp,p,data_vars=None,hb_correction=True):
        
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
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
            
        return pval_ds.assign_coords(lag=comp.lag)    
    
    def bootstrap_significance(self,bootnum,p,dim='time',synth_mode='markov',reuse_lag0_boots=False,con_func=mean_ds,cat_func=cat_occ_ds,data_vars=None,hb_correction=False):
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
        """
        lag_vals=list(self._lagged_variables)

        boots=self.compute_bootstraps(bootnum,dim,con_func,cat_func,0,synth_mode,data_vars)
        
        #reuse_lag0_boots=True can substantially reduce run time!
        if not reuse_lag0_boots:
                    boots=xr.concat([boots,*[self.compute_bootstraps(bootnum,dim,con_func,cat_func,t,synth_mode,data_vars)\
                        for t in lag_vals]],'lag').sortby('lag')
                
        sig_composite=self.get_significance(boots,self.composites,p,data_vars,hb_correction=hb_correction)
        
        self.composite_sigs=sig_composite
        return self.composite_sigs
    

    def compute_variable_mask(self,min_amp=0,min_dim_extents={},data_vars=None,drop_allnan_composites=False,skip_sig_test=False):
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
        """
        if data_vars is None:
            data_vars=list(self.composites.data_vars)

        #We consider as significant points that are not nan...
        is_sig=~np.isnan(self.composites[data_vars])

        # ...and optionally, that pass a bootstrap significance test
        if not skip_sig_test:
            is_sig=is_sig*self.composite_sigs[data_vars]

        #...and optionally, that have an amplitude exceeding some std threshold
        if min_amp!=0:
            is_sig=is_sig*self._is_min_amplitude(min_amp)

        #... and finally that has some minimum extent along certain dimensions:
        is_sig=self._set_shortlived_1s_in_ds_to_0(is_sig,min_dim_extents)

        if self.composite_mask is None:
            self.composite_mask=is_sig
        else:
            self.composite_mask=self.composite_mask.merge(is_sig)
        if drop_allnan_composites:
            self._drop_allnan_composites()
        return self.composite_mask

    def _is_min_amplitude(self,min_amp):
        stds=self.aggregate_variables(con_func=std_ds,cat_func=None)
        where_true=self.composites/stds
        #Assume categorical variables automatically pass this test
        cat_vars=xr.ones_like(self.composites).astype(bool).drop_vars(where_true.data_vars)
        return (np.abs(where_true)>min_amp).merge(cat_vars)

    def _set_shortlived_1s_in_1darr_to_v(self,arr,nmin,v):
        idx=np.hstack([0,np.where(np.diff(arr)!=0)[0]+1,len(arr)])
        ixlist=[arr[i:j] for i,j in zip(idx, idx[1:])]
        return np.hstack([i if len(i)>=nmin else np.ones_like(i)*v for i in ixlist])

    def _set_shortlived_1s_in_arr_to_v(self,arr,nmin,v,axis):
        A=np.moveaxis(arr,axis,0)
        vals=np.array([self._set_shortlived_1s_in_1darr_to_v(a,nmin,0) for a in A.reshape([arr.shape[axis],-1]).T])
        return np.moveaxis(vals.T.reshape(A.shape),0,axis)    

    def _set_shortlived_1s_in_da_to_0(self,da,nmin,dim):
        dim_axis=np.where(np.array(da.dims)==dim)[0]
        if dim_axis.size:
            return da.copy(data=self._set_shortlived_1s_in_arr_to_v(da.values,nmin,0,dim_axis[0]))
        else:
            return da

    def _set_shortlived_1s_in_ds_to_0(self,ds,nmin_dict):

        for dim,nmin in nmin_dict.items():
            ds=ds.map(self._set_shortlived_1s_in_da_to_0,args=[nmin,dim])

        return ds.astype(bool)

    def _drop_allnan_composites(self):
        mask=self.composite_mask
        for v in list(mask.data_vars):
            if (v in list(self.composites.data_vars)) and (np.sum(mask[v].values)==0):
                self.composites=self.composites.drop(v)
        return
        
    ### GENERATE INDEX ###
        
    def generate_variable_index(self,variables,dim='time',min_amp=None,min_stdev=None,cat_val=0,lag=0,add=True):
        """Adds an additional variable to LaggedAnalyser.variables.
        
        **Arguments**
        *variables*
        
        **Optional arguments**
        
        **returns**
        
        **Examples**
        """
        if (not min_amp is None) and (not min_stdev is None):
            raise(ValueError('Only one of min_amp or min_stdev should be provided'))

        comp=self.composites[variables].sel(cat_val=cat_val,lag=lag).copy(deep=True)
        sig=self.composite_sigs[variables].sel(cat_val=cat_val,lag=lag)
        ds=self.variables[variables]

        for var in sig:
            comp[var].values[~sig[var].values]=np.nan
        ds_anom=ds-self.get_variable_means()[variables]

        #Additionally mask out any gridpoints with small anomalies
        if min_stdev is not None:
            min_amp=ds.std(dim)
        if min_amp is not None:
            small_anoms=np.abs(comp)<min_amp
            for var in small_anoms:
                comp[var].values[small_anoms[var].values]=np.nan

        dim_list=[d for d in list(ds.dims) if d!=dim]
        indices=(ds_anom*comp).mean(dim_list)

        if add:
            for var in indices:
                self.add_variable(f'{var}_l{lag}m{cat_val}_ix',indices[var],is_cat=0)
        return indices
