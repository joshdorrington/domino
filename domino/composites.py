import numpy as np
import datetime as dt
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from util import holm_bonferroni_correction, offset_time_dim, standardise, is_two_valued,split_to_contiguous,is_time_type,make_all_dims_coords,drop_scalar_coords,squeeze_da
from regime import xr_reg_occurrence,get_transmat,synthetic_states_from_transmat
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
    """Analysis of lagged composites defined with respect to a categorical event series
    
        **Arguments:**
        
        *event*
            An xarray.DataArray with one dimension taking on categorical values, each defining a class of event (or non-event).
            
        **Optional arguments**
        
        *variables, name, is_categorical*
        
            Arguments for adding variables to the LaggedAnalyser. Identical behaviour to calling add_variables directly.
"""
    
    def __init__(self,event,variables=None,name=None,is_categorical=None):
        """Create a LaggedAnalyser object."""
        
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
        self.lagged_means=None
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
    
    
    #TODO: Add pad align
    def align(self,**kwargs):
        """A convenience wrapper equivalent to calling xr.align(LaggedAnalyser.variables,LaggedAnalyser.event,**kwargs)."""
        
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
        
        if offset==0:
            return
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
            
        *cat_func*
            The summary metric to use for categorical variables. Defaults to the occurrence probability of each categorical value. If None, then continuous variables will be ignored
            
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
    def aggregate_variables(self,dim='time',lag_vals='all',con_func=mean_ds,cat_func=cat_occ_ds):
        
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
        self.composites[name]=self.compute_derived_da(self.composites,func,composite_vars,as_anomaly)
        
        if self.lagged_means is not None:
            self.lagged_means[name]=self.compute_derived_da(self.lagged_means,func,composite_vars,as_anomaly)
            
        return

    ### Compute bootstraps ###
    
    #Top level func
    def compute_bootstraps(self,bootnum,dim='time',con_func=mean_ds,cat_func=cat_occ_ds,lag=0,synth_mode='markov',data_vars=None):
        
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
            "random": categorical values are randomly chosen with the same probability of occurrence as those found in *LaggedAnalyser.event*, but with no autocorrelation.
            'markov': A first order Markov chain is fitted to *LaggedAnalyser.event*, producing some autocorrelation and state dependence in the synthetic series. Generally a better approximation than "random" and so should normally be used.
            
        *data_vars*
            An iterable of strings, specifying for which variables bootstraps should be computed.
                
        **returns**
            An xarray.Dataset like *LaggedAnalyser.variables* but summarised according to *con_func* and *cat_func*, and with a new coordinate 'bootnum' of length *bootnum*.

        """
        if data_vars==None:
            data_vars=list(self.variables.data_vars)

        boots=self._add_derived_boots(self._compute_bootstraps(bootnum,dim,con_func,cat_func,lag,synth_mode,data_vars))
        if self.composites_are_anomaly:
            boots=boots-self.lagged_means.sel(lag=lag)
        return make_all_dims_coords(boots)
    
    
    def compute_derived_da(self,ds,func,varnames,as_anomaly):
        if as_anomaly:
            input_vars=[ds[v]-self.lagged_means[v] for v in varnames]
        else:
            input_vars=[ds[v] for v in varnames]
        return make_all_dims_coords(func(*input_vars))
    
    
    def _add_derived_boots(self,boots):
        for var in self._derived_variables:
            func,input_vars,as_anomaly=self._derived_variables[var]
            boots[var]=self.compute_derived_da(boots,func,input_vars,as_anomaly)
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
    
    def get_significance(self,bootstraps,comp,p,data_vars=None,hb_correction=False):
        
        """Computes whether a composite is significant with respect to a given distribution of bootstrapped composites.
        
        **Arguments**
        
            *bootstraps*
                An xarray.Dataset with a coordinate 'bootnum'
                
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
            
        return pval_ds.assign_coords(lag=comp.lag)    
    
    def bootstrap_significance(self,bootnum,p,dim='time',synth_mode='markov',reuse_lag0_boots=False,data_vars=None,hb_correction=False):
        
        """A wrapper around *compute_bootstraps* and *test_significance*, that calculates bootstraps and applies a significance test to a number of time lagged composites simulataneously.
        
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
    

    def compute_variable_mask(self,min_amp=0,min_dim_extents={},data_vars=None,drop_allnan_composites=False,skip_sig_test=False,inplace=True,overwrite=False):
        """Computes a Boolean mask to filter out features of variables with little relation to the event index. This is done through a flexible combination of statistical significance testing, applying a minimum amplitude threshold, and requiring unmasked features to span a particular spatial or temporal scale.
        
        **Optional arguments**
        
        *min_amp*
            A float, specifying a minimum absolute amplitude below which composite anomalies are masked. *min_amp* is intepreted as a fraction of the standard deviation of each variable, calculated along the shared coordinate between *LaggedAnalyser.event* and *LaggedAnalyser.variables*, at all points where *LaggedAnalyser.event* is defined.
            
        *min_dim_extents*
            A dictionary in the format {str:int} where keys specify coordinate names in *LaggedAnalyser.composites*, and values specify a minimum length threshold for each coordinate. Unmasked features (as defined by the amplitude test and/or the significant test) for a variable are masked anyway if they are not part of a contiguous sequence of unmasked anomalies that meet the minimum length threshold along each dimension. E.g. *min_dim_extents* ={'lat':5,'lon':5} would mean any unmasked features in a field variable that did not span 5 grid points in both the latitudinal and longitudinal direction, would be masked.
            
        *data_vars*
            An iterable of strings, specifying for which variables a mask should be computed.
        
        *drop_allnan_composites*
            A boolean, specifying whether variables which are masked at all coordinate values should be dropped FROM WHAT THOUGH? DOUBLE CHECK.
            
        *skip_sig_test*
            If True, no signifance test is applied, and filtering is done solely based on amplitude and extent.
        
        *inplace*
            A boolean, specifying whether the output should be saved to *LaggedAnalyser.composite_masks*
        
        *overwrite*
            A boolean, which if *inplace*=True, specifies whether overwriting an existing *LaggedAnalyser.composite_masks* is allowed.
            
        **returns**
            
            An xarray.Dataset like *LaggedAnalyser.variables* but with the *dim* dimension summarised according to *con_func* and *cat_func*, an additional *lag* coordinate, and with boolean data specifying whether each feature of each variable is masked or not.
            
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
        
        if inplace:
            if (self.composite_mask is None) or overwrite:
                self.composite_mask=is_sig
            else:
                self.composite_mask=self.composite_mask.merge(is_sig)
        if drop_allnan_composites:
            self._drop_allnan_composites()
        return is_sig

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
        
    def generate_variable_index(self,data_var,mask,dim='time',sel={},inplace=True,name=None):
        """Computes standardised scalar indices from masked composites, by taking a dot product between corresponding variable and composite DataArrays.
        
        **Arguments**
            
        *data_var*
            A string, specifying which variable should be used to compute the index.
        *mask*
            A Dataset with boolean entries. Only features of data_var where mask=True will be included in the dot product.
            
        **Optional arguments**
            
        *dim*
            A string, specifying the name of the shared dimension between *LaggedAnalyser.variables* and LaggedAnalyser.event*
            
        *sel*
            A dictionary whose keys should be coordinate names, and whose elements should be valid coordinate values, or ranges of coordinate values. Specifies how data_var should be subsetted prior to dot product computation.
            
        *inplace*
            A boolean, specifying whether the new index should be added to *LaggedAnalyser.variables*
            
        *name*
            A string, specifying the name of the new index
            
        **returns**
            An xr.DataArray
            
        """
        #if mask is None:
        #    mask=self.composite_mask
                    
        comp=self.composites[data_var].sel(**sel).copy(deep=True)
        mask=mask.sel(**sel)
        da=self.variables[data_var]

        comp.values[~mask[data_var].values]=np.nan
        mean=self.lagged_means.sel({key:val for key,val in sel.items() if key in self.lagged_means.dims})
        da_anom=da-mean[data_var]

        dim_list=[d for d in list(da.dims) if d!=dim]
        indices=standardise((da_anom*comp).mean(dim_list))
        
        if name is None:
            name=data_var+"_"+"".join([f'{key}{val}_' for key,val in sel.items()])
            name=name[:-1]
        indices=indices.rename(name)
        if inplace:
            self.add_variable(indices,name=name,is_categorical=0)
        return squeeze_da(indices)
