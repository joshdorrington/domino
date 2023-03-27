import numpy as np
import xarray as xr
import operator as op
from functools import partial

#Set the object up, pass in a mask, and get a filtered mask back
class MaskFilter(object):
    
    def __init__(self):
        
        self._filter_funcs=[]
        self._required_kwargs=[]

        self.filter_description=[]
        self.long_filter_description=[]
        
        
        self._filter_modes=['union','intersection','replace']
        
        self._comparators={
            'lt':op.lt,
            'gt':op.gt,
            'le':op.le,
            'ge':op.ge,
            'eq':op.eq,
            'neq':op.ne}
        return
    
    
    #valid inputs:
    #        an integer (1 value for everything)
    #        a dict of var names and integers (1 value per var)
    #        a dataset with values on each coord
    
    def add_value_filter(self,threshold, comparison_func, filter_mode='union',as_abs=False):
        
        assert comparison_func in self._comparators
        assert filter_mode in self._filter_modes
        
        comparator=self._comparators[comparison_func]
        if as_abs:
            comparator=lambda val_ds,thresh_ds: self._comparators[comparison_func](np.abs(val_ds),thresh_ds)
            
        value_filter=partial(self._apply_value_filter,threshold,comparator,filter_mode)
        
        self._filter_funcs.append(value_filter)
        self._required_kwargs.append('val_ds')
        self.filter_description.append('value filter')
        self.long_filter_description.append(\
            f'value filter, comparator: {comparison_func}, filter_mode: {filter_mode}, as_abs: {as_abs}')
        
        return
    def _apply_value_filter(self,threshold,comparator,filter_mode,mask_ds=None,val_ds=None):
        
        vfilt=comparator(val_ds,threshold)
        
        if filter_mode=='union':
            vfilt=vfilt or mask_ds
        elif filter_mode=='intersection':
            vfilt=vfilt*mask_ds
        else:
            assert filter_mode=='replace'
            
        return vfilt
    
    def add_convolution_filter():
        return
    def _apply_convolution_filter():
        return

    def add_continuity_filter():
        return
    
    
    def _apply_continuity_filter():
        return

    def filter_mask(self,mask_ds,**kwargs):
        mask_ds=mask_ds.copy(deep=True)
        for f,fname in zip(self._filter_funcs,self.filter_description):
            try:
                mask_ds=f(mask_ds,**kwargs)
            except Exception as e:
                raise(ValueError(f'Failure applying filter {fname}:\n{e}'))
        return mask_ds
        
    
        
class IndexComputer(object):
    """x
    
        **Arguments:**
        
        *a*
            xxx
            
        **Optional arguments**
        
        *a*
        
            xxx
"""
    def __init__(self):
        
        return
    
    def __repr__(self):
        return 1
    
    def __str__(self):
        return self.__repr__()
