import numpy as np
import xarray as xr
import sklearn.cross_decomposition as skcc

class PLSR_Reduction(object):
    """
    Wraps around the scikit-learn partial-least-squares-regression algorithm, supporting prediction-focused dimensionality reduction.
    
    **Arguments**
    
    *mode_num*
    
    An integer number of PLSR modes to retain. Defaults to the second dimension of the predictor variables *X* passed to *PLSR_Reduction.fit*
    
    """    
    def __init__(self,mode_num):
        """Initialise a PLSR_Reduction object"""
        self.mode_num=mode_num
        """@private"""
        return
    
    def __repr__(self):
        return 'A PLSR_Reduction object'
    
    def __str__(self):
        return self.__repr__()
    
    def _validate_shape(self,da):
        
        #Expand if necessary
        if da.ndim==1:
            da=da.expand_dims('feature')
            
        assert da.ndim==2
        assert self.sample_dim in da.dims
        
        d0,d1=da.dims
        #Transpose if necessary
        if d0!=self.sample_dim:
            da=da.T
        return da

        
    def fit(self,X,y,sample_dim='time'):
        """ 
        Performs the partial-least-squares regression of *X* against *y*, returning the transformed PLSR modes.
        
        **Arguments**
    
        *X*
        
        The predictor variables to transform. An xarray.Datarray with two coordinates representing the sample and feature dimensions as specified by *sample_dim*.
        
        *y*
        
        The target variables to be predicted. An xarray.Datarray with two coordinates representing the sample and feature dimensions as specified by *sample_dim*.
        
        **Optional arguments**
        
        *sample_dim*
        
        A string specifying the sample dimension which must be shared between *X* and *y*. Defaults to *time*.
        
        **Returns**
        
        *plsr_X*
        
        A DataArray containing PLSR modes, with a sample coordinate given by the intersection of the sample coordinates of *X* and *y*

        """
        X,y=xr.align(X.dropna(sample_dim),y.dropna(sample_dim))
        assert len(y>1)
        self.sample_dim=sample_dim
        self.X=self._validate_shape(X)
        self.y=self._validate_shape(y)
        
        d0,d1=self.X.dims
        assert d0==self.sample_dim
        self.feature_dim=d1
        
        model=skcc.PLSRegression(n_components=self.mode_num)
        model.fit(self.X.values,self.y.values)
        rotations=model.x_rotations_
        coords={d1:self.X[d1].values,'PLSR_mode':range(1,self.mode_num+1)}
        rotation_da=xr.DataArray(data=rotations,coords=coords,dims=[d1,'PLSR_mode'])

        plsr_X=self.X.values@rotations
        plsr_X=xr.DataArray(data=plsr_X,coords={sample_dim:self.X[sample_dim],'PLSR_mode':range(1,self.mode_num+1)})
        
        self.rotations=rotations
        self.rotation_da=rotation_da
        self.PLSR_X=plsr_X
        
        return plsr_X
    
    def project_quasimodes(self,X):
        
        """
        Use precomputed rotation matrix from calling *PLSR_Reduction.fit* to project *X* into the space of PLSR modes.
        This can be used to extend the PLSR modes to times when the target events are not defined.
        
        **Arguments**
        
        
        *X*
        
        The predictor variables to transform. An xarray.Datarray with two coordinates representing the sample and feature dimensions as specified by the *sample_dim* used to fit the PLSR regression.
        
        **Returns**
        
        *proj_plsr_X*
        
        A DataArray containing projected PLSR modes, with the same sample coordinate as *X*.
        
        """       
        
        #PLSR_projection
        
        X=self._validate_shape(X)
        
        
        d0,d1=X.dims
        proj_plsr_X=X.values@self.rotations
        proj_plsr_X=xr.DataArray(data=proj_plsr_X,coords={self.sample_dim:X[self.sample_dim],'PLSR_mode':range(1,self.mode_num+1)})
        return proj_plsr_X
    
    def project_pattern(self,X):
        """ 
        Compute the patterns corresponding to each PLSR mode, by computing a weighted sum of the spatial patterns corresponding to the predictor indices used to fit the partial least squares regression.

        **Arguments**
        
        
        *X*
        
        The pattern to transform. An xarray.Datarray of arbitrary shape, and with an identical feature coordinate to the predictor variables passed into *PLSR_Reduction.fit*.
        
        **Returns**
        
        *pattern*
        
        A DataArray containing the resulting weighted sums, with the same coordinates as *X*, except for the feature coordinate is replaced with the PLSR_mode coordinate.
        """
        assert self.feature_dim in X.dims
        pattern=(self.rotation_da*X).sum(self.feature_dim)
        return pattern
        
    