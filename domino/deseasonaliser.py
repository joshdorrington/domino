import numpy as np
import xarray as xr
import cftime
import pandas as pd
from scipy.optimize import leastsq
import os
from domino.util import xarr_times_to_ints

class Agg_Deseasonaliser(object):
    """
    Computes a seasonal cycle from an xarray dataset based on an aggregation
    criteria i.e. month, day of year, week of year etc.
    ----------
    Methods
    -------
    fit_cycle(DataArray,dim=str,agg=str): Aggregates the input DataArray arr.

    evaluate_cycle(data=DataArray): When data is None, creates a seasonal cycle
    DataArray the same shape as the input data used in fit_cycle. data can be
    a DataArray with a differing dim coordinate, but matching other coordinates.
    This can be used to fit a seasonal cycle on one dataset, but evaluate it on
    another, as long as the grid, levels, etc are the same.
    Returns The seasonal cycle as a DataArray
    """
    def __init__(self):
        self.data=None
        self.cycle_coeffs=None


    def fit_cycle(self,arr,dim="time",agg="dayofyear"):
        var=dim+"."+agg
        self.data=arr
        self.dim=dim
        self.agg=agg
        self.cycle_coeffs=arr.groupby(var).mean(dim=dim)
        return

    def evaluate_cycle(self,data=None,smooth=1):
        if data is None:
            data=self.data
        cycle=self.cycle_coeffs.sel({self.agg:getattr(data[self.dim].dt,self.agg).data})
        return self.smooth(cycle,smooth)
    
    def smooth(self,arr,w):
        if w==1:
            return arr
        else:
            return arr.rolling({self.agg:w},min_periods=1,center=True).mean()
        
class Agg_FlexiDeseasonaliser(Agg_Deseasonaliser):
    """Modifies Agg_Deseasonaliser, with the ability
    to handle custom summary funcs other than the mean.
    summary_func must take an xarray.groupby object, and
    accept a keyword 'dim'."""
    
    def fit_cycle(self,arr,summary_func,dim="time",agg="dayofyear"):
        var=dim+"."+agg
        self.data=arr
        self.dim=dim
        self.agg=agg
        self.cycle_coeffs=summary_func(arr.groupby(var),dim=dim)
        return

    
class Sinefit_Deseasonaliser(Agg_Deseasonaliser):

    """
    Same functionalityy as Agg_Deseasonaliser but fits a number of sin modes
    to the data using least squares fitting. This produces a smoother cycle than
    aggregating but can be slow for large datasets.
    ----------
    Methods
    -------
    fit_cycle(DataArray,dim=str,N=int,period=float): N sets the number of sin modes
    to fit as fractions of the period. i.e. the default of N=4 fits waves with
    periods of period, period/2, period/3 and period/4. period is a float setting
    the length of a year in days.

    evaluate_cycle(data=DataArray): As for Agg_Deseasonaliser
    """
    def _func_residual(self,p,x,t,N,period):
        return x - self._evaluate_fit(t,p,N,period)

    def _evaluate_fit(self,t,p,N,period):
        ans=p[1]
        for i in range(0,N):
            ans+=p[2*i+2] * np.sin(2 * np.pi * (i+1)/period * t + p[2*i+3])
        return ans

    def _lstsq_sine_fit(self,arr,t,N,period):

        #Guess initial parameters
        std=arr.std()
        p=np.zeros(2*(N+1))
        for i in range(0,N):
            p[2+2*i]=std/(i+1.0)
        plsq=leastsq(self._func_residual,p,args=(arr,t,N,period))
        return plsq[0]

    def fit_cycle(self,arr,dim="time",N=4,period=365.25):

            dims=arr.dims
            self.dim=dim
            self.data=arr
            self.N=N
            self.period=period
            t=xarr_times_to_ints(arr[dim])
            self.coeffs= xr.apply_ufunc(
                self._lstsq_sine_fit,
                arr,
                input_core_dims=[[dim]],
                output_core_dims=[["coeffs"]],
                vectorize=True,
                kwargs={"t":t,"N":N,"period":period})
            return

    def evaluate_cycle(self,data=None):
        if data is None:
            data=self.data
        dims=data.dims
        t=xarr_times_to_ints(data[self.dim])
        print(self.coeffs.shape)
        cycle=np.array([[self._evaluate_fit(t,c2.data,self.N,self.period)\
                for c2 in c1] for c1 in np.atleast_3d(self.coeffs)])

        return data.transpose(...,"time").copy(data=cycle).transpose(*dims)
