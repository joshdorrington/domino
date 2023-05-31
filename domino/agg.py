from domino.categorical_analysis import get_xr_occurrence

def aggregate_ds(ds,dim,agg_func,**agg_kwargs):
    """
    A wrapper function for applying aggregation functions to datasets.
    **Arguments**
    *ds* An xarray.Dataset to aggregate
    *dim* A string specifiying the dimension of *ds* to aggregate over.
    *agg_func* An aggregation function with signature (xarray.Dataset,string,**agg_kwargs)->(xarray.Dataset)
    
    **Outputs**
    An xarray.Dataset aggregated over *dim*.
    """
    return ds.map(agg_func,dim=dim,**agg_kwargs)
def mean_ds(ds,dim):
    """Returns the mean of the dataset over dim"""
    return aggregate_ds(ds,dim,lambda da,dim: da.mean(dim=dim))
def std_ds(ds,dim):
    """Returns the standard deviation of the dataset over dim"""
    return aggregate_ds(ds,dim,lambda da,dim: da.std(dim=dim))
def cat_occ_ds(ds,dim,cat_ds):
    """Returns the occurrence frequency of the dataset's categorical variables"""
    return aggregate_ds(ds,dim,get_xr_occurrence,s=cat_ds,coord_name='variable_cat_val')
