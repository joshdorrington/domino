from domino.categorical_analysis import get_xr_occurrence

def aggregate_ds(ds,dim,agg_func,**agg_kwargs):
    return ds.map(agg_func,dim=dim,**agg_kwargs)
def mean_ds(ds,dim):
    return aggregate_ds(ds,dim,lambda da,dim: da.mean(dim=dim))
def std_ds(ds,dim):
    return aggregate_ds(ds,dim,lambda da,dim: da.std(dim=dim))
def cat_occ_ds(ds,dim,cat_ds):
    return aggregate_ds(ds,dim,get_xr_occurrence,s=cat_ds,coord_name='variable_cat_val')
