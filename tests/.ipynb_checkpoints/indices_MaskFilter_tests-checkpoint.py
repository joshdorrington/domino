import sys
sys.path.append('../domino')
import indices
import xarray as xr
import importlib
comp_ds=xr.open_dataset('/data/ox5324/precursor_computation/composite_files/DJF_C_France/DJF_C_France_composites.nc')[['V300','HurricaneOcc']].sel(index_val=1,lag=[-2,-1,0])

sig_ds=xr.open_dataset('/data/ox5324/precursor_computation/composite_files/DJF_C_France/DJF_C_France_raw_masks.nc')[['V300','HurricaneOcc']].sel(index_val=1,lag=[-2,-1,0])

#Simple additive intersection filter
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(0.002,'ge',filter_mode='intersection')
add_intersection_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

#Simple additive union filter
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(0.002,'ge',filter_mode='union')
add_union_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

#Additive replace
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(0.002,'ge',filter_mode='replace')
add_replace_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

#Simple greater than abs val union filter
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(0.5,'gt',filter_mode='union',as_abs=True)
abs_val_intersection_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

#Not equal to zero
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(0.,'neq',filter_mode='replace')
neq_zero_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

#small sig vals
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter({'V300':2,'HurricaneOcc':0.003},'le',filter_mode='intersection')
small_sig_mask=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

lat_filter=xr.Dataset({'V300':xr.DataArray(data=np.arange(2,8.6,0.1),coords={'lat':sig_ds.lat}),
                       'HurricaneOcc':xr.DataArray(data=np.arange(-0.03,0.036,0.001),coords={'lat':sig_ds.lat})})

#lat filter
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(lat_filter,'lt',filter_mode='intersection')
lat_filter=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)

lon=xr.Dataset({'V300':xr.DataArray(data=np.linspace(-5,5,241),coords={'lon':sig_ds.lon}),
                       'HurricaneOcc':xr.DataArray(data=np.arange(-0.03,0.036,0.001),coords={'lat':sig_ds.lat})})

#Mixed filter
Mfilter=indices.MaskFilter()
Mfilter.add_value_filter(lon,'lt',filter_mode='intersection',as_abs=False)
lon_filter=Mfilter.filter_mask(mask_ds=sig_ds,val_ds=comp_ds)
