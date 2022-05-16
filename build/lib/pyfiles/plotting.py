import cartopy.crs as ccrs
import xarray as xr
import cmocean.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

__DEFAULT_PROJ__=ccrs.PlateCarree()
__DEFAULT_EXTENTS__=[-180,180,25,89]
__VAR_SCALING__=1.05

def cat_2d_plot(da,cmap=None,sig_da=None):
    if da.ndim!=2:
        raise(ValueError('We assume a 2d dataArray!'))
   
    if da.dims[0]!='variable_cat_val':
        if da.dims[1]!='variable_cat_val':
            print('Warning: Unclear which dimension is variable_cat_val!')
        else:
            da=da.T
            if sig_da is not None:
                sig_da=sig_da.T
    fig,ax=plt.subplots(1)
    try:
        labs=da.attrs['regime_labels']
    except:
        labs=np.arange(da.shape[0])
    if cmap is None:
        cmap=plt.colormaps['coolwarm']
   
    p=da.plot(ax=ax,cmap=cmap)
   
    ax.set_yticks(np.arange(da.shape[0]))
    ax.set_yticklabels(labs)
    
    ax.set_xticks(da[da.dims[1]].values)
    
    if sig_da is not None:
        pos=ax.get_position()
        sig_da.plot(colors=[(0,0,0, 0.8),(1,1,1, 0.0)],levels=[-0.5,0.5,1.5],ax=ax)
        fig.axes[-1].remove()
        ax.set_position(pos)
    return fig,ax


def cat_1d_plot(da,cmap=None,colors=None,sig_da=None):
    if da.ndim!=1:
        raise(ValueError('We assume a 1d dataArray!'))
   
    fig,ax=plt.subplots(1)
    try:
        labs=da.attrs['regime_labels']
    except:
        labs=np.arange(da.shape[0])
    if cmap is None:
        cmap=plt.colormaps['coolwarm']
    if colors is None:
        x=da.values
        x=(x-x.min())/(x.max()-x.min())
        colors=[cmap(X) for X in x]
   
    p=ax.bar(labs,da,color=colors)
   
    if sig_da is not None:
        is_sig=sig_da.values
        for i,b in zip(is_sig,p.patches):
            b.set_fill([None,1][int(i)])
    return fig,ax

def blank_carto_plot(x=1,y=1,proj=None,fig=None):
    if fig is None:
        fig=plt.figure()

    if proj is None:
        proj=__DEFAULT_PROJ__
    
    axis_set=np.array([[plt.subplot(y,x,j*x+i+1,projection=proj) for j in range(y)] for i in range(x)])    
    #remove 1d axes and then if 0d, make them a scalar:
    axis_set=np.squeeze(axis_set)
    if np.size(axis_set)==1:
        axis_set=axis_set.item()
        
    return fig,axis_set

def quickplot(da,proj=None,extents=None,clevs=None,ax=None,cmap=cm.balance,plot_kwargs={},sig_da=None,data_crs=None,fig=None):
    
    if da.ndim!=2:
        raise(ValueError('quickplot assumes 2D data!'))
        
    if proj is None:
        proj=__DEFAULT_PROJ__
    if extents is None:
        extents=__DEFAULT_EXTENTS__
    if clevs is None:
        lim=np.abs(da).max().values.item()*__VAR_SCALING__
        clevs=np.linspace(-lim,lim,21)
    if ax is None:
        fig,ax=blank_carto_plot(proj=proj,fig=fig)
    if data_crs is None:
        data_crs=ccrs.PlateCarree()
        
    fig=plt.gcf()
    
    da.plot(transform=data_crs,levels=clevs,cmap=cmap,**plot_kwargs,ax=ax)
    ax.set_extent(extents,crs=data_crs)
    
    #Shade over points that are False in a boolean da
    if sig_da is not None:
        sig_da.plot.contourf(transform=data_crs,levels=[0,0.5,1],colors=[(0,0,0,0.3),'none'],add_colorbar=False,ax=ax)
        
    ax.coastlines()

    return fig,ax   


def quickplot_grid(da,grid_dims,max_val=20,sig_da=None,proj=ccrs.PlateCarree(),fig=None,**quickplot_kwargs):
    
    fig,axes=blank_carto_plot(*grid_dims,proj,fig=fig)
    L=da.shape[0]
    if L>max_val:
        raise(ValueError(f'Length of dataarray, {L}, exceeds max_val, {max_val}. If you are sure this is correct, change max_val.'))
    
    
    for l,ax in zip(range(L),axes.T.reshape(-1)):
        if sig_da is None:
            quickplot(da[l],ax=ax,**quickplot_kwargs)
        else:
            quickplot(da[l],ax=ax,sig_da=sig_da[l],**quickplot_kwargs)

    return fig,axes        


def contourf_grid(da,dims,**kwargs):
    f,a=quickplot_grid(da,dims,**kwargs)
    d1,d2=dims
    f.set_figwidth(d1*12)
    f.set_figheight(d2*3)
    return f,a

def overlay_contour(da,ax,sig_da=None,**kwargs):
    L=da.shape[0]
    for l,a in zip(np.arange(L),ax.T.reshape(-1)):
        c=da[l].copy()
        data=c.data
        if sig_da is not None:
            data[~sig_da[l].values]=0
        c.data=data
        p=c.plot.contour(ax=a,**kwargs)
    return 
