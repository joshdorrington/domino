# Domino is a package for analysing composites of atmospheric data.
## Based on xarray, Domino makes it easy to calculate lagged composites of fields and scalar indices around categorical event time series, and to compute bootstrapped confidence bounds. This is still an alpha release! While core functionality is stable, there could be some bugs!

![title](Imgs/domino_logo.png)


## Documentation

See our [API reference](https://github.com/joshdorrington/domino/blob/master/docbuild/domino-composite.pdf) for a full description of all functionality.

[//]: # (Our preprint on Domino (under consideration at QJRMS), and its application to extreme rainfall prediction can be found [here](where))

## Examples

See our Jupyter notebook examples for more detailed discussion of how to apply Domino to different use cases.

Our [basic](https://github.com/joshdorrington/domino/blob/master/examples/basic_compositing.ipynb) and [advanced](https://github.com/joshdorrington/domino/blob/master/examples/advanced_compositing.ipynb) compositing guides cover the use of Domino's flexible LaggedAnalyser class to easily compute time-lagged composites and apply bootstrap significance tests to them.

Producing filtered precursor patterns from composites, and computing precursor activity indices from those, is covered in our [Index_Computation guide](https://github.com/joshdorrington/domino/blob/master/examples/precursor_index_computation.ipynb), while an introduction to assessing the predictive power of indices is in the [Index_Predictability guide](https://github.com/joshdorrington/domino/blob/master/examples/Index_predictability.ipynb).


## Install

domino can be installed using pip:
```
python -m pip install domino-composite
```
If you want to run the worked examples in the Jupyter notebooks you will need to download the [netcdf files containing example data](https://github.com/joshdorrington/domino/releases/tag/v1-data).