# Domino is a package for analysing composites of atmospheric data. This is an experimental release!

## Documentation

See https://weather-domino.readthedocs.io/en/latest/docs/composites.html , or documentation.pdf in this repo

## Examples

A basic introduction to using the package out of the box, with three example applications:

https://github.com/joshdorrington/domino/blob/main/examples/basic_usage.ipynb

A more in-depth discussion of how to customise the behaviour of the LaggedAnalyser class:

https://github.com/joshdorrington/domino/blob/main/examples/advanced_precursors.ipynb

## Install

At present, best way is to download and extract this repository, at your_favourite_path, and then at the top of your .py file, write: 
```
import sys
sys.path.append('your_favourite_path')
```
You can then import from domino like any other Python package.
