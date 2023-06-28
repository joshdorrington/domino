r'''

#Welcome to the Domino documentation!

The five main classes in Domino are:

*domino.core.LaggedAnalyser*

For computing lagged composites with respect to categorical time series.

*domino.core.PatternFilter*

For applying filters to Boolean masks.

*domino.core.IndexGenerator*

For reducing fields to scalar indices by projection onto masked field composites.

*domino.prediction.PredictionTest*

For assessing the predictive skill of scalar indices with respect to categorical events.

*domino.PLSR.PLSR_Reduction*

For reducing the dimensionality of a multivariate dataset using Partial least squares regression.


#If you're new to Domino, don't forget to check out the [worked examples](https://github.com/joshdorrington/domino/tree/master/examples)!
'''



import os
import sys
sys.path.insert(0, os.path.abspath('./'))