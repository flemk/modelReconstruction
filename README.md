# Model Reconstruction
this repository provides advanced utility to analyze n-dimensional systems and reconstruct them from its time series.

# ```reconstructionutils.py```
use this library to create reconstructed models from your time series. simply type:

```python
import reconstructionutils as ru

# set up your systems time series
series = [z_1, z_2, ..., z_n]

# create a model instance
system = ru.Model(series, 6)

# create a model and reconstruct your system
res = system.evaluate()
```

# ```stanpy.py```
view [flemk/ModelReconstruction](github.com/flemk/ModelReconstruction) for further details on usage of this module.

tba.

# ```cutility.py```
math's helper function are stored in this module.