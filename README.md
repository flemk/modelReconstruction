# Model Reconstruction
This repository provides advanced utility to analyze n-dimensional systems and reconstruct them from its time series.

# ```reconstructionutils.py```
Use this library to create reconstructed models from your time series. simply type:

```python
import reconstructionutils as ru

# set up your systems time series
series = [z_1, z_2, ..., z_n]

# create a model instance
system = ru.Model(series, 6)

# create a model and reconstruct your system
res = system.evaluate()
```

TODO:
- [ ] take the mean as defined in arithmetic mean in ```reconstructionutils.Model._retrieve_fit_coefficients```
- [ ] using ```np.gradient``` for derivative in ```reconstructionutils.Model.__init__```

# ```stanpy.py```
View [flemk/ModelReconstruction](github.com/flemk/StochasticAnalysis) for  examples using this module.

This module provides a class to determine drift- and diffusion-coefficients of n-dimensional time series by using their statistical definition.

```python
import stanpy as sp

time_series = [[1, 2, ...], [1, 2, ...]] # your time seres you want to analyze

analysis = sp.StochasticAnalysis(time_series)
analysis.analyze()

# drift and diffusion coefficients are now stored in:
analysis.drift()
analysis.diffusion()

# in the 2d case you can visualize them builtin:
analysis.visualize_2d()

# and you can reconstruct your series with choosen initial values:
r = analysis.reconstruct()

# by converting your coefficients into a FPE you might gain more insight:
f = analysis.solve_fpe()
```

# ```cutility.py```
Math's helper function are stored in this module. Featuring finite differences and upwind schemes as well as mulitdimensional polynominal exponents.