---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Simple Point Source Examples

This tutorial shows you hwo to work with ``pylira`` using a simple simulated
point source and Gaussian shaped point spread function.

We first define the required imports:
```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

from pylira.data import point_source_gauss_psf, gauss_and_point_sources_gauss_psf
from pylira.utils.plot import plot_example_dataset
from pylira import LIRADeconvolver, LIRADeconvolverResult
```

Then we can use 
```{code-cell} ipython3
data = point_source_gauss_psf()
```

```{code-cell} ipython3
plot_example_dataset(data)
```

```{code-cell} ipython3
data = gauss_and_point_sources_gauss_psf()
data["flux_init"] = data["flux"]

alpha_init = 0.1 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

dec = LIRADeconvolver(
    alpha_init=alpha_init,
    n_iter_max=3000,
    n_burn_in=200,
    ms_al_kap1=0,
    ms_al_kap2=1000,
    ms_al_kap3=10,
    fit_background_scale=False,
    random_state=np.random.RandomState(156)
)
```

```{code-cell} ipython3
%%time
result = dec.run(data)
result.write("test.fits.gz", overwrite=True)
```

```{code-cell} ipython3
res = LIRADeconvolverResult.read("test.fits.gz")
```

```{code-cell} ipython3
res.plot_posterior_mean(from_image_trace=True)
```

```{code-cell} ipython3
res.plot_parameter_traces()
```

```{code-cell} ipython3
res.plot_parameter_distributions()
```

```{code-cell} ipython3
res.plot_pixel_traces_region(center_pix=(16, 16), radius_pix=4)
```
