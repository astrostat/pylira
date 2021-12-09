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

# Simple Point Source Example

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

Then we can use the method `point_source_gauss_psf` to get a pre-defined test
dataset:

```{code-cell} ipython3
data = point_source_gauss_psf()
print(data.keys())
```

The `data` variable is a `dict` containing the `counts`, `psf`, `exposure`,
`background` and ground truth for `flux`. We ca illutrated the data using:

```{code-cell} ipython3
plot_example_dataset(data)
```

Next we define the `LIRADeconvolver` class, which holds the configuration
the algorithm will be run with:

```{code-cell} ipython3
alpha_init = 0.1 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

dec = LIRADeconvolver(
    alpha_init=alpha_init,
    n_iter_max=3000,
    n_burn_in=200,
    fit_background_scale=False,
)
```

We can print the instance to see the full configuration:

```{code-cell} ipython3
print(dec)
```

Now we have to define the initial guess for the flux and add it to the `data` dictionary.
For this we use the reserved `"flux_init"`key:

```{code-cell} ipython3
data["flux_init"] = np.ones(data["counts"].shape)
```

Finally we can run the LIRA algorithm using `.run()`:

```{code-cell} ipython3
%%time
result = dec.run(data)
```

To check the validity of the result we can plot the posterior mean:

```{code-cell} ipython3
result.plot_posterior_mean()
```

As well as the parameter traces:

```{code-cell} ipython3
result.plot_parameter_traces()
```

We can also plot the parameter distributions:

```{code-cell} ipython3
result.plot_parameter_distributions()
```

And pixel traces in a given region, to check for correlations with neighbouring pixels:

```{code-cell} ipython3
result.plot_pixel_traces_region(center_pix=(16, 16), radius_pix=2)
```

```{code-cell} ipython3

```
