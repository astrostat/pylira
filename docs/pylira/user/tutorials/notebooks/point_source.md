---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.13.7
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
import copy
import numpy as np
import matplotlib.pyplot as plt
from regions import CirclePixelRegion, PixCoord
from multiprocessing import Pool

from pylira.data import point_source_gauss_psf, gauss_and_point_sources_gauss_psf
from pylira.utils.plot import plot_example_dataset, plot_hyperpriors_lira
from pylira import LIRADeconvolver, LIRADeconvolverResult, LIRASignificanceEstimator

random_state = np.random.RandomState(148)
```

## Dataset

Then we can use the method `point_source_gauss_psf` to get a pre-defined test
dataset:

```{code-cell} ipython3
data = point_source_gauss_psf(random_state=random_state)
print(data.keys())
```

The `data` variable is a `dict` containing the `counts`, `psf`, `exposure`,
`background` and ground truth for `flux`. We can illustrate the data using:

```{code-cell} ipython3
plot_example_dataset(data)
```

## Configure and Run Deconvolution

Next we define the `LIRADeconvolver` class, which holds the configuration
the algorithm will be run with:

```{code-cell} ipython3
alpha_init = 0.1 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

dec = LIRADeconvolver(
    alpha_init=alpha_init,
    n_iter_max=5_000,
    n_burn_in=200,
    fit_background_scale=False,
    random_state=random_state
)
```

We can print the instance to see the full configuration:

```{code-cell} ipython3
print(dec)
```

Now we can also visualize the hyperprior distribution of the alpha parameter:

```{code-cell} ipython3
plot_hyperpriors_lira(
    ncols=1,
    figsize=(8, 5),
    alphas=np.linspace(0, 0.3, 100),
    ms_al_kap1=0,
    ms_al_kap2=1000,
    ms_al_kap3=3
);
```

Now we have to define the initial guess for the flux and add it to the `data` dictionary.
For this we use the reserved `"flux_init"`key:

```{code-cell} ipython3
data["flux_init"] = random_state.gamma(10, size=data["counts"].shape)
```

Finally we can run the LIRA algorithm using `.run()`:

```{code-cell} ipython3
%%time
result = dec.run(data)
```

### Results and Diagnosis

To check the validity of the result we can plot the posterior mean:

```{code-cell} ipython3
---
nbsphinx-thumbnail:
  tooltip: Learn how to use pylira on a simple point source example.
---
result.plot_posterior_mean(from_image_trace=True)
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
result.plot_pixel_traces_region(center_pix=(16, 16), radius_pix=2);
```

## Significance Estimation

The signifiance estimation follows the method proposed by [Stein et al. 2015](https://ui.adsabs.harvard.edu/abs/2015ApJ...813...66S/abstract).

### Null Hypothesis Simulations

First we simulate ("fake") observations for the null hypothesis: in this case we assume the null hypothesis includes the background only:

```{code-cell} ipython3
# number of simulations
n_iter = 10

# reduce number of iterations for LIRA
dec.n_iter_max = 1_000
```

```{code-cell} ipython3
datasets = []
data_null = copy.deepcopy(data)

for idx in range(n_iter):
    data_null["counts"] = random_state.poisson(data["background"])
    data_null["flux_init"] = random_state.gamma(10, size=data["counts"].shape)
    datasets.append(data_null)
```

```{code-cell} ipython3
%%time
with Pool(processes=8) as pool:
    results = pool.map(dec.run, datasets)
```

### Define Regions of Interest

```{code-cell} ipython3
# Region of interest for the source 
region_src = CirclePixelRegion(
    center=PixCoord(16, 16),
    radius=2
)

# Some control region 
region_bkg = CirclePixelRegion(
    center=PixCoord(23, 8),
    radius=5
)

labels = np.zeros((32, 32))

for idx, region in enumerate([region_src, region_bkg]):
    mask = region.to_mask()
    labels += (idx + 1) * mask.to_image(shape=labels.shape)
```

```{code-cell} ipython3
plt.imshow(labels)
plt.colorbar();
```

### Estimate and Visualize Distributions of $\xi$

First we instantiate the `LIRASignificanceEstimator` with the result obtained before, the results from the null hypothesis simulations and the label image:

```{code-cell} ipython3
est = LIRASignificanceEstimator(
    result_observed_im=result,
    result_replicates=results,
    labels_im=labels
)
```

Now we first estimate the p-values for the regions defined by the labels.

```{code-cell} ipython3
result_p_values = est.estimate_p_values(
    data=data, gamma=0.005
)
```

For the we consider the parameter $\xi$, defined as:

$\xi = \tau_1 / (\tau_1 + \tau_0)$

So the proportion of the total image intensity that is due to the added component $\tau_1$. The distribution of $\xi$ can be plotted the different regions. First the region of interest in the center:

```{code-cell} ipython3
ax = est.plot_xi_dist(
    xi_obs=result_p_values[2],
    xi_repl=result_p_values[1],
    region_id="1.0"
);
ax.set_ylim(0, 0.2);
ax.set_xlim(-8.3, 0.2);
```

In this case all the probability mass is in the region of the bright point source and the mean of $\xi$ approaches 1.

+++

We also illustrate the distribution of $\xi$ for second region, where we only expect background:

```{code-cell} ipython3
ax = est.plot_xi_dist(
    xi_obs=result_p_values[2],
    xi_repl=result_p_values[1],
    region_id="2.0"
);
ax.set_ylim(0, 0.2);
ax.set_xlim(-8.3, 0.2);
```

In this case the distribution is fully compatible with the distribution of the null hypothesis.

```{code-cell} ipython3

```
