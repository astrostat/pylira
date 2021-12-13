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

# Introduction to Deconvolution using MCMC Methods


This notebooks is a short introduction to the basics of the LIRA algorithm and the Bayesian / MCMC approach to the deconvolution problem. It uses a simple, pure Numpy implementation of the classical [Metropolis- Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithm to sample from the posteriror distribution.

We start with defining the required imports:

```{code-cell} ipython3
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from pylira.utils.convolution import fftconvolve
from pylira.data import gauss_and_point_sources_gauss_psf
from pylira.utils.plot import plot_example_dataset
```

To get reproducible results we seed the Numpy random number generation first:

```{code-cell} ipython3
rng = np.random.default_rng(765)
```

For this example we use one of the builtin datasets consisting of a central extend Gaussian source and near-by point sources with varying itensity:

```{code-cell} ipython3
data = gauss_and_point_sources_gauss_psf()
plot_example_dataset(data=data)
```

We start by defining the model function to compute the predicted number of counts from the given flux image, backgroun, point spread function and exposure:

```{code-cell} ipython3
def compute_npred(flux, psf, background, exposure):
    """Compute predicted number of counts"""
    npred = fftconvolve((flux + background) * exposure, psf)
    # The FFT can produce noise, which could cause values to be negative
    npred = np.clip(npred, 0, np.inf)
    return npred
```

Now we continue by defining the likelihood, prior and posterior functions.

The data is Poissonion in nature, so the likelihood is computed using the so called "Cash" statistics, see e.g. https://cxc.cfa.harvard.edu/sherpa/statistics/#cash for details.

```{code-cell} ipython3
def cash(counts, npred):
    """Cash statistics"""
    npred = np.clip(npred, 1e-25, np.inf)
    stat = 2 * (npred - counts * np.log(npred))
    return stat
```

The log prior is assumed to be constant for now:

```{code-cell} ipython3
def log_prior(flux):
    """Log prior"""
    # currently a flat prior
    return np.log(1)


def log_likelihood(flux, psf, background, counts, exposure):
    """Log likelihood"""
    npred = compute_npred(flux, psf, background, exposure)
    return cash(counts, npred).sum()


def log_posterior(flux, psf, background, counts, exposure):
    """Log posterior"""
    return log_prior(flux) + log_likelihood(
        flux=flux, psf=psf, background=background, counts=counts, exposure=exposure
    )
```

To solve the problem each pixel in the flux image is treated as and independent parameter, from which we can sample. For this we use a simple pure Numpy based implementation of the [Metropolis- Hastings](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) algorithm. For fast convergence we choose to sample the pixel wise flux valeus from a Gamma distribution. To avoid negative flux values and the alogithm getting stuck on a zero flux, we clip the current flux values at `flux_min`:

```{code-cell} ipython3
def random_coin(p):
    unif = np.random.uniform(0, 1)
    return unif <= p


def deconvolve_mcmc(log_posterior, flux_init, data, n_samples, flux_min=1e-2):
    """Deconvolve using Baysian MCMC"""
    image_trace, log_post_trace = [], []
    
    while len(image_trace) < n_samples:
        flux = rng.gamma(np.clip(flux_init, flux_min, np.inf))
        
        prob = log_posterior(flux=flux, **data)
        prob_init = log_posterior(flux=flux_init, **data)
        
        acceptance = min(np.exp(prob_init - prob), 1)
        
        if random_coin(acceptance) and np.isfinite(prob):
            flux_init = flux

        image_trace.append(flux)
        log_post_trace.append(prob)
    
    return {
        "image-trace": np.array(image_trace),
        "log-posterior-trace": np.array(log_post_trace),
    }
```

As starting values for the flux we use the scaled data themselves and then we sample 50.000 ierations to get sufficient statistics:

```{code-cell} ipython3
%%time

flux_init = (data["counts"] - data["background"]) / data["exposure"]

data.pop("flux", None)

result = deconvolve_mcmc(
    log_posterior=log_posterior,
    flux_init=flux_init,
    data=data,
    n_samples=50_000
)
```

To check the global convergence we can plot the trace of the log posterior:

```{code-cell} ipython3
plt.plot(result["log-posterior-trace"])
```

And finally plot the mean posterior as and estimate for the deconvolved flux:

```{code-cell} ipython3
n_burn_in = 5000
posterior_mean = np.mean(result["image-trace"][n_burn_in:], axis=0)
plt.imshow(posterior_mean, origin="lower")
plt.colorbar()
```
