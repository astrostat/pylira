from itertools import zip_longest
import numpy as np

__all__ = [
    "plot_example_dataset",
    "plot_parameter_traces",
    "plot_parameter_distributions",
]


def plot_example_dataset(data, figsize=(12, 7), **kwargs):
    """Plot example dataset

    Parameters
    ----------
    data : dict of `~numpy.ndarray`
        Data
    figsize : tuple
        Figure size
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.imshow`
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)

    for name, ax in zip(data.keys(), axes.flat):
        im = ax.imshow(data[name], origin="lower", **kwargs)
        ax.set_title(name.title())
        fig.colorbar(im, ax=ax)

    axes.flat[-1].set_visible(False)


def get_grid_figsize(width, ncols, nrows):
    height = width * (nrows / ncols)
    return width, height


def plot_trace(ax, idx, trace, n_burn_in, **kwargs):
    """Plot a single parameter trace

    Parameters
    ----------
    ax : `~matplotlib.pyplot.Axes`
        Plot axes
    idx : `~numpy.ndarray`
        Iteration
    trace : `~numpy.ndarray`
        Trace to plot
    n_burn_in : int
        Number of burn in iterations
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.plot`

    """
    burn_in = slice(0, n_burn_in)
    valid = slice(n_burn_in, -1)

    ax.plot(
        idx[burn_in],
        trace[burn_in],
        alpha=0.3,
        label="Burn in",
        **kwargs
    )
    ax.plot(idx[valid], trace[valid], label="Valid", **kwargs)
    ax.set_xlabel("Number of Iterations")

    mean = np.mean(trace[valid])
    ax.hlines(
        mean, n_burn_in, len(idx), color="tab:orange", zorder=10, label="Mean"
    )

    std = np.std(trace[valid])
    y1, y2 = mean - std, mean + std
    ax.fill_between(
        idx[valid],
        y1,
        y2,
        color="tab:orange",
        alpha=0.2,
        zorder=9,
        label=r"1 $\sigma$ Std. Deviation",
    )


def plot_parameter_traces(parameter_trace, config=None, figsize=None, ncols=3, **kwargs):
    """Plot parameters traces

    Parameters
    ----------
    parameter_trace : `~astropy.table.Table`
        Parameter trace table
    config : dict
        Config dictionary.
    figsize : tupe of float
        Figure size
    ncols : int
        Number of columns to plot.
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.plot`

    Returns
    -------
    axes : `~numpy.ndarray` of `~matplotlib.pyplot.Axes`
        Plotting axes
    """
    import matplotlib.pyplot as plt

    table = parameter_trace.copy()
    table.remove_columns(["iteration", "stepSize", "cycleSpinRow", "cycleSpinCol"])

    if config is None:
        config = table.meta

    kwargs.setdefault("color", "tab:blue")
    nrows = (len(table.colnames) // ncols) + 1

    if figsize is None:
        figsize = get_grid_figsize(width=16, ncols=ncols, nrows=nrows)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        gridspec_kw={"hspace": 0.25}
    )

    n_burn_in = config.get("n_burn_in", 0)
    idx = np.arange(len(table))

    for name, ax in zip_longest(table.colnames, axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        trace = parameter_trace[name]
        plot_trace(ax=ax, trace=trace, idx=idx, n_burn_in=n_burn_in, **kwargs)
        ax.set_title(name.title())
        if name == "logPost":
            ax.legend()

    return axes


def plot_parameter_distributions(parameter_trace, config=None, figsize=None, ncols=3, **kwargs):
    """Plot parameters traces

    Parameters
    ----------
    parameter_trace : `~astropy.table.Table`
        Parameter trace table
    config : dict
        Config dictionary
    figsize : tupe of float
        Figure size
    ncols : int
        Number of columns to plot.
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.hist`

    Returns
    -------
    axes : `~numpy.ndarray` of `~matplotlib.pyplot.Axes`
        Plotting axes
    """
    import matplotlib.pyplot as plt

    table = parameter_trace.copy()
    table.remove_columns(
        ["iteration", "stepSize", "cycleSpinRow", "cycleSpinCol", "logPost"]
    )

    if config is None:
        config = table.meta

    n_burn_in = config.get("n_burn_in", 0)

    nrows = (len(table.colnames) // ncols) + 1

    if figsize is None:
        figsize = get_grid_figsize(width=16, ncols=ncols, nrows=nrows)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        gridspec_kw={"hspace": 0.25}
    )

    kwargs.setdefault("color", "tab:blue")

    kwargs.setdefault("density", True)
    kwargs.setdefault("bins", int(np.sqrt(len(table))))

    has_legend = False

    for name, ax in zip_longest(table.colnames, axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        column = parameter_trace[name][n_burn_in:]
        is_finite = np.isfinite(column)
        n_vals, bins, _ = ax.hist(column[is_finite], label="Valid", **kwargs)

        column_burn_in = parameter_trace[name][:n_burn_in]
        is_finite_burn_in = np.isfinite(column_burn_in)
        n_vals_burn_in, _, _ = ax.hist(
            column_burn_in[is_finite_burn_in], alpha=0.3, label="Burn in", **kwargs
        )

        ax.set_title(name.title())
        ax.set_xlabel("Number of Iterations")

        y_max = np.max([n_vals, n_vals_burn_in])
        mean = np.mean(column[is_finite])
        ax.vlines(mean, 0, y_max, color="tab:orange", zorder=10, label="Mean")

        std = np.std(column[is_finite])
        x1, x2 = mean - std, mean + std

        ax.fill_betweenx(
            np.linspace(0, y_max, 10),
            x1,
            x2,
            color="tab:orange",
            alpha=0.2,
            zorder=9,
            label=r"1 $\sigma$ Std. Deviation",
        )

        if not has_legend:
            ax.legend()
            has_legend = True

    return axes


def plot_pixel_traces_region(image_trace, center_pix, radius_pix=5, posterior_mean=None, config=None, **kwargs):
    """Plot pixel traces in a circular region, given a position and radius.

    Parameters
    ----------
    image_trace : `~numpy.ndarray`
        Image traces array
    config : dict
        Configuration dictionary
    center_pix : tuple of int
        Pixel indices center
    radius_pix : float
        Radius of the of the region in pixels.
    posterior_mean : `~numpy.ndarray`
        Posterior mean.
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.plot`
    """
    import matplotlib.pyplot as plt

    if posterior_mean is None:
        posterior_mean = np.nanmean(self.image_trace[self.n_burn_in:], axis=0)

    y_idx, x_idx = np.meshgrid()
    y_center, x_center = center_pix

    offset = np.sqrt((y_center - y_idx) ** 2 + (x_center - x_idx) ** 2)
    idx = np.where(offset < radius)

    alpha = 1
    kwargs.setdefault("alpha", alpha)
    plt.plot(image_trace[idx, :], **kwargs)


