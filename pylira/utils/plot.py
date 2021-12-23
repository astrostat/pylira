from itertools import zip_longest
import numpy as np
from astropy.visualization import simple_norm


__all__ = [
    "plot_example_dataset",
    "plot_parameter_traces",
    "plot_parameter_distributions",
    "plot_pixel_trace",
    "plot_pixel_trace_neighbours",
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
        np.array([y1]),
        np.array([y2]),
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
            np.array([x1]),
            np.array([x2]),
            color="tab:orange",
            alpha=0.2,
            zorder=9,
            label=r"1 $\sigma$ Std. Deviation",
        )

        if not has_legend:
            ax.legend()
            has_legend = True

    return axes


def plot_pixel_trace(image_trace, center_pix, ax=None, config=None, **kwargs):
    """Plot pixel traces in a circular region, given a position and radius.

    Parameters
    ----------
    image_trace : `~numpy.ndarray`
        Image traces array
    center_pix : tuple of int
        Pixel indices center, order is (x, y).
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes
    config : dict
        Configuration dictionary
    **kwargs : dict
        Keyword arguments passed to `~matplotlib.pyplot.plot`

    Returns
    -------
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes

    """
    import matplotlib.pyplot as plt

    if config is None:
        config = {}

    if ax is None:
        ax = plt.gca()

    n_iter, n_y, n_x = image_trace.shape
    n_burn_in = config.get("n_burn_in", 0)

    idx = np.arange(n_iter)

    trace = image_trace[(Ellipsis,) + center_pix[::-1]].T

    kwargs.setdefault("color", "tab:blue")
    plot_trace(ax=ax, trace=trace, idx=idx, n_burn_in=n_burn_in, **kwargs)
    ax.set_title(f"Pixel trace for {center_pix}")
    ax.set_xlabel("Number of Iterations")
    ax.legend()
    return ax


def plot_pixel_trace_neighbours(
        image_trace, center_pix, radius_pix=1, cmap="Greys",  ax=None, **kwargs
):
    """Plot pixel traces in a given region.

    The distance to the center is encoded in the color the trace it plotted with.

    Parameters
    ----------
    image_trace : `~numpy.ndarray`
        Image traces array
    center_pix : tuple of int
        Pixel indices, order is (x, y). By default the trace at the center is plotted.
    radius_pix : float
        Radius in which the traces are plotted.
    cmap : str
        Colormapt o plot the traces with.
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes
    **kwargs : dict
        Keyword arguments forwarded to `~matplotlib.pyplot.plot`

    Returns
    -------
    ax : `~matplotlib.pyplot.Axes`
        Plotting axes

    """
    import matplotlib
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    _, ny, nx = image_trace.shape
    y, x = np.arange(ny).reshape((-1, 1)), np.arange(nx)
    offset_pix = np.sqrt((y - center_pix[1]) ** 2 + (x - center_pix[0]) ** 2)
    idx = np.where((offset_pix < radius_pix) & (offset_pix > 0))

    cmap = matplotlib.cm.get_cmap(cmap)
    norm = simple_norm(data=offset_pix[idx])

    idx = idx + (offset_pix[idx],)

    kwargs.setdefault("zorder", 0)
    kwargs.setdefault("alpha", 0.5)

    for idx_x, idx_y, offset in zip(*idx):
        trace = image_trace[(slice(None), idx_x, idx_y)]
        value = norm(offset)
        color = tuple(cmap(value)[0])
        ax.plot(trace, color=color,  **kwargs)

    return ax
