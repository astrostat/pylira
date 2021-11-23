from itertools import zip_longest
import numpy as np

__all__ = ["plot_example_dataset", "plot_parameter_traces", "plot_parameter_distributions"]


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


def plot_parameter_traces(parameter_trace, figsize=(16, 16), ncols=3, **kwargs):
    """Plot parameters traces

    Parameters
    ----------
    parameter_trace : `~astropy.table.Table`
        Parameter trace table
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

    nrows = (len(table.colnames) // ncols) + 1

    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize,
    )

    for name, ax in zip_longest(table.colnames, axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        ax.plot(parameter_trace[name], **kwargs)
        ax.set_title(name.title())
        ax.set_xlabel("Number of Iterations")

    return axes


def plot_parameter_distributions(parameter_trace, figsize=(16, 16), ncols=3, **kwargs):
    """Plot parameters traces

    Parameters
    ----------
    parameter_trace : `~astropy.table.Table`
        Parameter trace table
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
    table.remove_columns(["iteration", "stepSize", "cycleSpinRow", "cycleSpinCol", "logPost"])

    nrows = (len(table.colnames) // ncols) + 1

    fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=figsize,
    )

    kwargs.setdefault("density", True)
    kwargs.setdefault("bins", int(np.sqrt(len(table))))

    for name, ax in zip_longest(table.colnames, axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        column = parameter_trace[name]
        is_finite = np.isfinite(column)
        ax.hist(column[is_finite], **kwargs)
        ax.set_title(name.title())
        ax.set_xlabel("Number of Iterations")

    return axes
