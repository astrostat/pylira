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


def plot_parameter_traces(parameter_trace, config=None, figsize=(16, 16), ncols=3, **kwargs):
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

    kwargs.setdefault("color", "tab:blue")
    nrows = (len(table.colnames) // ncols) + 1

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
    )

    if config is None:
        config = table.meta

    n_burn_in = config.get("n_burn_in", 0)
    burn_in = slice(0, n_burn_in)
    valid = slice(n_burn_in, -1)
    idx = np.arange(len(table))

    for name, ax in zip_longest(table.colnames, axes.flat):
        if name is None:
            ax.set_visible(False)
            continue

        ax.plot(
            idx[burn_in],
            parameter_trace[name][burn_in],
            alpha=0.3,
            label="Burn in",
            **kwargs
        )
        ax.plot(idx[valid], parameter_trace[name][valid], label="Valid", **kwargs)
        ax.set_title(name.title())
        ax.set_xlabel("Number of Iterations")

        mean = np.mean(parameter_trace[name][valid])
        ax.hlines(
            mean, n_burn_in, len(idx), color="tab:orange", zorder=10, label="Mean"
        )

        std = np.std(parameter_trace[name][valid])
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

        if name == "logPost":
            ax.legend()

    return axes


def plot_parameter_distributions(parameter_trace, config=None, figsize=(16, 16), ncols=3, **kwargs):
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
    table.remove_columns(
        ["iteration", "stepSize", "cycleSpinRow", "cycleSpinCol", "logPost"]
    )

    if config is None:
        config = table.meta

    n_burn_in = config.get("n_burn_in", 0)

    nrows = (len(table.colnames) // ncols) + 1

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
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
