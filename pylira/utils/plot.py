

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
