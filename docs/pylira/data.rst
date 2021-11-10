****************
Example Datasets
****************

Illustration of example datasets.


Point Source and Gaussian PSF
-----------------------------

.. plot::

    from pylira.data import point_source_gauss_psf
    from pylira.utils.plot import plot_example_dataset

    data = point_source_gauss_psf()
    plot_example_dataset(data)