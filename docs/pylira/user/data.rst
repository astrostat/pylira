Example Datasets
================

Illustration of example datasets.


Point Source and Gaussian PSF
-----------------------------

.. plot::
    :include-source:

    from pylira.data import point_source_gauss_psf
    from pylira.utils.plot import plot_example_dataset

    data = point_source_gauss_psf()
    plot_example_dataset(data)


Disk Source and Gaussian PSF
----------------------------

.. plot::
    :include-source:

    from pylira.data import disk_source_gauss_psf
    from pylira.utils.plot import plot_example_dataset

    data = disk_source_gauss_psf()
    plot_example_dataset(data)


Gaussian and Point Sources and Gaussian PSF
-------------------------------------------

.. plot::
    :include-source:

    from pylira.data import gauss_and_point_sources_gauss_psf
    from pylira.utils.plot import plot_example_dataset

    data = gauss_and_point_sources_gauss_psf()
    plot_example_dataset(data)
