**********
User Guide
**********

This is how to use ``pylira``:

.. code::

    from pylira import LIRADeconvolver
    from pylira.data import point_source_gauss_psf

    data = point_source_gauss_psf()
    data["flux_init"] = data["flux"]
    deconvolve = LIRADeconvolver(
        alpha_init=np.ones(np.log2(data["counts"].shape[0]).astype(int))
    )
    result = deconvolve.run(data=data)


.. toctree::
   :maxdepth: 2
   :hidden:

   data

