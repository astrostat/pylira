import numpy as np
from numpy.testing import assert_allclose
import pylira
from pylira.data import point_source_gauss_psf, disk_source_gauss_psf
from pylira import LIRADeconvolver


def test_import_name():
    assert pylira.__name__ == "pylira"


def test_image_analysis():
    assert pylira.image_analysis is not None


def test_lira_deconvolver():
    deconvolve = LIRADeconvolver(
        alpha_init=np.array([1, 2, 3])
    )

    assert deconvolve.alpha_init.dtype == np.float64
    assert_allclose(deconvolve.alpha_init, [1., 2., 3.])


def test_lira_deconvolver_run_point_source():
    data = point_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10
    )
    result = deconvolve.run(data=data)

    assert(result[16][16] > 700)


def test_lira_deconvolver_run_disk_source():
    data = disk_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10
    )
    result = deconvolve.run(data=data)

    assert(result[16][16] > 0.2)
