import numpy as np
from numpy.testing import assert_allclose
import pylira
from pylira.data import point_source_gauss_psf
from pylira import LIRADeconvolver


def test_import_name():
    assert pylira.__name__ == "pylira"


def test_image_analysis():
    assert pylira.image_analysis is not None


def test_data_point_source_gauss_psf():
    data = point_source_gauss_psf()

    assert_allclose(data["counts"][0][0], 2)
    assert_allclose(data["exposure"][0][0], 1)
    assert_allclose(data["background"][0][0], 2.)
    assert_allclose(data["psf"][0][0], 3.413792e-05, rtol=1e-5)


def test_lira_deconvolver():
    deconvolve = LIRADeconvolver(
        alpha_init=np.array([1, 2, 3])
    )

    assert deconvolve.alpha_init.dtype == np.float64
    assert_allclose(deconvolve.alpha_init, [1., 2., 3.])


# TODO: this still fails with a segfault...
def test_lira_deconvolver_run():
    data = point_source_gauss_psf()
    data["flux_init"] = data["flux"]

    deconvolve = LIRADeconvolver(
        alpha_init=np.ones(data["psf"].shape[0])
    )
    result = deconvolve.run(data=data)

    assert(result[16][16]>900)
