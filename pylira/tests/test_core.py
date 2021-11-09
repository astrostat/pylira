from numpy.testing import assert_allclose
import pylira
from pylira.data import point_source_gauss_psf


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
