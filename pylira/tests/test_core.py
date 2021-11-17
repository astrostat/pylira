import pytest
import numpy as np
from numpy.testing import assert_allclose
import pylira
from pylira.data import (
    point_source_gauss_psf,
    disk_source_gauss_psf,
    gauss_and_point_sources_gauss_psf
)
from pylira import LIRADeconvolver, LIRADeconvolverResult


@pytest.fixture(scope="session")
def lira_result(tmpdir_factory):
    data = point_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    tmpdir = tmpdir_factory.mktemp("data")
    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
        fit_background_scale=True
    )
    return deconvolve.run(data=data)


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

    config = deconvolve.to_dict()

    assert_allclose(config["alpha_init"], [1, 2, 3])
    assert not config["fit_background_scale"]

    assert "alpha_init" in str(deconvolve)


def test_lira_deconvolver_run_point_source(lira_result):
    assert_allclose(lira_result[16][16], 955.675754, rtol=1e-2)
    assert_allclose(lira_result.posterior_mean, lira_result.posterior_mean_from_trace, atol=1e-2)

    assert (lira_result.posterior_mean[16][16] > 700)
    assert lira_result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in lira_result.config


def test_lira_deconvolver_run_disk_source(tmpdir):
    data = disk_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
        fit_background_scale=True,
        random_state=np.random.RandomState(156)
    )
    result = deconvolve.run(data=data)

    assert(result.posterior_mean[16][16] > 0.2)
    assert_allclose(result[16][16], 17.45414, rtol=1e-2)

    assert result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in result.config

    assert_allclose(result.posterior_mean, result.posterior_mean_from_trace, atol=1e-2)


def test_lira_deconvolver_run_gauss_source(tmpdir):
    data = gauss_and_point_sources_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
        fit_background_scale=True,
        random_state=np.random.RandomState(156)
    )
    result = deconvolve.run(data=data)

    assert(result.posterior_mean[16][16] > 0.2)

    assert result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in result.config

    assert_allclose(result.posterior_mean, result.posterior_mean_from_trace, atol=1e-2)


def test_lira_deconvolver_result_write(tmpdir, lira_result):
    filename = tmpdir / "test.fits.gz"
    lira_result.write(filename)


def test_lira_deconvolver_result_read(tmpdir, lira_result):
    filename = tmpdir / "test.fits.gz"
    lira_result.write(filename)

    new_result = LIRADeconvolverResult.read(filename)

    assert_allclose(lira_result.config["alpha_init"], new_result.config["alpha_init"])
    assert_allclose(lira_result.posterior_mean, new_result.posterior_mean)

    assert lira_result.image_trace.shape == new_result.image_trace.shape
    assert_allclose(result[16][16], 22.753878)

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")
    assert trace_par["smoothingParam0"][-1] > 0
