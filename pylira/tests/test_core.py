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
        fit_background_scale=True,
        random_state=np.random.RandomState(156)
    )
    return deconvolve.run(data=data)


def test_np_random_state():
    # test to check numpy random state is platform independent
    random_state = np.random.RandomState(1234)

    assert random_state.randint(0, 10) == 3
    assert random_state.randint(0, 10) == 6


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
    assert_allclose(lira_result.posterior_mean[16][16], 955.7, rtol=3e-2)
    assert_allclose(lira_result.posterior_mean, lira_result.posterior_mean_from_trace, atol=1e-2)

    assert (lira_result.posterior_mean[16][16] > 700)
    assert lira_result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in lira_result.config

    assert_allclose(result[16][16], 955.7, rtol=3e-2)
    assert_allclose(result[0][0], 0.032, atol=0.1)

    # check total flux conservation
    assert_allclose(result.sum(), data["flux"].sum(), rtol=3e-2)

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")
    assert_allclose(trace_par["smoothingParam0"][-1], 0.019, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam1"][-1], 0.019, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam2"][-1], 0.042, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam3"][-1], 0.058, rtol=3e-2)


def test_lira_deconvolver_run_disk_source(tmpdir):
    data = disk_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = 0.02 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=1000,
        n_burn_in=100,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
        fit_background_scale=True,
        random_state=np.random.RandomState(156)
    )
    result = deconvolve.run(data=data)

    assert(result.posterior_mean[16][16] > 0.2)
    assert_allclose(result[16][16], 0.229, rtol=3e-2)
    assert_allclose(result[0][0], 0.0011, atol=0.1)

    # check total flux conservation
    # TODO: improve accuracy
    assert_allclose(result.sum(), data["flux"].sum(), rtol=0.4)

    assert result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in result.config

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")
    assert_allclose(trace_par["smoothingParam0"][-1], 0.023, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam1"][-1], 0.026, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam2"][-1], 0.15, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam3"][-1], 0.048, rtol=3e-2)


def test_lira_deconvolver_run_gauss_source(tmpdir):
    data = gauss_and_point_sources_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = 0.02 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=1000,
        n_burn_in=100,
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
    assert_allclose(result[16][16], 22.753878, rtol=1e-2)
    assert_allclose(result[16][16], 0.338, rtol=3e-2)
    assert_allclose(result[0][0], 0.0011, atol=0.1)

    # check total flux conservation
    # TODO: improve accuracy
    assert_allclose(result.sum(), data["flux"].sum(), rtol=0.4)

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")
    assert_allclose(trace_par["smoothingParam0"][-1], 0.038, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam1"][-1], 0.030, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam2"][-1], 0.0038, rtol=3e-2)
    assert_allclose(trace_par["smoothingParam3"][-1], 0.162, rtol=3e-2)
