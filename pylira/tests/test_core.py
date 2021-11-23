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

    idx = slice(deconvolve.n_burn_in, -1)
    assert_allclose(np.mean(trace_par["smoothingParam0"][idx]), 0.056, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam1"][idx]), 0.060, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam2"][idx]), 0.060, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam3"][idx]), 0.062, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam4"][idx]), 0.070, rtol=5e-2)


def test_lira_deconvolver_run_disk_source(tmpdir):
    data = disk_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = 0.02 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=1000,
        n_burn_in=100,
        ms_al_kap1=0,
        ms_al_kap2=1000,
        ms_al_kap3=10,
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
    assert_allclose(result.sum(), 1400, rtol=0.1)

    assert result.parameter_trace["smoothingParam0"][-1] > 0
    assert "alpha_init" in result.config

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")

    idx = slice(deconvolve.n_burn_in, -1)
    assert_allclose(np.mean(trace_par["smoothingParam0"][idx]), 0.08, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam1"][idx]), 0.20, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam2"][idx]), 0.31, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam3"][idx]), 0.34, rtol=5e-2)


def test_lira_deconvolver_run_gauss_source(tmpdir):
    data = gauss_and_point_sources_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = 0.02 * np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=1000,
        n_burn_in=100,
        ms_al_kap1=0,
        ms_al_kap2=1000,
        ms_al_kap3=10,
        fit_background_scale=False,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
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

    # check at point source positions
    assert_allclose(result[16][26], 151.0, rtol=3e-2)
    assert_allclose(result[16][6], 2.88, rtol=3e-2)
    assert_allclose(result[26][16], 1337.0, rtol=3e-2)
    assert_allclose(result[6][16], 319.0, rtol=3e-2)
    assert_allclose(result[0][0], 0, atol=0.1)

    # check total flux conservation
    # TODO: improve accuracy
    assert_allclose(result.sum(), 3430, rtol=0.1)

    trace_par = read_parameter_trace_file(tmpdir / "parameter-trace.txt")

    idx = slice(deconvolve.n_burn_in, -1)
    assert_allclose(np.mean(trace_par["smoothingParam0"][idx]), 0.04, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam1"][idx]), 0.10, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam2"][idx]), 0.18, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam3"][idx]), 0.30, rtol=5e-2)
    assert_allclose(np.mean(trace_par["smoothingParam4"][idx]), 0.36, rtol=5e-2)
