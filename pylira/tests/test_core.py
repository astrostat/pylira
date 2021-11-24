import numpy as np
from numpy.testing import assert_allclose
import pylira
from pylira.data import (
    point_source_gauss_psf,
    disk_source_gauss_psf,
    gauss_and_point_sources_gauss_psf
)
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

    config = deconvolve.to_dict()

    assert_allclose(config["alpha_init"], [1, 2, 3])
    assert config["filename_out"] == "output.txt"


def test_lira_deconvolver_run_point_source(tmpdir):
    data = point_source_gauss_psf()
    data["flux_init"] = data["flux"]

    alpha_init = np.ones(np.log2(data["counts"].shape[0]).astype(int))

    deconvolve = LIRADeconvolver(
        alpha_init=alpha_init,
        n_iter_max=100,
        n_burn_in=10,
        filename_out=tmpdir / "image-trace.txt",
        filename_out_par=tmpdir / "parameter-trace.txt",
        fit_background_scale=True
    )
    result = deconvolve.run(data=data)

    posterior_mean = result["posterior-mean"]

    assert(posterior_mean[16][16] > 700)

    trace_par = result["parameter-trace"]
    assert trace_par["smoothingParam0"][-1] > 0
    assert "alpha_init" in trace_par.meta


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
        fit_background_scale=True
    )
    result = deconvolve.run(data=data)
    posterior_mean = result["posterior-mean"]

    assert(posterior_mean[16][16] > 0.2)

    trace_par = result["parameter-trace"]
    assert trace_par["smoothingParam0"][-1] > 0
    assert "alpha_init" in trace_par.meta


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
        fit_background_scale=True
    )
    result = deconvolve.run(data=data)
    posterior_mean = result["posterior-mean"]

    assert(posterior_mean[16][16] > 0.2)

    trace_par = result["parameter-trace"]
    assert trace_par["smoothingParam0"][-1] > 0
    assert "alpha_init" in trace_par.meta
