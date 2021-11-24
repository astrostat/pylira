from pathlib import Path
import numpy as np
from . import image_analysis
from .utils.io import read_parameter_trace_file, read_image_trace_file


DTYPE_DEFAULT = np.float64

__all__ = ["LIRADeconvolver"]


class LIRADeconvolver:
    """LIRA image deconvolution method

    Parameters
    ----------
    alpha_init : `~numpy.ndarray`
        Initial alpha parameters. The length must be n for an input image of size 2^n x 2^n
    n_iter_max : int
        Max. number of iterations.
    n_burn_in : int
        Number of burn-in iterations.
    fit_background_scale : bool
        Fit background scale.
    save_thin : True
        Save thin?
    ms_ttlcnt_pr: float
        Multiscale prior TODO: improve description
    ms_ttlcnt_exp: float
        Multiscale prior TODO: improve description
    ms_al_kap1: float
        Multiscale prior TODO: improve description
    ms_al_kap2: float
        Multiscale prior TODO: improve description
    ms_al_kap3: float
        Multiscale prior TODO: improve description
    filename_out: str or `Path`
        Output filename
    filename_out_par: str or `Path`
        Parameter output filename

    Examples
    --------
    This how to use the class:

    .. code::

        from pylira import LIRADeconvolver
        from pylira.data import point_source_gauss_psf

        data = point_source_gauss_psf()
        data["flux_init"] = data["flux"]
        deconvolve = LIRADeconvolver(
            alpha_init=np.ones(np.log2(data["counts"].shape[0]).astype(int))
        )
        result = deconvolve.run(data=data)

    """

    def __init__(
            self,
            alpha_init,
            n_iter_max=3000,
            n_burn_in=1000,
            fit_background_scale=False,
            save_thin=True,
            ms_ttlcnt_pr=1,
            ms_ttlcnt_exp=0.05,
            ms_al_kap1=0.0,
            ms_al_kap2=1000.0,
            ms_al_kap3=3.0,
            filename_out="output.txt",
            filename_out_par="output-par.txt",
    ):
        self.alpha_init = np.array(alpha_init, dtype=DTYPE_DEFAULT)
        self.n_iter_max = n_iter_max
        self.n_burn_in = n_burn_in
        self.fit_background_scale = fit_background_scale
        self.save_thin = save_thin
        self.ms_ttlcnt_pr = ms_ttlcnt_pr
        self.ms_ttlcnt_exp = ms_ttlcnt_exp
        self.ms_al_kap1 = ms_al_kap1
        self.ms_al_kap2 = ms_al_kap2
        self.ms_al_kap3 = ms_al_kap3
        self.filename_out = Path(filename_out)
        self.filename_out_par = Path(filename_out_par)

    def _check_input_sizes(self, obs_arr):
        obs_shape = obs_arr.shape[0]
        if obs_shape & (obs_shape - 1) != 0:
            raise ValueError(
                f"Size of the input observation must be a power of 2. Size given: {obs_shape}")

        if len(self.alpha_init) != np.log2(obs_shape):
            raise ValueError(
                f"Number of elements in alpha_init must be {np.log2(obs_shape)}.\
                     Size given: {len(self.alpha_init)} ")

    def to_dict(self):
        """Convert deconvolver configuration to dict, with simple data types.

        Returns
        -------
        data : dict
            Parameter dict.
        """
        data = {}
        data.update(self.__dict__)
        data["alpha_init"] = self.alpha_init.tolist()
        data["filename_out"] = str(self.filename_out)
        data["filename_out_par"] = str(self.filename_out_par)
        return data

    def run(self, data):
        """Run the algorithm

        Parameters
        ----------
        data : dict of `~numpy.ndarray`
            Data

        Returns
        -------
        result : dict
            Result dictionary containing "posterior-mean" (`~numpy.ndarray`)
            and "parameter-trace" (`~astropy.table.Table`).
        """
        data = {name: arr.astype(DTYPE_DEFAULT) for name, arr in data.items()}
        self._check_input_sizes(data["counts"])

        posterior_mean = image_analysis(
            observed_im=data["counts"],
            start_im=data["flux_init"],
            psf_im=data["psf"],
            expmap_im=data["exposure"],
            baseline_im=data["background"],
            max_iter=self.n_iter_max,
            burn_in=self.n_burn_in,
            save_thin=self.save_thin,
            fit_bkgscl=int(self.fit_background_scale),
            out_img_file=str(self.filename_out),
            out_param_file=str(self.filename_out_par),
            alpha_init=self.alpha_init,
            ms_ttlcnt_pr=self.ms_ttlcnt_pr,
            ms_ttlcnt_exp=self.ms_ttlcnt_exp,
            ms_al_kap1=self.ms_al_kap1,
            ms_al_kap2=self.ms_al_kap2,
            ms_al_kap3=self.ms_al_kap3,
        )

        return LIRADeconvolverResult(
            posterior_mean=posterior_mean,
            config=self.to_dict()
        )


class LIRADeconvolverResult:
    """LIRA deconvolution result object.

    Parameters
    ----------
    config : `dict`
        Configuration from the `LIRADeconvolver`
    posterior_mean : `~numpy.ndarray`
        Posterior mean
    wcs : `~astropy.wcs.WCS`
        World coordinate transform object
    """
    def __init__(self, config, posterior_mean=None,  wcs=None):
        self._config = config
        self._posterior_mean = posterior_mean
        self._wcs = wcs
        self._image_trace = None
        self._parameter_trace = None

    @property
    def config(self):
        """Optional wcs"""
        return self._config

    @property
    def wcs(self):
        """Optional wcs"""
        return self._wcs

    @property
    def n_burn_in(self):
        """Number of burn in iterations"""
        return self.config.get("n_burn_in", 0)

    @property
    def posterior_mean(self):
        """Posterior mean (`~numpy.ndarray`)"""
        if self._posterior_mean is None:
            self._posterior_mean = np.nanmean(self.image_trace[self.n_burn_in:], axis=0)

        return self._posterior_mean

    @property
    def image_trace(self):
        """Image trace (`~numpy.ndarray`)"""
        # TODO: this currently handles only in memory data, this might not scale for
        # many iterations and/or large images
        if self._image_trace is None:
            filename = self.config.get("filename_out")
            self._image_trace = read_image_trace_file(filename)

        return self._image_trace

    @property
    def parameter_trace(self):
        """Parameter trace (`~astropy.table.Table`)"""
        if self._parameter_trace is None:
            filename = self.config.get("filename_out_par")
            self._parameter_trace = read_parameter_trace_file(filename)
            # TODO: add config to meta data of table, not sure whether it's the right place.
            self._parameter_trace.meta.update(self.config)

        return self._parameter_trace
