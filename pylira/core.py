from pathlib import Path
import numpy as np
from . import image_analysis

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
    filename_out_pat: str or `Path`
        Parameyter output filename

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
        if (obs_shape & (obs_shape-1) != 0):
            raise ValueError(
                f"Size of the input observation must be a power of 2. Size given: {obs_shape}")

        if (self.alpha_init.shape[0] != np.log2(obs_shape)):
            raise ValueError(
                f"Number of elements in alpha_init must be {np.log2(obs_shape)}. Size given: {self.alpha_init.shape[0]} ")

    def run(self, data):
        """Run the algorithm

        Parameters
        ----------
        data : dict of `~numpy.ndarray`
            Data

        Returns
        -------
        result : `~numpy.ndarray`
            Mean posterior.
        """
        data = {name: arr.astype(DTYPE_DEFAULT) for name, arr in data.items()}
        self._check_input_sizes(data["counts"])

        result = image_analysis(
            observed_im=data["counts"],
            start_im=data["flux_init"],
            psf_im=data["psf"],
            expmap_im=data["exposure"],
            baseline_im=data["background"],
            max_iter=self.n_iter_max,
            burn_in=self.n_burn_in,
            save_thin=self.save_thin,
            out_img_file=str(self.filename_out),
            out_param_file=str(self.filename_out_par),
            alpha_init=self.alpha_init,
            ms_ttlcnt_pr=self.ms_ttlcnt_pr,
            ms_ttlcnt_exp=self.ms_ttlcnt_exp,
            ms_al_kap1=self.ms_al_kap1,
            ms_al_kap2=self.ms_al_kap2,
            ms_al_kap3=self.ms_al_kap3,
        )
        return result
