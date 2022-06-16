from pathlib import Path
import numpy as np
from astropy.table import Table
from scipy.ndimage import labeled_comprehension
from . import image_analysis
from .utils.io import (
    read_parameter_trace_file,
    read_image_trace_file,
    IO_FORMATS_WRITE,
    IO_FORMATS_READ,
)
from .utils.plot import (
    plot_parameter_traces,
    plot_parameter_distributions,
    plot_pixel_trace,
    plot_pixel_trace_neighbours,
)


DTYPE_DEFAULT = np.float64

__all__ = ["LIRADeconvolver", "LIRADeconvolverResult"]


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
    random_state : `~numpy.random.RandomState`
        Random state

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
        random_state=None,
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

        if random_state is None:
            random_state = np.random.RandomState(None)

        self.random_state = random_state

    def __str__(self):
        """String representation"""
        cls_name = self.__class__.__name__
        info = cls_name + "\n"
        info += len(cls_name) * "-" + "\n\n"
        data = self.to_dict()

        for key, value in data.items():
            info += f"\t{key:21s}: {value}\n"

        return info.expandtabs(tabsize=4)

    def _check_input_sizes(self, obs_arr):
        obs_shape = obs_arr.shape[0]
        if obs_shape & (obs_shape - 1) != 0:
            raise ValueError(
                f"Size of the input observation must be a power of 2. Size given: {obs_shape}"
            )

        if len(self.alpha_init) != np.log2(obs_shape):
            raise ValueError(
                f"Number of elements in alpha_init must be {np.log2(obs_shape)}.\
                     Size given: {len(self.alpha_init)} "
            )

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
        data.pop("filename_out")
        data.pop("filename_out_par")
        # TOOD: serialise random state for reproducibility?
        data.pop("random_state")
        return data

    def run(self, data):
        """Run the algorithm

        Parameters
        ----------
        data : dict of `~numpy.ndarray`
            Data

        Returns
        -------
        result : `LIRADeconvolverResult`
            Result object.
        """
        data = {name: arr.astype(DTYPE_DEFAULT) for name, arr in data.items()}
        self._check_input_sizes(data["counts"])

        random_seed = self.random_state.randint(1, np.iinfo(np.uint32).max)

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
            random_seed=random_seed,
        )

        parameter_trace = {"filename": str(
            self.filename_out_par), "format": "ascii"}
        image_trace = {"filename": str(self.filename_out), "format": "ascii"}

        config = self.to_dict()
        config["random_seed"] = random_seed
        return LIRADeconvolverResult(
            posterior_mean=posterior_mean,
            parameter_trace=parameter_trace,
            image_trace=image_trace,
            config=config,
        )


class LIRADeconvolverResult:
    """LIRA deconvolution result object.

    Parameters
    ----------
    config : `dict`
        Configuration from the `LIRADeconvolver`
    posterior_mean : `~numpy.ndarray`
        Posterior mean
    parameter_trace : `~astropy.table.Table` or dict
        Parameter trace. If a dict is provided it triggers the lazy loading.
        The dict must contain the argument to `read_parameter_trace_file`.
    image_trace : `~astropy.table.Table` or dict
        Image trace. If a dict is provided it triggers the lazy loading.
        The dict must contain the argument to `read_image_trace_file`.
    wcs : `~astropy.wcs.WCS`
        World coordinate transform object
    """

    def __init__(
        self,
        config,
        posterior_mean=None,
        parameter_trace=None,
        image_trace=None,
        wcs=None,
    ):
        self._config = config
        self._posterior_mean = posterior_mean
        self._wcs = wcs
        self._image_trace = image_trace
        self._parameter_trace = parameter_trace

    @property
    def config(self):
        """Configuration data (`dict`)"""
        return self._config

    @property
    def config_table(self):
        """Configuration data as table (`~astropy.table.Table`)"""
        config = Table()

        for key, value in self.config.items():
            if key == "alpha_init":
                value = [value]
            config[key] = value

        return config

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
        return self._posterior_mean

    @property
    def posterior_mean_from_trace(self):
        """Posterior mean computed from trace(`~numpy.ndarray`)"""
        return np.nanmean(self.image_trace[self.n_burn_in:], axis=0)

    @property
    def image_trace(self):
        """Image trace (`~numpy.ndarray`)"""
        # TODO: this currently handles only in memory data, this might not scale for
        # many iterations and/or large images
        if isinstance(self._image_trace, dict):
            self._image_trace = read_image_trace_file(**self._image_trace)

        return self._image_trace

    @property
    def parameter_trace(self):
        """Parameter trace (`~astropy.table.Table`)"""
        if isinstance(self._parameter_trace, dict):
            self._parameter_trace = read_parameter_trace_file(
                **self._parameter_trace)
            # TODO: add config to meta data of table, not sure whether it's the right place.
            self._parameter_trace.meta.update(self.config)

        return self._parameter_trace

    def plot_pixel_traces_region(self, center_pix, radius_pix=0, figsize=(16, 6)):
        """Plot pixel traces in a given region.

        Parameters
        ----------
        center_pix : tuple of int
             Pixel indices, order is (x, y). By default the trace at the center is plotted.
        radius_pix : float
           Radius in which the traces are plotted.
        figsize : tuple of float
           Figure size
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig = plt.figure(figsize=figsize)

        data = self.posterior_mean_from_trace

        ax_image = plt.subplot(1, 2, 1, projection=self.wcs)
        im = ax_image.imshow(data, origin="lower")
        fig.colorbar(im, ax=ax_image, label="Posterior Mean")

        radius = max(radius_pix, 1)
        artist = Circle(center_pix, radius=radius, color="w", fc="None")
        ax_image.add_artist(artist)

        ax_trace = plt.subplot(1, 2, 2, projection=self.wcs)

        plot_pixel_trace(
            image_trace=self.image_trace,
            center_pix=center_pix,
            ax=ax_trace,
            config=self.config,
        )

        plot_pixel_trace_neighbours(
            image_trace=self.image_trace,
            center_pix=center_pix,
            radius_pix=radius_pix,
            ax=ax_trace,
        )

    def plot_pixel_trace(self, center_pix=None, **kwargs):
        """Plot pixel trace at a given position.

        Parameters
        ----------
        center_pix : tuple of int
             Pixel indices, order is (x, y). By default the trace at the center is plotted.
        **kwargs : dict
            Keyword arguments forwarded to `plot_pixel_trace`
        """
        if center_pix is None:
            # choose center as default
            center_pix = tuple(np.array(self.posterior_mean.shape) // 2)

        plot_pixel_trace(
            image_trace=self.image_trace,
            config=self.config,
            center_pix=center_pix,
            **kwargs,
        )

    def plot_pixel_trace_neighbours(self, center_pix=None, radius_pix=0, **kwargs):
        """Plot pixel traces in a given region.

        Parameters
        ----------
        center_pix : tuple of int
            Pixel indices, order is (x, y). By default the trace at the center is plotted.
        radius_pix : float
            Radius in which the traces are plotted.
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.plot`
        """
        if center_pix is None:
            # choose center as default
            center_pix = tuple(np.array(self.posterior_mean.shape) // 2)

        plot_pixel_trace_neighbours(
            image_trace=self.image_trace,
            center_pix=center_pix,
            radius_pix=radius_pix,
            **kwargs,
        )

    def plot_posterior_mean(self, from_image_trace=False, **kwargs):
        """Plot posteriror mean

        Parameters
        ----------
        from_image_trace : bool
            Recompute posterior from image trace.
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.imshow`
        """
        import matplotlib.pyplot as plt

        fig = plt.gcf()

        if from_image_trace:
            data = self.posterior_mean_from_trace
        else:
            data = self.posterior_mean

        ax = plt.subplot(projection=self.wcs)
        im = ax.imshow(data, origin="lower", **kwargs)
        fig.colorbar(im, ax=ax, label="Posterior Mean")

    def plot_image_trace_interactive(self, **kwargs):
        """Plot image trace interactively

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to `~matplotlib.pyplot.imshow`
        """
        import matplotlib.pyplot as plt
        from ipywidgets import IntSlider
        from ipywidgets.widgets.interaction import interact

        kwargs.setdefault("interpolation", "nearest")
        kwargs.setdefault("origin", "lower")

        slider = IntSlider(
            value=0,
            min=0,
            max=self.image_trace.shape[0] - 1,
            description="Select idx: ",
            continuous_update=False,
            style={"description_width": "initial"},
            layout={"width": "50%"},
        )

        @interact(idx=slider)
        def _plot_interactive(idx):
            ax = plt.subplot(projection=self.wcs)
            im = ax.imshow(self.image_trace[idx], **kwargs)
            plt.colorbar(im, ax=ax, label="Flux")

    def plot_parameter_traces(self, **kwargs):
        """Plot parameter traces

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to `plot_parameter_traces`
        """
        plot_parameter_traces(self.parameter_trace,
                              config=self.config, **kwargs)

    def plot_parameter_distributions(self, **kwargs):
        """Plot parameter distributions

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments forwarded to `plot_parameter_distributions`
        """
        plot_parameter_distributions(
            self.parameter_trace, config=self.config, **kwargs)

    def write(self, filename, overwrite=False, format="fits"):
        """Write result fo file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        overwrite : bool
            Overwrite file.
        format : {"fits"}
            Format to use.
        """
        filename = Path(filename)

        if format not in IO_FORMATS_WRITE:
            raise ValueError(
                f"Not a valid format '{format}', choose from {list(IO_FORMATS_WRITE)}"
            )

        writer = IO_FORMATS_WRITE[format]
        writer(result=self, filename=filename, overwrite=overwrite)

    @classmethod
    def read(cls, filename, format="fits"):
        """Write result fo file

        Parameters
        ----------
        filename : str or `Path`
            Output filename
        format : {"fits"}
            Format to use.

        Returns
        -------
        result : `~LIRADeconvolverResult`
            Result object
        """
        filename = Path(filename)

        if format not in IO_FORMATS_READ:
            raise ValueError(
                f"Not a valid format '{format}', choose from {list(IO_FORMATS_READ)}"
            )

        reader = IO_FORMATS_READ[format]
        kwargs = reader(filename=filename)
        return cls(**kwargs)


class LIRASignificanceEstimator:
    """
    Estimate the significance of emission from specified regions
    using the method described in Stein et al. (2015)

    Parameters
    ----------
    result_observed_im: `~LIRADeconvolverResult`
        LIRA result for the observed image
    result_replicates: list
        LIRA result array for the baseline images
    labels_im: `~numpy.ndarray`
        Image with regions where each region is indicated with a unique integer
    """

    def __init__(
        self,
        result_observed_im,
        result_replicates,
        labels_im,
    ):
        self._result_observed_im = result_observed_im
        self._result_replicates = result_replicates
        self._labels_im = labels_im

        self._labels = labels_im.unique()

    def _get_im_subset(self, im_trace, iter, img_dim):
        return im_trace[iter*img_dim:(iter+1)*img_dim, :]

    def _estimate_xi(self, result):
        xi_regions = []
        burnin = result.config['n_burn_in']
        n_iter = result.config['n_iter_max']
        thin = result.config['thin']
        fit_bkgscl = result.config['fit_background_scale']
        bkg_scale_trace = result.parameter_trace['bkgScale']
        image_dim = result.config['data']['background'].shape[0]
        image_trace = result.image_trace

        baseline_im = result.config['data']['background']

        baseline_sum = labeled_comprehension(
            baseline_im, self._labels_im, self._labels, np.sum, float, 0)

        # loop over each image from the trace and estimate xi
        for iter in range(burnin, n_iter, thin):

            tau_1 = labeled_comprehension(
                self._get_im_subset(image_trace, iter,
                                    image_dim), self._labels_im, self._labels, np.sum, float, 0
            )

            tau_0 = baseline_sum
            if fit_bkgscl == 1:
                tau_0 = baseline_sum * bkg_scale_trace[iter]

            xi_regions.append(tau_1/(tau_1+tau_0))

        # each row is a distribution of xi for one region
        xi_regions = np.array(xi_regions).T

        return {
            self._labels[i]: xi_regions[i] for i in range(self._labels.shape[0])
        }
    
    def _estimate_test_statistic(tail,observed_dist):
        return (observed_dist>=tail).sum()/observed_dist.shape[0]

    def _estimate_pval_ul(gamma,test_stat):
        """
        Stein et al. (2015) eq. 22
        """
        return gamma/test_stat



    def estimate_p_values(self,gamma=0.005):

        xi_dist_observed_im = self._estimate_xi(self._result_observed_im)
        xi_dist_replicates = [self._estimate_xi(result_replicate) for result_replicate in self._result_replicates]

        xi_dist_merged_replicates = {
            self._labels[i]: [] for i in range(self._labels.shape[0])
        }

        for xi_replicate in xi_dist_replicates.items():
            for k,v in xi_replicate:
                xi_dist_merged_replicates[k] = np.concat(xi_dist_merged_replicates[k],v)


        xi_dist_merged_replicates = {
            k: np.flatten(v) for k,v in xi_dist_merged_replicates.items()
        }

        
        #find the 1-gamma percentile
        tail_1_gamma = {
            k: np.percentile(v,(1-gamma)*100) for k,v in xi_dist_merged_replicates.items()
        }

        #find the number of values in the xi_dist_observed beyond these percentiles
        test_statistic = {
            k: self._estimate_test_statistic(v,xi_dist_observed_im[k]) for k,v in tail_1_gamma.items()
        }

        #estimate upper limit on p-values
        p_value_ul = {
            k: self._estimate_pval_ul(gamma,v) for k,v in test_statistic.items()
        }

        return p_value_ul


