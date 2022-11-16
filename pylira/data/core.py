import numpy as np
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS

__all__ = [
    "point_source_gauss_psf",
    "disk_source_gauss_psf",
    "gauss_and_point_sources_gauss_psf",
    "lincoln",
    "chandra_gc",
]


def point_source_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=3,
    source_level=1000,
    background_level=2,
    random_state=None,
):
    """Get point source with Gaussian PSF test data.

    The exposure is assumed to be constant.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape)

    flux = np.zeros(shape)
    flux[shape[0] // 2, shape[1] // 2] = source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux,
    }


def disk_source_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=3,
    source_level=1000,
    source_radius=3,
    background_level=2,
    random_state=None,
):
    """Get disk source with Gaussian PSF test data.

    The exposure has a gradient of 50% from left to right.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    source_radius : float
        Radius of the disk source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape) + 0.5 * np.linspace(-1, 1, shape[0])

    flux = (
        source_level
        * Tophat2DKernel(
            radius=source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
        ).array
    )

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = convolve_fft((flux + background) * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux,
    }


def gauss_and_point_sources_gauss_psf(
    shape=(32, 32),
    shape_psf=(17, 17),
    sigma_psf=2,
    source_level=1000,
    source_radius=2,
    background_level=2,
    random_state=None,
):
    """Get data with a Gaussian source in the center and point sources of varying brightness
    of 100%, 30%, 10% and 3% of the Gaussian source.

    The exposure has a gradient of 50% from top to bottom.

    Parameters
    ----------
    shape : tuple
        Shape of the data array.
    shape_psf : tuple
        Shape of the psf array.
    sigma_psf : float
        Width of the psf in pixels.
    source_level : float
        Total integrated counts of the source
    source_radius : float
        Radius of the disk source
    background_level : float
        Background level in counts / pixel.
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape) + 0.5 * np.linspace(-1, 1, shape[0]).reshape((-1, 1))

    flux = (
        source_level
        * Gaussian2DKernel(
            source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
        ).array
    )

    for fraction, idx_x, idx_y in zip(
        [1, 0.3, 0.1, 0.03], [16, 16, 26, 6], [26, 6, 16, 16]
    ):
        flux[idx_y, idx_x] = fraction * source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = convolve_fft((flux + background) * exposure, psf)

    counts = random_state.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux,
    }


def lincoln(psf=Gaussian2DKernel(3), random_state=None):
    """Get Abraham Lincoln image similar to [Esch2004]_.

    The exposure is unity and background is zero.

    Parameters
    ----------
    psf : `~astropy.convolution.Kernel2D`
        PSF Kernel to be used. Default is Gaussian of sigma = 3 pixels.
    random_state : `~numpy.random.RandomState`
        Random state

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    import matplotlib.image as mpimg

    if random_state is None:
        random_state = np.random.RandomState(None)

    filename = get_pkg_data_filename("files/lincoln.png", package="pylira.data")
    flux = np.sum(mpimg.imread(filename), axis=-1)
    flux = flux.max() - flux

    background = np.zeros(flux.shape)
    exposure = np.ones(flux.shape)

    npred = np.clip(convolve_fft((flux + background) * exposure, psf), 0, np.inf)
    counts = random_state.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux,
    }


def chandra_gc(obs_id=4683, cutout=True):
    """Get Chandra Galactic Center example dataset.

    The exposure is assumed unity and background is zero.

    Parameters
    ----------
    obs_id : {4683, 4684}
        Which observation id
    cutout : bool
        Use a smaller data cutout

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    filename = get_pkg_data_filename(
        f"files/chandra-counts-obs-id-{obs_id}.fits.gz", package="pylira.data"
    )
    counts = fits.getdata(filename)
    wcs = WCS(fits.getheader(filename))

    filename = get_pkg_data_filename(
        f"files/chandra-psf-obs-id-{obs_id}.fits.gz", package="pylira.data"
    )
    psf = fits.getdata(filename)

    if cutout:
        psf_cutout = (slice(14, 31), slice(15, 32))
        psf = psf[psf_cutout]

        counts_cutout = (slice(65, 193), slice(64, 192))
        counts = counts[counts_cutout]
        wcs = wcs[counts_cutout]

    background = np.zeros(counts.shape)
    exposure = np.ones(counts.shape)

    return {
        "counts": counts,
        "psf": psf,
        "exposure": exposure,
        "background": background,
        "wcs": wcs,
    }
