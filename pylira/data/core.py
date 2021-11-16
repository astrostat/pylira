import numpy as np
from astropy.convolution import Gaussian2DKernel, Tophat2DKernel, convolve_fft


__all__ = ["point_source_gauss_psf", "disk_source_gauss_psf"]


def point_source_gauss_psf(
        shape=(32, 32),
        shape_psf=(17, 17),
        sigma_psf=3,
        source_level=1000,
        background_level=2,
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

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    np.random.seed(836)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape)

    flux = np.zeros(shape)
    flux[shape[0] // 2, shape[1] // 2] = source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = np.random.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux
    }


def disk_source_gauss_psf(
        shape=(32, 32),
        shape_psf=(17, 17),
        sigma_psf=3,
        source_level=1000,
        source_radius=3,
        background_level=2,
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

    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    np.random.seed(836)

    background = background_level * np.ones(shape)
    exposure = np.ones(shape) + 0.5 * np.linspace(-1, 1, shape[0])

    flux = source_level * Tophat2DKernel(
        radius=source_radius, x_size=shape[1], y_size=shape[1], mode="oversample"
    ).array

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(flux * exposure, psf)

    counts = np.random.poisson(npred)
    return {
        "counts": counts,
        "psf": psf.array,
        "exposure": exposure,
        "background": background,
        "flux": flux
    }