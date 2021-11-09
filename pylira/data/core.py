import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft


__all__ = ["point_source_gauss_psf"]


def point_source_gauss_psf(
        shape=(32, 32),
        shape_psf=(16, 16),
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

    signal = np.zeros(shape)
    signal[shape[0] // 2, shape[1] // 2] = source_level

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1])
    npred = background + convolve_fft(signal, psf)

    counts = np.random.poisson(npred)
    exposure = np.ones(shape)
    return {
        "counts": counts,
        "psf": psf,
        "exposure": exposure,
        "background": background,
    }

