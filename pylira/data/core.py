import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve_fft


__all__ = ["get_point_source_gauss_psf"]


def get_point_source_gauss_psf(sigma_psf=3):
    """Get point source with Gaussian PSF test data

    Parameters
    ----------


    Returns
    -------
    data : dict of `~numpy.ndarray`
        Data dictionary
    """
    np.random.seed(836)
    shape, shape_psf = (32, 32), (16, 16)

    psf = Gaussian2DKernel(sigma_psf, x_size=shape_psf[1], y_size=shape_psf[1]).array

    alpha_init = np.ones(psf.shape[0])

    background = 2 * np.ones(shape)
    signal = np.zeros(shape)
    signal[shape[0] // 2, shape[1] // 2] = 1000

    npred = background + convolve_fft(signal, psf)

    counts = np.random.poisson(npred)
    exposure = np.ones(shape)
    return {
        "observed_im": counts.astype(np.float64),
        "start_im": npred.astype(np.float64),
        "psf_im": psf.astype(np.float64),
        "expmap_im": exposure.astype(np.float64),
        "baseline_im": background.astype(np.float64),
        "alpha_init": alpha_init.astype(np.float64),
    }

