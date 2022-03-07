import numpy as np


def f_hyperprior_esch(alpha, delta=1, index_alpha=1, index=0):
    """Hyperprior distribution following Esch et. al. 2004

    Parameters
    ----------
    alpha : float or `~numpy.ndarray`
        Alpha parameter
    delta : float or `~numpy.ndarray`
        Delta parameter
    index_alpha : float or `~numpy.ndarray`
        Index alpha parameter
    index : float or `~numpy.ndarray`
        Index parameter

    Returns
    -------
    value : `~numpy.ndarray`
        Value
    """
    prefactor = (alpha * delta) ** index
    exponential = np.exp(-delta * alpha ** index_alpha / index_alpha)
    return prefactor * exponential


def f_hyperprior_lira(alpha, ms_al_kap1=0, ms_al_kap2=1000, ms_al_kap3=3):
    """Hyperprior distribution following LIRA implementation

    Parameters
    ----------
    alpha : float or `~numpy.ndarray`
        Alpha parameter
    ms_al_kap1: float or `~numpy.ndarray`
        Multiscale prior parameter.
    ms_al_kap2: float or `~numpy.ndarray`
        Multiscale prior parameter.
    ms_al_kap3: float or `~numpy.ndarray`
        Multiscale prior parameter.

    Returns
    -------
    value : `~numpy.ndarray`
        Value

    """
    delta = ms_al_kap2 * ms_al_kap3
    index = ms_al_kap1
    index_alpha = ms_al_kap3
    return f_hyperprior_esch(alpha=alpha, delta=delta, index=index, index_alpha=index_alpha)
