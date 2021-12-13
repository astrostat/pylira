import numpy as np


__all__ = ["fftconvolve"]


def _centered(arr, newshape):
    # Return the center newshape portion of the array.
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


def fftconvolve(image, kernel):
    """Convolve an image with a kernel

    Minimal pure Numpy re-implementation of `scipy.signal.fftconvolve` jut to
    avoid the dependency for now.

    Parameters
    ----------
    image : `~numpy.ndarray`
        Image array
    kernel : `~numpy.ndarray`
        Kernel array

    Returns
    -------
    result : `~numpy.ndarray`
        Convolution result.
    """
    shape_image = image.shape
    shape_kernel = kernel.shape

    shape = [shape_image[i] + shape_kernel[i] - 1 for i in range(image.ndim)]

    ftimage = np.fft.rfft2(image, s=shape)
    ftkernel = np.fft.rfft2(kernel, s=shape)
    result = np.fft.irfft2(ftimage * ftkernel)
    return _centered(result, image.shape)
