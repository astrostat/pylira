import numpy as np
from astropy.table import Table


def read_parameter_trace_file(filename):
    """Read LIRA parameter output file

    Parameters
    ----------
    filename : str or Path
        File name.

    Returns
    -------
    table :  `~astropy.table.Table`
        Parameter table with one row per iteration.
    """
    # TODO: maybe rename columns to more descriptive names
    # and provide meta information
    table = Table.read(filename, format="ascii")
    return table


def read_image_trace_file(filename):
    """Read LIRA image trace file

    Parameters
    ----------
    filename : str or Path
        File name.

    Returns
    -------
    trace :  `~numpy.ndarray`
        Three dimensional numpy array of the shape (niter, ny, nx)
        representing the trace of the output image.
    """
    data = np.loadtxt(filename)

    shape_y, shape_x = data.shape
    n_iter = shape_y // shape_x

    return data.reshape((n_iter, shape_x, shape_x))
