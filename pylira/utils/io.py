import numpy as np
from astropy.table import Table
from astropy.io import fits


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


def write_to_fits_hdulist(result, filename, overwrite):
    """Convert LIRA result to list of FITS hdus.

    Parameters
    ----------
    result : `LIRADeconvolverResult`
        Deconvolution result.
    filename : `Path`
        Output filename
    overwrite : bool
        Overwrite file.
    """
    if result.wcs:
        header = result.wcs.to_header()
    else:
        header = None

    primary_hdu = fits.PrimaryHDU(
        header=header, data=result.posterior_mean,
    )

    table = result.parameter_trace.copy()
    table.meta = None
    parameter_trace_hdu = fits.BinTableHDU(
        table, name="PARAMETER_TRACE"
    )

    image_trace_hdu = fits.ImageHDU(
        header=header, data=result.image_trace, name="IMAGE_TRACE"
    )

    config = Table()

    for key, value in result.config.items():
        if key == "alpha_init":
            value = [value]
        config[key] = value

    config_hdu = fits.BinTableHDU(config, name="CONFIG")

    hdulist = fits.HDUList([
        primary_hdu, parameter_trace_hdu, image_trace_hdu, config_hdu
    ])

    hdulist.writeto(filename, overwrite=overwrite)


IO_FORMATS = {
    "fits": write_to_fits_hdulist
}
