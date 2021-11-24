import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS


def read_parameter_trace_file(filename, format="ascii"):
    """Read LIRA parameter output file

    Parameters
    ----------
    filename : str or Path
        File name.
    format : {"ascii", "fits"}
        Table format

    Returns
    -------
    table :  `~astropy.table.Table`
        Parameter table with one row per iteration.
    """
    # TODO: maybe rename columns to more descriptive names
    # and provide meta information
    table = Table.read(filename, format=format)
    return table


def read_image_trace_file(filename, format="ascii"):
    """Read LIRA image trace file

    Parameters
    ----------
    filename : str or Path
        File name.
    format : {"ascii", "fits"}
        Table format

    Returns
    -------
    trace :  `~numpy.ndarray`
        Three dimensional numpy array of the shape (niter, ny, nx)
        representing the trace of the output image.
    """
    if format == "ascii":
        data = np.loadtxt(filename)

        shape_y, shape_x = data.shape
        n_iter = shape_y // shape_x

        return data.reshape((n_iter, shape_x, shape_x))
    elif format == "fits":
        return fits.getdata(filename, hdu="IMAGE_TRACE")
    else:
        raise ValueError(f"Not a supported format {format}")


def write_to_fits(result, filename, overwrite):
    """Write LIRA result to FITS.

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

    config_hdu = fits.BinTableHDU(result.config_table, name="CONFIG")

    hdulist = fits.HDUList([
        primary_hdu, parameter_trace_hdu, image_trace_hdu, config_hdu
    ])

    hdulist.writeto(filename, overwrite=overwrite)


def read_from_fits(filename):
    """Read LIRA result from FITS.

    Parameters
    ----------
    filename : `Path`
        Output filename

    Returns
    -------
    result : dict
       Dictionary with init parameters for `LIRADeconvolverResult`
    """
    hdulist = fits.open(filename)

    wcs = WCS(hdulist["PRIMARY"].header)

    config_table = Table.read(hdulist["CONFIG"])
    config = dict(config_table[0])

    paramter_trace = Table.read(hdulist["PARAMETER_TRACE"])
    posterior_mean = hdulist["PRIMARY"].data

    # define location for lazy loading
    image_trace = {"filename": filename, "format": "fits"}
    return {
        "posterior_mean": posterior_mean,
        "config": config,
        "parameter_trace": paramter_trace,
        "image_trace": image_trace,
        "wcs": wcs
    }


IO_FORMATS_WRITE = {
    "fits": write_to_fits
}

IO_FORMATS_READ = {
    "fits": read_from_fits
}
