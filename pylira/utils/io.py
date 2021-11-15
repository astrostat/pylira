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
    # and provide meat information
    table = Table.read(filename, format="ascii")
    return table
