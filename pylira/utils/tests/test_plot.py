from astropy.utils.data import get_pkg_data_filename
from pylira.utils.plot import plot_parameter_traces, plot_parameter_distributions, plot_pixel_trace
from pylira.utils.io import read_parameter_trace_file, read_image_trace_file


def test_plot_parameter_traces():
    filename = get_pkg_data_filename("files/output-par.txt", package="pylira.data")
    table = read_parameter_trace_file(filename)

    axes = plot_parameter_traces(table)
    assert axes.size == 9


def test_plot_parameter_distributions():
    filename = get_pkg_data_filename("files/output-par.txt", package="pylira.data")
    table = read_parameter_trace_file(filename)

    axes = plot_parameter_distributions(table)
    assert axes.size == 9


def test_plot_pixel_trace():
    filename = get_pkg_data_filename("files/output.txt", package="pylira.data")

    image_trace = read_image_trace_file(filename)

    ax = plot_pixel_trace(image_trace=image_trace, center_pix=(3, 3), )
    assert ax is not None
