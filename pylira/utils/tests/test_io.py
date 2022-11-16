from numpy.testing import assert_allclose
from astropy.utils.data import get_pkg_data_filename
from pylira.utils.io import read_image_trace_file, read_parameter_trace_file


def test_read_parameter_trace_file():
    filename = get_pkg_data_filename("files/output-par.txt", package="pylira.data")

    table = read_parameter_trace_file(filename)

    assert table.colnames[:3] == ["iteration", "logPost", "stepSize"]
    assert_allclose(table["logPost"][1], 66.5734)


def test_read_image_trace_file():
    filename = get_pkg_data_filename("files/output.txt", package="pylira.data")

    data = read_image_trace_file(filename)

    assert data.shape == (10, 32, 32)
    assert_allclose(data[0][16][16], 979.513855, rtol=1e-6)
