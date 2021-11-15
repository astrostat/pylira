from numpy.testing import assert_allclose
from pylira.utils.io import read_parameter_trace_file
from astropy.utils.data import get_pkg_data_filename


def test_read_parameter_trace_file():
    filename = get_pkg_data_filename("files/output-par.txt", package="pylira.data")

    table = read_parameter_trace_file(filename)

    assert table.colnames[:3] == ["iteration", "logPost", "stepSize"]
    assert_allclose(table["logPost"][1], 66.5734)
