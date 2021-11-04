import numpy as np
from numpy.testing import assert_allclose
import pylira


def test_import_name():
    assert pylira.__name__ == "pylira"


def test_c_extension_function():
    assert pylira.add(3, 4) == 7


def test_c_extension_function_numpy_vectorize():
    a = np.array([1, 2, 3])
    b = np.array([4, 3, 2])
    result = pylira.add(a, b)

    assert result.dtype == "int32"
    assert_allclose(result, 5)


def test_c_extension_function_numpy():
    a = np.array([1, 2, 3])
    b = np.array([4, 3, 2])
    result = pylira.add_arrays(a, b)

    # this casts the type on input
    assert result.dtype == "float64"
    assert_allclose(result, 5)


def test_c_extension_struct():
    p = pylira.Pet("Molly")
    assert p.getName() == "Molly"

    p.setName("Charly")
    assert p.getName() == "Charly"
