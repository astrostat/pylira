import pylira


def test_import_name():
    assert pylira.__name__ == "pylira"


def test_c_extension_function():
    assert pylira.add(3, 4) == 7


def test_c_extension_struct():
    p = pylira.Pet("Molly")
    assert p.getName() == "Molly"

    p.setName("Charly")
    assert p.getName() == "Charly"

