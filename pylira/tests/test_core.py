def test_import():
    import pylira

    assert pylira.__name__ == "pylira"


def test_c_extension():
    import pylira as p

    assert p.add(3, 4) == 7
