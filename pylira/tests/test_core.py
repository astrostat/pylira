import pylira


def test_import_name():
    assert pylira.__name__ == "pylira"


def test_image_analysis():
    assert pylira.image_analysis is not None
