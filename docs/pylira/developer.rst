***********************
Developer Documentation
***********************

This is the developer documentation.

Running Tests
-------------

Note: running tests is no longer done using ``python setup.py test``. Instead
you will need to run::

    tox -e test

If you don't already have tox installed, you can install it with::

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest


Building Docs
-------------

Building the documentation is no longer done using
``python setup.py build_docs``. Instead you will need to run::

    tox -e build_docs

If you don't already have tox installed, you can install it with::

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html
