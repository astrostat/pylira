#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import os
import subprocess
import sys
from pathlib import Path
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

# First provide helpful messages if contributors try and run legacy commands
# for tests or docs.

TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:

    tox -e test

If you don't already have tox installed, you can install it with:

    pip install tox

If you only want to run part of the test suite, you can also use pytest
directly with::

    pip install -e .[test]
    pytest

For more information, see:

  http://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if "test" in sys.argv:
    print(TEST_HELP)
    sys.exit(1)

DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:

    tox -e build_docs

If you don't already have tox installed, you can install it with:

    pip install tox

You can also build the documentation with Sphinx directly using::

    pip install -e .[docs]
    cd docs
    make html

For more information, see:

  http://docs.astropy.org/en/latest/install.html#builddocs
"""

if "build_docs" in sys.argv or "build_sphinx" in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)

VERSION_TEMPLATE = """
# Note that we need to fall back to the hard-coded version if either
# setuptools_scm can't be imported or setuptools_scm can't determine the
# version, so we catch the generic 'Exception'.
try:
    from setuptools_scm import get_version
    version = get_version(root='..', relative_to=__file__)
except Exception:
    version = '{version}'
""".lstrip()


sources = ["pylira/src/lirabind.cpp"]
include_dirs = ["pylira/src/"]

# Add R install dir

process = subprocess.run(["R", "RHOME"], capture_output=True, encoding="utf-8")
r_home = Path(process.stdout.strip())

if r_home.exists():
    print(f"Found existing R installation in {r_home}")
else:
    raise ValueError("pylira requires a working R installation")

include_dirs += [str(r_home / "include/")]
library_dirs = [
    str(r_home.parent),
    str(r_home / "library"),
]

ext_modules = [
    Pybind11Extension(
        name="_lira",
        sources=sources,
        include_dirs=include_dirs,
        libraries=["Rmath"],
        library_dirs=library_dirs,
    ),
]

setup(
    use_scm_version={
        "write_to": os.path.join("pylira", "version.py"),
        "write_to_template": VERSION_TEMPLATE,
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
