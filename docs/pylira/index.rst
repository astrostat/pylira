***************
Getting Started
***************

Pylira is a Python package for Bayesian low-counts image reconstruction and analysis.
See e.g. [Esch2004]_, [Connors2007]_, [Connors2011]_ and [Stein2015]_.


Installation
============
Pylira requires a working `R` installation to build against. On Mac-OS you can typically use::

    brew install r

On Linux based systems the following should work::

    sudo apt-get install r-base-dev r-base r-mathlib

Once `R` is installed you can install Pylria directly from source using::

    git clone https://github.com/astrostat/pylira.git
    cd pylira
    pip install -e .

