.. toctree::
    :hidden:
    
    installation_git.rst
    installation_sdist.rst
    installation_wheel.rst

Installation
============

The Installation process needs three steps:

1. Install a python distribution (if not installed on your computer yet)
2. Install AMfe
3. Install pyMKL for speedup (optional but highly recommended)


Install Python Distribution
---------------------------

Before you are able to install AMfe
you need a Python environment.
We recommend to install the Anaconda distribution.
It is a free python distribution which is both available for windows and for linux/mac and
focuses on scientific computing which is best suitable for programs like AMfe.
The distribution may be used for business or private purposes.
You do not need root access to install Anaconda.
You may install it in your home-directory.

How to install Anaconda
^^^^^^^^^^^^^^^^^^^^^^^

  1. Download Anaconda from http://www.continuum.io/downloads.
  2. Install Anaconda with installation instructions from http://www.continuum.io/

  .. important:: Make sure that you install an Anaconda version that provides a Python-version newer or equal Python 3.5

  .. note:: We recommend to add anaconda to your PATH-variable after installation.

When you have installed anaconda successfully the most important packages for
using AMfe, such as numpy and scipy, should also have been installed automaticly.

Other Distributions
^^^^^^^^^^^^^^^^^^^

If you do not want to install anaconda and want to use another python distribution,
check if you have a python version greater than 3.5 and that you have installed the newest versions of the packages listed in
the following table:

 +---------------+--------+--------+--------+--------+----------+-------+
 | Module        | Numpy  | Scipy  | Pandas | Sphinx | Numpydoc | h5py  |
 +---------------+--------+--------+--------+--------+----------+-------+
 | Min. Version  | 1.11.1 | 0.18.1 | 0.18.1 | 1.4.6  | 0.6.0    | 2.6.0 |
 +---------------+--------+--------+--------+--------+----------+-------+

Usually you can check the version number by first opening a pyhton3 console via:: bash

  user@host:~$ python3

and then entering::

  >>> import modulename
  >>> print(modulename.__version__)


Python Editor/IDE
^^^^^^^^^^^^^^^^^

For a matlab like environment we recommend Spyder.
Spyder includes a python editor with syntax-highlighting, IPython Condole,
Documentation-Window, Variable explorer and more.
In Anaconda it is usually already installed.


Install AMfe
------------

There are different ways to install AMfe.
The way which is best for you depends on your application and plans for using AMfe.
The following table lists different ways and potiential users

+-----------------------+------------------------------------------------------+---------------+
| Way of installation   | Recommended for                                      | Platforms     | 
+-----------------------+------------------------------------------------------+---------------+
| Git clone/fork        | Developers, Users who want to change/edit            | linux/win/mac |
|                       | source-code or want to contribute to AMfe-project    |               |
+-----------------------+------------------------------------------------------+---------------+
| Non-Developer         | Users who like to build source-code by themselves,   | linuc/win/mac |
| source-code           | but who do not intend to change the source-code.     |               |
|                       | People who do not intend to change the source        |               |
|                       | but want best performance of AMfe by using compilers |               |
|                       | on their own target machine.                         |               |
+-----------------------+------------------------------------------------------+---------------+
| Already built code    | Users who are not familiar with compilers/computers  | linux/win/mac |
|                       | or users who want a system out of the box.           |               |
+-----------------------+------------------------------------------------------+---------------+



Click on one of the following links for details:

:ref:`1. Installation git clone/fork <installation_git>`

:ref:`2. Installation Non-Developer source-code <installation_sdist>`

:ref:`3. Installation Already built code <installation_wheel>`







Install pyMKL
-------------

For speedup of the solver-routines, it is recommended to install pyMKL.
This is package is a wrapper for speed-up the Pardiso-Solver by using the Intel Math Kernel Library.
The package pyMKL can be downloaded from https://github.com/c-meyer/pyMKL

Download the repository, unzip it to an arbitrary folder change to it and run::

    python setup.py install
    

This command installs the pyMKL-package to your python environment.




.. todo: shift this section elsewhere:

Building Releases For Developers
--------------------------------

1. Git version: Nothing to do. Just push to release-branch with all updated files.
2. Source-Release: Run python setup.py sdist
3. Wheels: python setup.py bdist_wheel  Important: It must be build for every platform and python version.






