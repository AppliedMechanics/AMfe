.. toctree::
    :hidden:
    
    installation_git.rst
    installation_sdist.rst
    installation_wheel.rst

Installation
============

The Installation process needs two steps:

1. Install a python distribution (if not installed on your computer yet)
2. Install AMfe


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


Python Editor/IDE
^^^^^^^^^^^^^^^^^

For a matlab like environment we recommend Spyder.
Spyder includes a python editor with syntax-highlighting, IPython Condole,
Documentation-Window, Variable explorer and more.
In Anaconda it is usually already installed.

For development we recommend the IDE PyCharm it has many features and is recommended
for those that that have a bit experience in programming.


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
| Already built code    | Users who are not familiar with compilers/computers  | linux/(mac?)  |
|                       | or users who want a system out of the box.           |               |
+-----------------------+------------------------------------------------------+---------------+



Click on one of the following links for details:

:ref:`1. Installation git clone/fork <installation_git>`

:ref:`2. Installation Non-Developer source-code <installation_sdist>`

:ref:`3. Installation Already built code <installation_wheel>`


Remarks on AMfe-versions
------------------------
Release-tags have three digits separated by points. These release-tags increase from one version of AMfe to the
next one, depending on the amount and range of new features and changes. Each version of the Master-branch gets a
new release-tag based on the following classification:

X.Y.Z =>    X: major release => large amount of new features and changes or major restructuring compared to previous
                                major release
            Y: minor release => new features and contributions
            Z: bug fixes     => bug fixes

For users and developers it is generally recommended to check the release notes of a new version, because APIs
might change especially when upgrading to a new major release.


