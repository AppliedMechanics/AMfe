AMfe - Finite Element Research Code at the Chair of Applied Mechanics
---------------------------------------------------------------------
(c) 2016 Lehrstuhl für Angewandte Mechanik, Technische Universität München


This Finite Element Research code is developed, maintained and used by a part of the numerics group of AM.


Overview:
---------

1.  [Installation](#1-installation)
2.  [Documentation](#2-documentation)
3.  [Fortran-Routines](#3-fortran-routines)
4.  [Hints](#4-hints)


1. Installation
--------------

Before installing the AMfe package, check, if the latest python version and all necessary modules are installed. For managing the python packages, the distribution *Anaconda* is highly recommended. It has a very easy and effective packaging system and can thus handle all Python sources needed for this project. For installation and usage of Anaconda checkout http://docs.continuum.io/anaconda/install#anaconda-install.

   - Python version 3.5 or higher
   - `numpy`, `scipy` and `pandas`
   - for fast fortran execution a running fortran compiler (e.g. gcc)
   - for building the documentation `sphinx` and `numpydoc`
   - for checking the code: `pylint`

For installing the package type

    python setup.py develop

in the main folder. This should build the fortran routines and install the python module in-place, i.e. when you do changes to the source code they will be used the next time the module is loaded.

If you do not want to install the FORTRAN-routines, you can add the flag `no_fortran` to your installation command:

    python setup.py develop no_fortran


2. Documentation
----------------
Further documentation to this code is in the folder `docs/`. For building the documentation, type

    python setup.py build_sphinx

The documentation will be built in the folder `docs/` available as html in `build`. 

3. Fortran-Routines
-------------------
In order to use the fast Fortran routines, which are used within the assembly process, a working Fortan compiler (e.g. `gfortran`, `gfortran-4.8`) has to be installed.


4. Hints
-----------

### Sphinx:

`sphinx` has to be installed for `python3`. Maybe, `sphinx` was automatically installed for `python2`.
```python
Using `python3`, one can test which `sphinx`-version is installed:
python3
>>> import sphinx
>>> sphinx.__version__
```
The version shuld be at least `'1.3.1'`.


### IDEs:

The best IDE for Python is Spyder, which has sort of a MATLAB-Style look and feel. Other editors integrate very well into Python like Atom, as well as PyCharm, which is an IDE for Python. 

I personally work with Spyder and Atom. Spyder is part of anaconda and can be installed via 
     
     conda install spyder
     
Spyder also provides nice features like built-in debugging, static code analysis with pylint and a profiling tool to measure the performance of the code. 