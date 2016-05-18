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

Before installing the AMfe package, check, if the latest python version and all necessary modules are installed. For managing the python packages, the **Python distribution Anaconda** is **highly recommended**. It has a very easy and effective packaging system and can thus handle all Python sources needed for this project. For installation and usage of Anaconda checkout http://docs.continuum.io/anaconda/install#anaconda-install.

   - Python version 3.5 or higher
   - `numpy`, `scipy` and `pandas`
   - for fast fortran execution a running fortran compiler (e.g. gcc)
   - for building the documentation `sphinx` and `numpydoc`
   - for checking the code: `pylint`

For getting the package type

    git clone git@gitlab.lrz.de:AMCode/AMfe.git

in your console. Git will clone the repository into the current folder.
For installing the package type

    python setup.py develop

in the main folder. This should build the fortran routines and install the python module in-place, i.e. when you do changes to the source code they will be used the next time the module is loaded.

If you do not want to install the FORTRAN-routines, you can add the flag `no_fortran` to your installation command:

    python setup.py develop no_fortran

If no FORTRAN-compile is found, the installation will work only with the `no_fortran`-flag.

For getting the full speed of the Intel MKL library, which provides a fast solver for sparse systems, install `pyMKL` by running

    git clone https://github.com/Rutzmoser/pyMKL.git
    cd pyMKL
    python setup.py install

which installs the pyMKL library. After that run, you may delete the folder `pyMKL`. 

2. Documentation
----------------
Further documentation to this code is in the folder `docs/`. For building the documentation, type

    python setup.py build_sphinx

The documentation will be built in the folder `docs/` available as html in `build`.
**Attention** There is a bug in the recent versions of sphinx, where the `@`-operator for the matrix-multiplication cannot be resolved. To overcome the problem downgrade the sphinx-version to `1.3.1`, where this bug is not present, by typing `conda install sphinx=1.3.1`.

3. Fortran-Routines
-------------------
In order to use the fast Fortran routines, which are used within the assembly process, a working Fortan compiler (e.g. `gfortran`, `gfortran-4.8`) has to be installed.


4. Hints
-----------

### Python and the Scientific Ecosystem
Though Python is a general purpose programming language, it provides a great ecosystem for scientific computing. As resources to learn both, Python as a language and the scientific Python ecosystem, the following resources are recommended to become familiar with them. As these topics are interesting for many people on the globe, lots of resources can be found in the internet.

##### Python language:
- [A byte of Python:](http://python.swaroopch.com/) A good introductory tutorial to Python. My personal favorite.
- [Learn Python the hard way:](http://learnpythonthehardway.org/book/) good introductory tutorial to the programming language.
- [Youtube: Testing in Python ](https://www.youtube.com/watch?v=FxSsnHeWQBY) This amazing talk explains the concept and the philosophy of unittests, which are used in the `amfe` framework.

##### Scientific Python Stack (numpy, scipy, matplotlib):
- [Scipy Lecture Notes:](http://www.scipy-lectures.org/) Good and extensive lecture notes which are evolutionary improved online with very good reference on special topics, e.g. sparse matrices in `scipy`.
- [Youtube: Talk about the numpy data type ](https://www.youtube.com/watch?v=EEUXKG97YRw) This amazing talk **is a must-see** for using `numpy` arrays properly. It shows the concept of array manipulations, which are very effective and powerful and extensively used in `amfe`.
- [Youtube: Talk about color maps in matplotlib](https://youtu.be/xAoljeRJ3lU?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) This interesting talk is a little off-topic but cetainly worth to see. It is about choosing a good color-map for your diagrams.
- [Youtube: Talk about the HDF5 file format and the use of Python:](https://youtu.be/nddj5OA8LJo?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) Maybe of interest, if the HDF5 data structure, in which the simulation data are extracted, is of interest. This video is no must-have.


### IDEs:

The best IDE for Python is Spyder, which has sort of a MATLAB-Style look and feel. Other editors integrate very well into Python like Atom, as well as PyCharm, which is an IDE for Python.

I personally work with Spyder and Atom. Spyder is part of anaconda and can be installed via

     conda install spyder

Spyder also provides nice features like built-in debugging, static code analysis with pylint and a profiling tool to measure the performance of the code.

### Profiling the code

a good profiling tool is the cProfile moudule. It runs with

    python -m cProfile -o stats.dat myscript.py

The stats.dat file can be analyzed using the `snakeviz`-tool which is a Python tool which is available via `conda` or `pip` and runs with a web-based interface. To start run

    snakeviz stats.dat

in your console.


### Theory of Finite Elements
The theory for finite elements is very well developed, though the knowledge is quite fragmented. When it comes to element technology for instance, good benchmarks and guidelines are often missed. A good guideline is the [Documentation of the CalculiX-Software-Package](http://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/ccx.html) which covers a lot about element technology, that is also used in AMfe. CalculiX is also an OpenSource Finite Element software written in FORTRAN an C++.
