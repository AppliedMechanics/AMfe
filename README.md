AMfe - Finite Element Research Code at the Chair of Applied Mechanics
=====================================================================

(c) 2018 Lehrstuhl für Angewandte Mechanik, Technische Universität München


This Finite Element Research code is developed, maintained and used by a part of the numerics group of AM.

Coverage Report for Refactoring 2018:
[![coverage report](https://gitlab.lrz.de/AMCode/AMfe/badges/refactoring-2018/coverage.svg)](https://gitlab.lrz.de/AMCode/AMfe/commits/refactoring-2018)

Pipeline Status:
[![pipeline status](https://gitlab.lrz.de/AMCode/AMfe/badges/refactoring-2018/pipeline.svg)](https://gitlab.lrz.de/AMCode/AMfe/commits/refactoring-2018)

**Important Info:** The project has been moved recently. So please update your remote via:

    git remote set-url origin https://gitlab.lrz.de/AM/AMfe.git


Overview:
---------

1.  [Installation](#installation-of-amfe)
2.  [Documentation](#documentation)
3.  [Workflow](#workflow-for-pre--and-postprocessing)
4.  [Hints](#hints)


Installation of AMfe
--------------------

Before installing the AMfe package, check, if the latest python version and all necessary modules are installed.
For managing the python packages, the **Python distribution Anaconda** is **highly recommended**.
It has a very easy and effective packaging system and can thus handle all Python sources needed for this project.
For installation and usage of Anaconda checkout http://docs.continuum.io/anaconda/install#anaconda-install.

   - Python version 3.6 or higher
   - `numpy`, `scipy`, `pandas`, `h5py`, `matplotlib`, `dill`
   - for fast fortran execution a running fortran compiler (e.g. gcc and gfortran)
   - for building the documentation `sphinx`, `numpydoc`, `sphinx_rtd_theme`
   - for testing `nose`
   - for checking the code readability: `pylint`

I recommend to create a separate environment in anaconda for your amfe installation.
Then you later have the opportunity to create a new environment for other projects that can have different
requirements (such as python 2.7 instead 3.6). 

For getting the package type

    git clone https://gitlab.lrz.de/AMCode/AMfe.git

in your console. Git will clone the repository into the current folder.
For installing the package type

    python setup.py develop

in the main folder. This should build the fortran routines and install the python module in-place,
i.e. when you do changes to the source code they will be used the next time the module is loaded.

If you do not want to install the FORTRAN-routines, you can add the flag `no_fortran` to your installation command:

    python setup.py develop no_fortran

If no FORTRAN-compile is found, the installation will work only with the `no_fortran`-flag.

Documentation
-------------
Further documentation to this code is in the folder `docs/`. For building the documentation, go to the `docs/` folder
and type

    make html

The documentation will be built in the folder `docs/` available as html in `_build`.
If the command above does not work, the execution of `python setup.py build_sphinx` in the main-folder
also builds the documentation.

Workflow for Pre- and Postprocessing
------------------------------------
Preprocessing and postprocessing is not part of the code AMfe, but the open source tools gmsh
and Paraview are recommended:

- [gmsh](http://gmsh.info) The open-source meshing tool can create unstructured meshes for 2D and 3D geometries. The geometry can either be built inside the tool or outside in a CAD program with the `.stp`-file imported into gmsh. In order to define volumes for materials or points/lines/surfaces for boundaries, physical groups must be assigned in gmsh.
- [ParaView](http://www.paraview.org) With ParaView the results can be analyzed. For showing the displacements, usually it is very handy to apply the *Warp By Vector* filter to see the displaced configuration.

AMfe provides a Mesh-Exporter for the Preprocessor [GiD](http://www.gidhome.com/).
This preprocessor can be downloaded as community edition or can be accessed at our chair.
See the AM Wiki for more information how to access the professional edition at our chair.

Hints
-----

#### Python and the Scientific Ecosystem


Though Python is a general purpose programming language, it provides a great ecosystem for scientific computing.
As resources to learn both, Python as a language and the scientific Python ecosystem,
the following resources are recommended to become familiar with them.
As these topics are interesting for many people on the globe, lots of resources can be found in the internet.

##### Python language:
- [A byte of Python:](http://python.swaroopch.com/) A good introductory tutorial to Python. My personal favorite.
- [Learn Python the hard way:](http://learnpythonthehardway.org/book/) good introductory tutorial to the programming language.
- [Youtube: Testing in Python ](https://www.youtube.com/watch?v=FxSsnHeWQBY) This amazing talk explains the concept
and the philosophy of unittests, which are used in the `amfe` framework.

##### Scientific Python Stack (numpy, scipy, matplotlib):
- [Scipy Lecture Notes:](http://www.scipy-lectures.org/) Good and extensive lecture notes which are evolutionary improved online with very good reference on special topics, e.g. sparse matrices in `scipy`.
- [Youtube: Talk about the numpy data type ](https://www.youtube.com/watch?v=EEUXKG97YRw) This amazing talk **is a must-see** for using `numpy` arrays properly. It shows the concept of array manipulations, which are very effective and powerful and extensively used in `amfe`.
- [Youtube: Talk about color maps in matplotlib](https://youtu.be/xAoljeRJ3lU?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) This interesting talk is a little off-topic but cetainly worth to see. It is about choosing a good color-map for your diagrams.
- [Youtube: Talk about the HDF5 file format and the use of Python:](https://youtu.be/nddj5OA8LJo?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) Maybe of interest, if the HDF5 data structure, in which the simulation data are extracted, is of interest. This video is no must-have.

##### Version Control with git:
- [Cheat sheet with the important git commands](https://www.git-tower.com/blog/git-cheat-sheet/) Good cheatsheet with all the commands needed for git version control.
- [Youtube: git-Workshop](https://youtu.be/Qthor07loHM) This workshop is extensive and time intensive but definetely worth the time spent. It is a great workshop introducing the concepts of git in a well paced manner ([The slides are also available](https://speakerdeck.com/singingwolfboy/get-started-with-git)).
- [Youtube: git-Talk](https://youtu.be/ZDR433b0HJY) Very fast and informative technical talk on git. Though it is a little bit dated, it is definitely worth watching. 

##### gmsh:
- [HowTo for generating structured meshes in gmsh](https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH) This tutorial is about the generation of structured meshes in gmsh, in this case for the use in the CFD-Framework OpenFOAM. Nonetheless, everything can be used for AMfe as well.

#### IDEs:

A good IDE to start with is Spyder, which has sort of a MATLAB-Style look and feel.
It is part of anaconda ans provides nice features like built-in debugging, static code analysis with pylint and a
profiling tool to measure the performance of the code.

Other editors integrate very well into Python like Atom.

I personally work with PyCharm, which is an IDE for Python. However as it provides many functions one could be
overwhelmed by it at first. 

---------------------------------------
**Hint** 

On Mac OS X `Spyder 2` may run very slow, as there are some issues with the graphical frontent library, pyqt4. These issues are resolved on `Spyder 3` by using pyqt5, which can already be installed on anaconda as beta version resolving all these issues. To install `Spyder 3`, use either 

    conda update qt pyqt
    conda install -c qttesting qt pyqt
    conda install -c spyder-ide spyder==3.0.0b6
   
or (which worked better for me)
   
    pip install --pre -U spyder

-------------------------------------

#### Profiling the code

a good profiling tool is the cProfile module. It runs with

    python -m cProfile -o stats.dat myscript.py

The stats.dat file can be analyzed using the `snakeviz`-tool which is a Python tool which is available via `conda` or `pip` and runs with a web-based interface. To start run

    snakeviz stats.dat

in your console.


#### Theory of Finite Elements
The theory for finite elements is very well developed, though the knowledge is quite fragmented. When it comes to element technology for instance, good benchmarks and guidelines are often missed. A good guideline is the [Documentation of the CalculiX-Software-Package](http://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/ccx.html) which covers a lot about element technology, that is also used in AMfe. CalculiX is also an OpenSource Finite Element software written in FORTRAN an C++.
