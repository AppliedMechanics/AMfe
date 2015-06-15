How to run the AMfe Code
========================

You need the following stuff installed:

- Python 3.4 or higher
- numpy, scipy
- `ParaView <http://www.paraview.org>`_ for Postprocessing
- `gmsh <http://geuz.org/gmsh/>`_ for Preprocessing
- Some Python-Packages in order to build this documentation
   - sphinx
   - numpydoc

For Python exist several ways how to install it on your computer. We recommend to install Anaconda, which is a Package manager and lets you install easily all additional packages and tools related to Python.

After installing the `Anaconda Manager <https://store.continuum.io/cshop/anaconda/>`_ (make sure, that you install the Python 3 version of it) run in your bash-console

>>> conda install numpy
>>> conda install scipy
>>> conda install sphinx
>>> conda install numpydoc

For a Matlab-like Development center we recommend `Spyder <http://spyder-ide.blogspot.de>`_. Spyder can also easily be installed via Anaconda:

>>> conda install spyder


Getting into the code
"""""""""""""""""""""
For getting started and familiar with the code, we recommend to start with the examples. They show some cases that are working and are not too complicated. 


General Notes on the amfe finite element code
=============================================
The amfe finite element toolbox has the goal to provide a fast to develop, simple and out of the box usable finite element library for use in research. Therefore the focus is on flexibility to adapt the code to new problems and to easily implement a new method and not on runtime performance.


Indexing:
"""""""""

There is a conflict in different ecosystems, which indexing style is better:
starting a list with index 0 or with index 1. Both sides have different advantages and shortcomings; The main issue is, that the workflow in amfe has canges in indexing incorporated. They show up when the indexing style changes. So there are following indexing-ecosystems:

Index starts with 0:

- Python list and array indexing, and so is the amfe-code
- paraview

Index stars with 1:

- gmsh node-, line-, and element-numbering

So the rule is, that the system works on indexing 0, and the import data from gmsh are changed. So should be done when importing ANSYS-files as well.


Tips & Tricks:
====================

How to plot matrices in matplotlib:

>>> from matplotlib import pyplot as plt; import scipy as sp
>>> A = sp.random.rand(10, 10)
>>> plt.matshow(A)
>>> plt.colorbar()
>>> plt.set_cmap('jet') # 'jet' is default; others looking good are 'hot'

Plot on log scales:
>>> from matplotlib import pyplot as plt; import scipy as sp
>>> x = np.arange(200)
>>> plt.semilogy(x)
>>> plt.semilogx(x)
>>> plt.loglog(x)

Check out more on http://matplotlib.org/examples/color/colormaps_reference.html


