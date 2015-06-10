How to run the AMfe Code
========================

You need the following stuff installed: 

- Python 3.4 or higher
- numpy, scipy
- ParaView for Postprocessing
- gmsh for Preprocessing
- Some Python-Packages in order to build this documentation
	- sphinx
	- numpydoc

For Python exist several ways how to install it on your computer. We recommend to install Anaconda. Anaconda is a Package manager and lets you install easily all additional packages and tools related to Python. 

After installing the Anaconda Manager from https://store.continuum.io/cshop/anaconda/ run in your bash-console

>>> conda install numpy
>>> conda install scipy
>>> conda install sphinx
>>> conda install numpydoc

For a Matlab-like Development center we recommend Spyder. Spyder can also easily be installed via Anaconda:

>>> conda install spyder3




General Notes on the amfe finite element code
===============================================


Meshing:
----------

Indexing:
""""""""""""

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

How to plot matrices efficiently in matplotlib:

>>> from matplotlib import pyplot as plt; import scipy as sp
>>> A = sp.random.rand(10, 10)
>>> plt.matshow(A)
>>> plt.colorbar()
>>> plt.set_cmap('jet') # 'jet' is default; others looking good are 'hot'

Check out more on
__ http://matplotlib.org/examples/color/colormaps_reference.html
