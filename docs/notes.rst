How to run the AMfe Code
========================

You need the following stuff installed:

- `Python 3.4 <http://www.python.org>`_ or higher
- `numpy, scipy <http://www.scipy.org>`_
- `ParaView <http://www.paraview.org>`_ for Postprocessing
- `gmsh <http://geuz.org/gmsh/>`_ for Preprocessing
- Some Python-Packages in order to build this documentation
   - `sphinx <http://www.sphinx-doc.org/>`_
   - `numpydoc <https://pypi.python.org/pypi/numpydoc>`_

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
The amfe finite element toolbox has the goal to provide a fast to develop, simple and out of the box usable finite element library for use in research. Therefore the focus is on flexibility to adapt the code to new problems and to easily implement a new method and not on runtime performance. In the future, maybe some element formulations should be implemented in a fast, compiled language as `Cython <http://www.cython.org>`_, FORTRAN or C. By now, the amfe code is fully written in Python. 


Indexing:
"""""""""

There is a conflict in different ecosystems, which indexing style is better:
starting a list with index 0 or with index 1. Both sides have different advantages and shortcomings; The main issue is, that the workflow in amfe has changes in indexing incorporated. They show up when the indexing style changes. So there are following indexing-ecosystems:

Index starts with 0:

- Python list and array indexing, and so is the amfe-code
- paraview

Index stars with 1:

- FORTRAN
- gmsh node-, line-, and element-numbering

So the rule is, that the system works on indexing 0, and the import data from gmsh are changed. So should be done when importing ANSYS-files as well.


Tips & Tricks:
==============

How to plot matrices in matplotlib:

>>> from matplotlib import pyplot as plt; import scipy as sp
>>> A = sp.random.rand(10, 10)
>>> plt.matshow(A)
>>> plt.colorbar()
>>> plt.set_cmap('jet') # 'jet' is default; others looking good are 'hot'

How to show the sparsity pattern of a sparse matrix :code:`A_csr`:

>>> plt.spy(A_csr, marker=',')

You can use different markers, as :code:`','` are pixels and very small, they make sense when large matrices are involved. However, for small matrices, :code:`'.'` gives a good picture. 

Plot on log scales:

>>> from matplotlib import pyplot as plt; import scipy as sp
>>> x = np.arange(200)
>>> # plot y with logscale
>>> plt.semilogy(x)
>>> # plot x with logscale
>>> plt.semilogx(x)
>>> # plot x and y in logscale
>>> plt.loglog(x)

Check out more on http://matplotlib.org/examples/color/colormaps_reference.html


FORTRAN
=======

It seems that FORTRAN is a very good companion to Python in order to speed the time critical things up. It is possible to write functions in fortran that are executed at lightspeed, especially when loops or matrix-vector-multiplications are heavily involved. 

As wrapper the tool `f2py` can be used. It is included in numpy and gives the full support for numpy-arrays. 


gmsh
====

Some information on gmsh would be cool here; how to use it in an efficient way... 