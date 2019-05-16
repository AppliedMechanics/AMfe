.. AMfe documentation master file, created by
   sphinx-quickstart on Tue Jun  9 20:54:07 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

##################
AMfe Documentation
##################


.. rubric:: The AMfe Documentation

AMfe is built in order to provide a simple and lean, yet powerful finite element framework for research purpose.


**********************************
How the documentation is organized
**********************************

    * :doc:`Tutorials (Getting started Guide)<chapters/getting_started/index>`
      are a good starting point to learn how to use AMfe. Start here if you are
      new to AMfe.

    * :doc:`Fundamentals (Topic guides)<chapters/fundamentals/index>`
      explains different parts and concepts of AMfe. It is the heart of
      documentation and mostly intended for users that are familiar with basics
      of AMfe or users that have already done the Tutorials

    * :doc:`Examples<chapters/examples/index>` shows some examples for
      different tasks. This part of documentation can be used if is interested
      in how to solve specific problem. For many problems an example can be
      found here that can be altered for own needs.

    * :doc:`Reference<chapters/package_doc/index>` is the API documentation
      of the whole package.




The idea
--------

The idea of how AMfe is used is quite different to classic finite element
programs. In classic finite element programs the process of analyses is strictly
divided in three steps: Preprocessing, Solving, Postprocessing.

In AMfe you have to do these three tasks, too, but you are not limited to do
them strictly in that order.
While classic finite element programs need an input file
which is passed to the solver, AMfe is interpreted. The advantage of this
structure is that you are very flexible during the simulation.
Assume you have done a transient analysis. But afterwards you realize that you
need some further time steps to simulate. Then you only have to call the
solver to solve the system for these timesteps. There is no new mesh generation
or preallocation of global vectors needed. For linear systems even a new
assembly is not needed. Classic programs do not allow this or have a very
difficult API to achieve the same to use old simulation data for a new solution.
