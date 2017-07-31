How AMfe is structured
======================


Core Modules
------------

AMfe has a very modular structure. There are seven core modules which are needed for
nonlinear finite element simulations:

* Mesh :py:mod:`amfe.mesh`
* Material :py:mod:`amfe.material`
* Element :py:mod:`amfe.element`
* Assembly :py:mod:`amfe.assembly`
* Boundary :py:mod:`amfe.boundary`
* Mechanical System :py:mod:`amfe.mechanical_system`
* Solver :py:mod:`amfe.solver`


Each module undertakes a specific task in the progress of finite element
analyzes. :numref:`fig_amfestructure` shows how the core modules' main classes
interact with each other.
The Mechanical System module is the head of all modules.
It provides the :py:class:`amfe.mechanical_system.MechanicalSystem` class
which keeps most tasks needed to do a finite element analysis
in a short and very easy to learn API. The class' attributes have pointers
to instances of classes from other modules like Mesh, Assembly or Boundary.
These relations are represented by the arrows in :numref:`fig_amfestructure`.
Thus, it puts all together in a handy interface.

The main tasks of each module can be summarized as follows:

* **Mesh:** The mesh module provides a Mesh class which handles everything
  regarding the geometry and mesh topology.
  It gives access to node coordinates and element topology.
  It also provides import-functions for several mesh formats like gmsh or Ansys.
  Furthermore several meshes can be connected by mesh tying techniques.

* **Assembly:** The assembly module provides an Assembly class which provides
  methods for assembly. These methods are usually called in every timestep of
  the simulation when new internal and external force vectors and tangent
  stiffness matrices are needed for time integration and the newton solvers.
  The class knows the mapping between local and global degrees of freedom and
  also has a pointer to a mesh-class and thus knows how the elements have
  to be assembled.

* **Element:** The Element module provides an Element class with all methods
  needed to return element properties such as the nonlinear internal element
  force vector which is needed for assembling. Each element-type that is
  implemented in AMfe inherits from this base Element class. This structure
  makes it very easy to implement new elements as only a new subclass has to be
  generated where only the calculation of element forces etc. have to be
  implemented.

* **Material:** The Material module provides Material classes for different
  materials. Like in the Element module, there exists a basis Material class
  where all different material classes for different consitutive laws
  inherit from. Ínstances of element class have a pointer to an instance of a
  material class which makes it very easy to map elements to materials.

* **Boundary:** The Boundary class helps to define Dirichlet boundary
  conditions.

* **Mechanical System:** As already mentioned, the Mechanical System module
  provides a MechanicalSystem-class that keeps all together and provides a
  very easy API for all tasks needed in a finite element analysis.

* **Solver:** The solver module provides numerical solver for different
  analyses. For example it provides time integrators, eigensolvers and linear
  solvers.


.. _fig_amfestructure:
.. figure:: ../../static/img/fundamentals/Amfe_structure.svg
  :height: 750ex

  Core Modules of Amfe.



Model Order Reduction
---------------------

A main application for AMfe is model order reduction of nonlinear mechanical
systems. AMfe provides many tools for model order reduction. Therefore a
subpackage **hyper_red** was created.

.. todo::

  Explain hyper_red modules and Mechanical System MOR-classes
