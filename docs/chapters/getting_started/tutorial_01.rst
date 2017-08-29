Tutorial 1: Static Problem
==========================

In this tutorial we want to walk through a full simulation process.
You will learn how to use the `MechanicalSystem()`-class which wraps all information about your problem and
provides methods for solving it.

Problem
-------

.. _pcmesh:
.. figure:: ../../static/img/tutorial_01/pressure_corner.svg
  :height: 750ex

  Mesh of pressure corner.

:numref:`pcmesh` shows a meshed 2D corner. It is meshed by the open-source
software `Gmsh <http://gmsh.info/>`_.
You can find the Mesh-File in AMfe's *examples* folder.
Every edge and every surface is associated with a physical-group-number.
The surface of the corner has the physical-group-number 11.
The small edges belong to physical group 9 and 10, and the outer and inner edges
belong to physical group 13 and 12, respectively.


Solving problem with AMfe
-------------------------

Setting up new simulation
^^^^^^^^^^^^^^^^^^^^^^^^^

For setting up a new simulation first import the amfe-package to your namespace
and initialize an instance of the :py:class:`MechanicalSystem <amfe.mechanical_system.MechanicalSystem>` class::

  import amfe
  my_system = amfe.MechanicalSystem()


The MechanicalSystem object will later contain all information about your simulation.
This includes the mesh information, materials, boundary conditions etc.

We save absolute paths for the mesh-file, which will be used as geometry input, and the output-file, which will be used for writing the results,
in variables::

  input_file = amfe.amfe_dir('meshes/gmsh/pressure_corner.msh')
  output_file = amfe.amfe_dir('results/pressure_corner/pressure_corner_linear')

The function :func:`amfe_dir(path) <amfe.tools.amfe_dir>` converts a relative path to the directory your amfe-package is installed to an absolute path.
The output file must be entered without file-extension.

Define materials
^^^^^^^^^^^^^^^^

Materials are defined by creating instances of
:py:class:`HyperelasticMaterial <amfe.material.HyperelasticMaterial>` class or
of its derived classes.
In this example we only use one material for the full domain::

  my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)

The statement above defines a St.-Venant-Kirchhoff-Material with parameters for steel and a thickness of 1 mm.
The plane_stress flag means that there are no forces in the third space dimension (out of plane) assumed.


Load Mesh
^^^^^^^^^

Next we have to load the mesh-information in the MechanicalSystem instance.
This can be done by the
:func:`MechanicalSystem.load_mesh_from_gmsh(mshfile, phygroup, material) <amfe.mechanical_system.MechanicalSystem.load_mesh_from_gmsh>` method.
It needs three parameters:
1. Path to Mesh-File
2. Physical-group-number which will be associated with material of the third parameter
3. Material

.. todo::
  Check if parameter 2 is correctly described?

We want to associate the whole surface with the previously defined Material `my_material`.
Therefore we run::

  my_system.load_mesh_from_gmsh(input_file, 11, my_material)


Apply boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^

In general there are two different types of boundary conditions:

  - Dirichlet boundary conditions (given displacements)
  - Neumann boundary conditions (given stresses)

As stated in problem-section the edges with physical group numbers 9 and 10
are enforced to only move in y and x direction, respectively.
Therefore we want to define two dirichlet boundary conditions which can be
applied by
:func:`apply_dirichlet_boundaries(physical_group, directions) <amfe.mechanical_system.MechanicalSystem.apply_dirichlet_boundaries>` ::

  my_system.apply_dirichlet_boundaries(9, 'x')
  my_system.apply_dirichlet_boundaries(10, 'y')

A Neumann boundary condition is needed to apply the normal forces on the
inner edges of the corner. This can be done by
:func:`apply_dirichlet_boundaries(physical_group, magnitude, directions, time_function) <amfe.mechanical_system.MechanicalSystem.apply_neumann_boundaries>` ::

  my_system.apply_neumann_boundaries(12, 1E9, 'normal', lambda t: t)

For details about how to use the methods for applying boundary conditions different to these,
see documentation for :py:class:`MechanicalSystem <amfe.mechanical_system.MechanicalSystem>` class.


Solve
^^^^^

At this point the problem is entirely defined.
The function :func:`solve_nonlinear_displacement <amfe.solve_nonlinear_displacement>`
solves the static problem::

  amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=50, track_niter=True)

The first parameter is the MechanicalSystem instance.
The second parameter defines the number of load steps used for the solution.
Track_niter is a flag states that the number of iterations will be saved to
attribute *iteration_info* of the MechanicalSystem instance.

For viewing the results one can use the MechanicalSystem data structure.
There is an exporter for the open-source postprocessing tool `Paraview <http://www.paraview.org/>`_ implemented.
The exporter is called by::

  my_system.export_paraview(output_file)

that needs an absolute path as argument for writing the output.
