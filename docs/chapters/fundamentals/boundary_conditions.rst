Boundary Conditions
===================

In general there are two types of boundary conditions one can impose on structures:

1. Dirichlet Boundary Conditions
2. Neumann Boundary Conditions

Dirichlet Boundary Conditions impose displacements on certain degrees of freedom.
Neumann Boundary Conditions impose forces on certain degrees of freedom.

This guide shows how boundary conditions are imposed in AMfe

.. note::

    This guide goes a bit in detail how AMfe treats boundary conditions. If you only want to know how to impose the
    boundary conditions in a convenient way, then it is recommended to read the guide about mechanical_system module
    instead.



Dirichlet Boundary Conditions
-----------------------------



.. _tab_mesh_no_properties:

.. table:: Important properties that control the dirichlet boundary conditions

    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | Property                                                                                                        | Description                                                                                                            |
    +=================================================================================================================+========================================================================================================================+
    | :py:attr:`Mesh.nodes_dirichlet<amfe.mesh.Mesh.nodes_dirichlet>`                                                 | Contains a unique set of all node-ids where Dirichlet boundary conditions are imposed to                               |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`Mesh.dofs_dirichlet<amfe.mesh.Mesh.dofs_dirichlet>`                                                   | Contains a unique set of all global dof-ids where Dirichlet boundary conditions are imposed to                         |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.slave_dofs<amfe.boundary.DirichletBoundary.slave_dofs>`                             | Contains the constrained dofs                                                                                          |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.B<amfe.boundary.DirichletBoundary.B>`                                               | A mapping matrix between the free dofs of the unconstrained and the constrained system                                 |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.no_of_constrained_dofs<amfe.boundary.DirichletBoundary.no_of_constrained_dofs>`     | Contains the number of free dofs of the constrained system                                                             |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.no_of_unconstrained_dofs<amfe.boundary.DirichletBoundary.no_of_unconstrained_dofs>` | Contains the number of dofs of the unconstrained system (all dofs when no constraints were imposed)                    |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+

Dirichlet boundary conditions are set if all these properties are set properly.
Usually this is done in two steps:

1. Set the properties of the mesh class (:py:attr:`Mesh.nodes_dirichlet<amfe.mesh.Mesh.nodes_dirichlet>`, :py:attr:`Mesh.dofs_dirichlet<amfe.mesh.Mesh.dofs_dirichlet>`)
2. Create a DirichletBoundary object and assign its properties (:py:attr:`DirichletBoundary.slave_dofs<amfe.boundary.DirichletBoundary.slave_dofs>`,
    :py:attr:`DirichletBoundary.B<amfe.boundary.DirichletBoundary.B>`, :py:attr:`DirichletBoundary.no_of_constrained_dofs<amfe.boundary.DirichletBoundary.no_of_constrained_dofs>`,
    :py:attr:`DirichletBoundary.no_of_unconstrained_dofs<amfe.boundary.DirichletBoundary.no_of_unconstrained_dofs>`)

Example-The hard way
^^^^^^^^^^^^^^^^^^^^

In this example we show the deepest way to assign Dirichlet boundary conditions in AMfe.
It is shown that setting the all the properties shown in the table above is sufficient to apply Dirichlet boundary
conditions. Consider the following example:

.. _simple_geo_dbc:
.. figure:: ../../static/img/simple_geo.svg

  Simple mesh-geometry

:numref:`simple_geo_dbc` shows a simple mesh-geometry with two elements and 6 nodes.

We want to fix node 0 in x- and y-direction and node 3 in x-direction. The first step is to set the properties of the
Mesh-class::

    >>> msh = amfe.Mesh()
    >>> ... # Several operations to define the mesh above...
    >>> dirichlet_nodes = np.array([0,3])
    >>> dirichlet_dofs = np.array([0,1,6])
    >>> msh.nodes_dirichlet = dirichlet_nodes
    >>> msh.dofs_dirichlet = dirichlet_dofs

The second step is to set the properties of the DirichletBoundary class::

    >>> dic = amfe.DirichletBoundary()
    >>> dic.no_of_unconstrained_dofs = msh.no_of_dofs
    >>> dic.no_of_constrained_dofs = msh.no_of_dofs - len(msh.dofs_dirichlet)
    >>> dic.slave_dofs = msh.dofs_dirichlet
    >>> dic.B = [



Another way::

    >>> msh = amfe.Mesh()
    >>> ... # Several operations to define the mesh topology...
    >>> nodes, dofs = msh.set_dirichlet_bc(2,output='external')
    >>> dic.constrain_dofs(dofs)




Convenient way
^^^^^^^^^^^^^^

The same steps can be done in a more convenient way if a mesh-property (a physical group is avilable).
The first step can be done by the method :py:meth:`set_dirichlet_bc<amfe.mesh.Mesh.set_dirichlet_bc>`::

    >>> msh = amfe.Mesh()
    >>> ... # Several operations to define the mesh above...
    >>> msh.set_dirichlet_bc(1,'xy')
    >>> msh.set_dirichlet_bc(2,'x')

The second step (Constrain the dofs in the Dirichlet-Class) can be done by the method :py:meth:`constrain_dofs<amfe.boundary.DirichletBoundary>`::

    >>> dic = amfe.DirichletBoundary()
    >>> dic.no_of_unconstrained_dofs = msh.no_of_dofs
    >>> dic.constrain_dofs(msh.constrained_dofs dirichlet_dofs)
  HIER NOCHMAL PRÜFEN!


Most convenient way
^^^^^^^^^^^^^^^^^^^



Simple Constraints:
^^^^^^^^^^^^^^^^^^^

Example::

    >>> msh = amfe.Mesh()
    >>> ... # Several operations to define the mesh topology...
    >>> nodes, dofs = msh.set_dirichlet_bc(2,output='external')
    >>> dirichlet_class.apply_master_slave_list([[dofs[0], dofs[1:], None],])


DirichletBoundary-Class
-----------------------

The DirichletBoundary-Class helps to handle Dirichlet Boundary conditions.
