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
    | :py:attr:`DirichletBoundary.slave_dofs<amfe.boundary.DirichletBoundary.slave_dofs>`                             | Contains the constrained dofs?                                                                                         |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.B<amfe.boundary.DirichletBoundary.B>`                                               | A mapping matrix between the free dofs of the unconstrained and the constrained system                                 |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.no_of_constrained_dofs<amfe.boundary.DirichletBoundary.no_of_constrained_dofs>`     | Contains the number of free dofs of the constrained system                                                             |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`DirichletBoundary.no_of_unconstrained_dofs<amfe.boundary.DirichletBoundary.no_of_unconstrained_dofs>` | Contains the number of dofs of the unconstrained system (all dofs when no constraints were imposed)                    |
    +-----------------------------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+


