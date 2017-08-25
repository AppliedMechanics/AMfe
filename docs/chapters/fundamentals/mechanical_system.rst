Mechanical System
=================

Most things you want to do can probably done with :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`-class.

.. _tab_mechanical_system_properties_classes:

.. table:: Properties of :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`with pointers to AMfe-classes

    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Property                                                                                                        | Description                                                                                                                                                    |
    +=================================================================================================================+================================================================================================================================================================+
    | :py:attr:`mesh_class<amfe.mechanical_system.MechanicalSystem.mesh_class>`                                       | Contains an instance of a :py:class:`Mesh<amfe.mesh.Mesh>`-class that describes the mesh the mechanical system is associated with                              |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`assembly_class<amfe.mechanical_system.MechanicalSystem.mesh_class>`                                   | Contains an instance of an :py:class:`Assembly<amfe.assembly.Assembly>`-class that handles the assembly of the elements                                        |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`dirichlet_class<amfe.mechanical_system.MechanicalSystem.dirichlet_class>`                             | Contains an instance of a :py:class:`DirichletBoundary<amfe.boundary.DirichletBoundary>`-class that handles the dirichlet boundary conditions and constraints. |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
