Mechanical System
=================

Most things you want to do can probably done with :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`-class.

.. _tab_mechanical_system_properties_classes:

.. table:: Properties of :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>` with pointers to AMfe-classes

    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Property                                                                                                        | Description                                                                                                                                                    |
    +=================================================================================================================+================================================================================================================================================================+
    | :py:attr:`mesh_class<amfe.mechanical_system.MechanicalSystem.mesh_class>`                                       | Contains an instance of a :py:class:`Mesh<amfe.mesh.Mesh>`-class that describes the mesh the mechanical system is associated with                              |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`assembly_class<amfe.mechanical_system.MechanicalSystem.mesh_class>`                                   | Contains an instance of an :py:class:`Assembly<amfe.assembly.Assembly>`-class that handles the assembly of the elements                                        |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:attr:`dirichlet_class<amfe.mechanical_system.MechanicalSystem.dirichlet_class>`                             | Contains an instance of a :py:class:`DirichletBoundary<amfe.boundary.DirichletBoundary>`-class that handles the dirichlet boundary conditions and constraints. |
    +-----------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+


Importing Meshes
----------------

The method
:py:meth:`load_mesh_from_gmsh(path_to_mesh_file, phys_group, material, scale_factor=1)<amfe.mechanical_system.MechanicalSystem.load_mesh_from_gmsh>`
loads a mesh from a gmsh file saved in path_to_mesh_file and assigns a :py:class:`Material<amfe.material.Material>`
object to all elements that belong to physical group phys_group.
By passing a scale_factor one can scale the whole mesh to larger or smaller size.

Example::

    >>> mechanical_system = amfe.MechanicalSystem()
    >>> my_material = amfe.KirchhoffMaterial()
    >>> mechanical_system.load_mesh_from_gmsh('home/user/path/to/mesh.msh',1,my_material)


.. todo::

    Show how to add other materials to elements


Applying boundary conditions
----------------------------


Dirichlet Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the method
:py:meth:`apply_dirichlet_boundaries(key, coord, mesh_prop='phys_group')<amfe.mechanical_system.MechanicalSystem.apply_dirichlet_boundaries`>
to apply Dirichlet boundary conditions (with displacement = 0) that will be assigned to all elements that belong
to one mesh property.
You can choose the mesh property that is used to assign a boundary condition by passing the optional mesh_prop
parameter.
The default value is phys_group. You have to choose a column name in the :py:attr:`el_df<amfe.mesh.Mesh.el_df>`
attribute. You can show which values are allowed by running
mechanical_system.mesh_class.mesh_information().

The key parameter chooses the id of the mesh_property where the boundary condition shall be assigned to.
The coord parameter defines to which global coordinates the boundary condition shall be assigned to.
Examples: 'x','xy','yz','xyz' etc.

Example: Apply a Dirichlet boundary condition to physical group no. 105 in x and z direction::

    >>> mechanical_system.apply_dirichlet_boundaries(105,'xz')


.. note::

    Currently only Dirichlet boundary conditions with displacement = 0 are implemented.


Neumann Boundary Conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the method
:py:meth:`apply_neumann_boundaries(key, val, direct, time_func=None, shadow_area=False, mesh_prop='phys_group')<amfe.mechanical_system.MechanicalSystem.apply_neumann_boundaries>`
to apply Neumann boundary conditions that shall be assigned to all elements that belong to one mesh property.
The key and mesh_prop parameters define the mesh property. mesh_prop defines the column name in the :py:attr:`el_df<amfe.mesh.Mesh.el_df>`
attribute of the mesh and key defines the id of the mesh_property where the boundary conditions shall be assigned to.

The parameter direct describes the direction of the applied force
Eihter you pass a tuple of coordinates or you pass the string 'normal'.
If one choosee a tuple of coordinates, e.g. (1,2,1), the force amplitudes will be scaled by 1 in the x and z direction and
scaled by 2 in the y direction. If one chooses direct='normal' the force will point in normal direction to the surface of
the current (spatial) configuration.

The parameter val is an overall scaling factor for the applied force.
One can pass a function that depends on a parameter t to define time dependent forces.
Example: Define a linear time function::

    >>> def time_func(t):
    >>>     return t

The shadow_area parameter is Boolean parameter. If this parameter is set to true, the applied force will be scaled
by the projected area of the boundary in the direction of the applied force. This only makes sense if a tuple of
coordinates are passed for the direct-parameter.
Example: Apply a linearly increased force with 1000 units after one second in x direction on physical group 105::

    >>> mechanical_system.apply_neumann_boundaries(105,1000,(1,0,0), lambda t: t)



Getter functions
----------------

There are many so called getter functions to get system vectors and matrices. These functions are listed in table
:numref:`tab_mechanical_system_getter_methods`. These functions expect the current displacement vector of the
constrained system and the current time.

.. _tab_mechanical_system_getter_methods:

.. table:: Getter methods of :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>` class.

    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | Method                                                                                                          | Description                                                                                       |
    +=================================================================================================================+===================================================================================================+
    | :py:meth:`f_int(u_constr, t)<amfe.mechanical_system.MechanicalSystem.f_int>`                                    | Returns the nonlinear internal restoring force vector                                             |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:meth:`M(u_constr, t)<amfe.mechanical_system.MechanicalSystem.M>`                                            | Returns the mass matrix                                                                           |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:meth:`f_ext(u_constr, du_constr, t)<amfe.mechanical_system.MechanicalSystem.f_ext>`                         | Returns the external force vector                                                                 |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:meth:`K(u_constr, t)<amfe.mechanical_system.MechanicalSystem.K>`                                            | Returns the tangential stiffness matrix                                                           |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:meth:`K_and_f(u_constr, t)<amfe.mechanical_system.MechanicalSystem.K_and_f>`                                | Returns both the tangential stiffness matrix and the nonlinear internal restoring force vector    |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:meth:`D(u_constr, t)<amfe.mechanical_system.MechanicalSystem.D>`                                            | Returns the damping matrix                                                                        |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+


Writing timesteps for later export
----------------------------------

The :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`-class provides properties to store simulation
results:

.. _tab_mechanical_system_output_props:

.. table:: Output properties

    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | Property                                                                                                        | Description                                                                                       |
    +=================================================================================================================+===================================================================================================+
    | :py:attr:`T_output<amfe.mechanical_system.MechanicalSystem.T_output>`                                           | Stores timesteps                                                                                  |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:attr:`u_output<amfe.mechanical_system.MechanicalSystem.u_output>`                                           | Stores displacements for each timestep (full displacement vector including constrained dofs)      |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:attr:`S_output<amfe.mechanical_system.MechanicalSystem.S_output>`                                           | Stores stress for each timestep                                                                   |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:attr:`E_output<amfe.mechanical_system.MechanicalSystem.E_output>`                                           | Stores strain for each timestep                                                                   |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | :py:attr:`iteration_info<amfe.mechanical_system.MechanicalSystem.iteration_info>`                               | Stores iteration infos of solvers if activated (time, number of iterations, residual)             |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+


To write a timestep run the
:py:meth:`write_timestep(t,u_constr)<amfe.mechanical_system.MechanicalSystem.write_timestep>` method.
The method expects a float number t which is the current time for which a displacement vector u shall be stored.
The passed displacement vector is the vector of the constrained system. The displacement vector that is stored is the
vector of the full mesh.
The stresses and strains are stored automatically if the property stress_recovery is set to true.

.. note::

    The stresses and strains which will be stored are the results of the last performed assembly.


All stored values can be dropped by running the
:py:meth:`clear_timesteps()<amfe.mechanical_system.MechanicalSystem.clear_timesteps>` method.


Time integrators
----------------

The :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`-class provides methods that return the linear
equation of time integration schemes.

.. note::

    If you want to do a time integration, use the time integration functions of the :py:mod:`solver<amfe.solver>` module.


Currently two methods are implemented:

1. :py:meth:`S_and_res(u_constr, du_constr, ddu_constr, dt, t, beta, gamma)<amfe.mechanical_system.MechanicalSystem.S_and_res>`
   Returns S and res for solver to solve :math:`S \Delta q = -res` of the Newmark time integration scheme.
   One has to pass the current displacements, velocities and accelerations of the constrained system, the timestep dt,
   the current time t and the Newmark parameters beta and gamma.

2. :py:meth:`gen_alpha(q, dq, ddq, q_old, dq_old, ddq_old, f_ext_old, dt, t, alpha_m, alpha_f, beta, gamma)<amfe.mechanical_system.MechanicalSystem.gen_alpha>`
   Returns S, res and f_ext for solver to solve :math:`S \Delta q = -res` of the Generalized Alpha time integration scheme.


Rayleigh Damping
----------------

The :py:class:`MechanicalSystem<amfe.mechanical_system.MechanicalSystem>`-class provides a method to add Rayleigh
Damping to the system.
The method :py:meth:`apply_rayleigh_damping(alpha, beta)<amfe.mechanical_system.MechanicalSystem.apply_rayleigh_damping>`
calculates the damping matrix of the system by using the relation

.. math:: \textbf{D} = \alpha \textbf{M} + \beta \textbf{K}


The damping matrix is stored in the :py:attr:`D_constr<amfe.mechanical_system.MechanicalSystem.D_constr>` property.


Helper functions
----------------

If you want to clean up the mesh from floating nodes that are not connected to any element, run the
:py:meth:`deflate_mesh()<amfe.mechanical_system.MechanicalSystem.deflate_mesh>` method.


.. todo::

    tie_mesh() not documented yet.