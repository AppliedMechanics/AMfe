Mesh
====

The mesh-module has the following task:

- Provide data structures to describe all mesh properties e.g. nodes and elements
- Provide functions for import and export meshes
- Some Helper functions for Boundary Conditions and other



Fundamentals of the Mesh class
------------------------------


Defining a Mesh
^^^^^^^^^^^^^^^


The :py:class:`Mesh-Class <amfe.mesh.Mesh>` handles many things concerning the mesh.

One can instanciate a mesh object via::

    >>> import amfe
    >>> my_mesh = amfe.Mesh()

The mesh class has some basic important properties which describe the mesh-topology.
These properties are:

+----------------------+----------------------------------------------------------------------------------------+
| nodes                | numpy.ndarray which contains the coordinates of all nodes in reference configuration   |
+----------------------+----------------------------------------------------------------------------------------+
| connectivity         | array that maps the nodes to elements                                                  |
+----------------------+----------------------------------------------------------------------------------------+
| ele_obj              | list that maps element-objects to all elements                                         |
+----------------------+----------------------------------------------------------------------------------------+
| neumann_connectivity | array that maps the nodes to Neumann-elements (needed for Neumann boundary conditions) |
+----------------------+----------------------------------------------------------------------------------------+
| neumann_obj          | list that maps Neumann-element-objects to all elements                                 |
+----------------------+----------------------------------------------------------------------------------------+



Consider the following example:

.. _simple_geo:
.. figure:: ../../static/img/simple_geo.svg

  Simple mesh-geometry

:numref:`simple_geo` shows a simple mesh-geometry with two elements and 6 nodes.

.. note::

  The indices for the node-ids start with zero


To configure this mesh we first need to save the node-coordinates.
As it is a 2d-Problem the z-coordinate can be dropped::

    >>> import numpy as np
    >>> my_mesh.nodes = np.array([[0., 0.],
                                  [2., 0.],
                                  [4., 0.],
                                  [0., 1.5],
                                  [2., 1.5],
                                  [2., 1.5]])
    >>> my_mesh.no_of_dofs_per_node = 2


Then we have to tell AMfe how the elements are configured. There are two Quad4
elements that are numbered in circular in mathematical positive direction::

    >>> my_mesh.connectivity = [np.array([0, 1, 4, 3], dtype='int'), np.array([1, 2, 5, 4], dtype='int')]


Now element-topology is set. Next we need to apply element-data with material information to each element.
This is done by asserting pointers to objects of the :class:`Element <amfe.element.Element>` class.
The easiest way for this assertion is::

    >>> import copy
    >>> my_material = amfe.KirchhoffMaterial()
    >>> quad_class = copy.deepcopy(my_mesh.element_class_dict['Quad4'])
    >>> quad_class.material = my_material
    >>> object_series = [quad_class, quad_class]
    >>> my_mesh.ele_obj.extend(object_series)


At last we need to update some Mesh-properties, such as no_of_dofs-property via running

    >>> my_mesh._update_mesh_props()

Now all necessary properties of the Mesh class are set. Assembly classes can now
handle the assembly of that mesh.



Other properties and limits of the mesh class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beside the main properties nodes, connectivity, ele_obj, neumann_connectivity and neumann_obj
the mesh has some other properties:

The first type of properties are **numbers** of different properties of the mesh which are listed in table :numref:`tab_mesh_no_properties`.
A very important property of this type is :py:attr:`no_of_dofs_per_node<amfe.mesh.Mesh.no_of_dofs_per_node>`.
This property has to be set manually if the mesh has not been imported by import-functions such as :py:meth:`import_msh<amfe.mesh.Mesh.import_msh>`.
Currently there are two possible values for this method. You can set it as 2 for 2d problems or as 3 for 3d problems.

.. note::
    This property is a very central variable in AMfe and has large effects e.g. to the global numbering of dofs.
    Do not change this property unless you know what you are doing.
    **Currently the mesh class is limited to describe meshes with elements that have no rotational degrees of freedom.**


.. _tab_mesh_no_properties:

.. table:: Additional "number of"-properties
    
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | Property                             | Description                                                                                                            |
    +======================================+========================================================================================================================+
    | no_of_dofs_per_node: int             | 2 for 2D problems, 3 for 3D problems                                                                                   |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | no_of_elements: int                  | Number of elements in the mesh associated with an alement object                                                       |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | no_of_nodes: int                     | Number of nodes of the whole system                                                                                    |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | no_of_dofs: int                      | Number of dofs in the system including constrained dofs                                                                |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+

  

Another type of additional properties of the Mesh-class is dictionaries of available element types (s. :numref:`tab_mesh_dict_properties`).
There is one dictionary for simple elements such as Hex or Quad elements and another dictionary
for boundary elements which are used to apply neumann boundary conditions.
The element_class_dict should not be changed by users.
The property is only used in method :py:meth:`load_group_to_mesh<amfe.mesh.Mesh.load_group_to_mesh>`
to generate pointers to element objects with assigned materials.


.. _tab_mesh_dict_properties:

.. table:: Additional properties with dictionaries for elements

    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | Property                             | Description                                                                                                            |
    +======================================+========================================================================================================================+
    | element_class_dict: dict             | Dictionary containing objects of elements which can be copied to instantiate Element objects with material information |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | element_boundary_class_dict: dict    | Same as above but for Boundary-elements                                                                                |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+



The last type of properties is properties concerning Dirichlet boundary conditions (s. :numref:`tab_mesh_dirichlet_properties`).
These properties are set by the method :py:meth:`set_dirichlet_bc<amfe.mesh.Mesh.set_dirichlet_bc>` which is explained in
section :doc:`boundary_conditions`.

.. _tab_mesh_dirichlet_properties:

.. table:: Additional properties concerning Dirichlet boundary conditions
    
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | Property                             | Description                                                                                                            |
    +======================================+========================================================================================================================+
    | nodes_dirichlet (ndarray)            | node ids of nodes with Dirichlet boundary condition                                                                    |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+
    | dofs dirichlet                       | contains the dofs constrained by a Dirichlet boundary condition                                                        |
    +--------------------------------------+------------------------------------------------------------------------------------------------------------------------+




Importing Meshes
----------------

Basic proceeding
^^^^^^^^^^^^^^^^

If your mesh data is available as datafiles in gmsh, Nastran, Ansys or Abaqus-format
you can use one of the very convenient import-functions.

There are two ways to import mesh-data. The first way is to do the import in
two steps:

1. Import the node and element data via an import function described below.

2. Create instances of materials and assign them
   to a so called 'mesh_property' by using the method
   :py:meth:`load_group_to_mesh<amfe.mesh.Mesh.load_group_to_mesh>`.

The first step creates a pandas-Dataframe which contents mesh data of the
imported file. This can also contain so called physical groups which represent
domains of the mesh that have the same properties (e.g. same material).
The Dataframe is stored in the Mesh-class-property *el_df*.

You can print some information about physical_groups by using the
:py:meth:`mesh_information()<amfe.mesh.Mesh.mesh_information()>` method.
This can help you assigning materials to the right phyiscal groups in second
step.

In the second step you have to assign materials to the mesh. This can be done
by using the :py:meth:`load_group_to_mesh<amfe.mesh.Mesh.load_group_to_mesh>`
method. It expects two arguments: *key* and *material*.
The first argument is an ID for the mesh-property you want to assign the
material to. The second is an :py:class:`Material<amfe.material.Material>`
object.

During call of this function the connectivity list of the elements is generated
and stored in the properties `connectivity` and `neumann_connectivity` (for
Neumann elements). Furthermore the method 
creates one Element object for each element type the material has been
assigned to.


Another way - the second way - is to directly import both node/element data and
property/material data in one step. Currently this can be done via a
MechanicalSystem method.

.. note::
    
    The second method is only implemented for meshes with only one material
    in the whole domain.
    
    
Mesh Deflation
^^^^^^^^^^^^^^

If the mesh is imported by an import function, it can happen that there
are imported nodes that are not connected to any element.
If you want to clean the mesh i.e. remove those nodes, run::

    >>> my_mesh.deflate_mesh()
    
This method removes the nodes from the connectivity_lists and from the
el_df Property.


Import-Functions for certain data-formats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gmsh
""""

For example the mesh
above is available as gmsh-file::

    $MeshFormat
    2.2 0 8
    $EndMeshFormat
    $Nodes
    6
    1 0 0 0
    2 2 0 0
    3 4 0 0
    4 0 1.5 0
    5 2 1.5 0
    6 4 1.5 0
    $EndNodes
    $Elements
    2
    1 3 2 1 1 1 2 5 4
    2 3 2 1 1 2 3 6 5
    $EndElements

Then you can easily import the data via::

    >>> filename = '/home/user/path/to/file.msh'
    >>> my_mesh.import_msh(filename)

Then you can assign materials to physical_groups that are defined in the gmsh file
(see gmsh-documentation for more information about physical_groups).
In this example the physical_group with id=1 is assigned by my_material::

    >>> my_material = amfe.KirchhoffMaterial()
    >>> my_mesh.load_group_to_mesh(1,my_material)

Nastran
"""""""

.. warning::
    
    The Nastran import is in development and does only work for a small
    subset of possible Nastran-meshes.


The same procedure can be done for Nastran-Meshes::

  >>> my_mesh.import_bdf(filename)
  
But as stated above, this only works for a small subset of possible nastran
meshes.

Abaqus
""""""

Ansys meshes can be imported by::

  >>> my_mesh.import_inp(filename)


CSV
"""

.. warning::
    
    The CSV import is deprecated/experimental and does not work properly.
    
    
There is also the possibility to import mesh data from a csv-Datafile::

    >>> my_mesh.import_csv(filename_nodes, filename_elements, explicit_node_numbering, ele_type)
    
Here two filenames must be passed to the function. The first csv-file contains
the nodes, the second contains the elements. The flag *explicit_node_numbering*
can be set to true if the first column of the nodes-file contains IDs for
the nodes (default is false). The *ele_type* parameter can be used to specifiy the order of
elements. The parameter must be the name of the element passed as a string,
e.g. 'Tri6' if you want second order elements for a triangular mesh.


Mesh Tying
----------

.. todo::
    Explain Mesh Tying


Helper-Functions
----------------


.. todo::
    Helper Functions:
    
    * create_xdmf_form_hdf5 (besser in Postprocessing/Paraview stecken)
    * check_dir
    * prettify_xml
    * shape2str
    * h5_set_attributes
    * variable: element_mapping_list, conversion_dict