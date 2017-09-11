Export and Postprocessing
=========================

Most convenient export method
-----------------------------

The most convenient way to export data to use it in a postprecessor like paraview, is to call the
:py:meth:`MechanicalSystem.export_paraview(filename, field_list)<amfe.mechanical_system.MechanicalSystem.export_paraview>`
method. This method exports data to hdf5 fileformat and xdmf for reading the hdf5 in postprocessors such as Paraview.

If field_list is empty, the method exports T_output, u_output, S_output, E_output and the B matrix of the Dirichlet class.
These entities will be always exported.
The field_list parameter can be used to pass further datasets that shall be exported to paraview.
This parameter is a list that contains tuples with the data variable to export and a dictionary that describes the data
variable for Paraview.

Table :numref:`tab_paraview_field_list` lists the keys of the dictionary:

.. _tab_paraview_field_list:

.. table:: Paraview Field List Keys

    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | Key                                                                                                             | Description                                                                                       |
    +=================================================================================================================+===================================================================================================+
    | Paraview                                                                                                        | True or False, Flag to control if data is either only exported to hdf5 or also to xdmf            |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | Name                                                                                                            | Name to identify data                                                                             |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | AttributeType                                                                                                   | Describes the type of Data for Paraview (e.g. 'Tensor6'), see Paraview Documentation for details  |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | Center                                                                                                          | Describes if it is a node or element entitiy (e.g. 'Node')                                        |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+
    | NoOfComponents                                                                                                  | Integer that describes the number of components of the vector                                     |
    +-----------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------+

Example::

    >>> filename = '/home/user/path/to/file_without_extension'
    >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
    >>> my_mechanical_system.export_paraview(filename, field_list)


Xdmf-Export in detail
---------------------

The
:py:meth:`MechanicalSystem.export_paraview(filename, field_list)<amfe.mechanical_system.MechanicalSystem.export_paraview>`
performs some standard steps such as defining some default values for export and then calls the method
:py:meth:`Mesh.save_mesh_xdmf()<amfe.mesh.Mesh.save_mesh_xdmf>`.
This method belongs to the :py:class:`Mesh<amfe.mesh.Mesh>`-class and actually performs the export.
The export can be done by calling::

    >>> field_list = [(...)] # Defining the list of tuples for data description
    >>> my_mesh.save_mesh_xdmf(filename, field_list, bmat, u, timesteps)

Hereby bmat is the B-matrix from the dirichlet-class, u are the full displacements and timesteps are the timesteps.
The filename-parameter gives the filename the xdmf-file is saved as.
The parameters u and timesteps can be passed if one wants to save the displacements of the geometry for different
timesteps. Hereby u is passed as a numpy array.
The parameter field_list can be used to store additional fields in hdf5. The parameter is a list of tuples that contain
two items. First the data-array and second a dictionary that contains metadata.
If the key **Cell** is set to **Center** then this dataset is seen as belonging to elements.
If the key **ParaView** is set to **True** the dataset is both imported to hdf5 and to paraview xdmf.
The other attributes are passed to the function :py:func:`create_xdmf_from_hdf5<amfe.mesh.create_xdmf_from_hdf5>` . See documentation of xdmf and Paraview for
more details.

.. warning::

    This function cannot export meshes that have several different element types. It only can export meshes that have
    just one element type inside (homogeneous). If nonhomogeneous meshes are passed, the function exports the largest
    part of the mesh with one element type.

