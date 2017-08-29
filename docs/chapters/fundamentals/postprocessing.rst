Export and Postprocessing
=========================

Export Meshes and displacements over time
-----------------------------------------

To export meshes there is one method that exports the mesh to paraview-xdmf-format: :py:meth:`Mesh.save_mesh_xdmf()<amfe.mesh.Mesh.save_mesh_xdmf>`.
This method belongs to the :py:class:`Mesh<amfe.mesh.Mesh>`-class.
The export can be done by calling::

    >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
    >>> my_mesh.save_mesh_xdmf(filename, field_list, bmat, u, timesteps)

The filename-parameter gives the filename the xdmf-file is saved as.
The parameters u and timesteps can be passed if one wants to save the displacements of the geometry for different
timesteps. Hereby u is passed as a numpay array.
The parameter field_list can be used to store additional fields in hdf5. The parameter is a list of tuples that contain
two items. First the data-array and second a dictionary that contains metadata.
If the key **Cell** is set to **Center** then this dataset is seen as belonging to elements.
If the key **ParaView** is set to **True** the dataset is both imported to hdf5 and to paraview xdmf.
The other attributes are passed to the function :py:func:`create_xdmf_from_hdf5<amfe.mesh.create_xdmf_from_hdf5>` . See documentation of xdmf and paraview for
more details.
