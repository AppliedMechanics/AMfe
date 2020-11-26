# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#


__all__ = ['PostProcessorWriter',
           'PostProcessorReader']


class PostProcessorReader:
    def __init__(self):
        """
        Constructor for PostProcessorReader Base Class

        """
        return

    def parse(self, builder):
        """
        Parse the given postprocessor data

        Parameters
        ----------
        builder : PostProcessorWriter
            PostProcessorWriter object that is used for building the desired datastructure

        Returns
        -------
        None
        """
        pass


class PostProcessorWriter:
    def __init__(self, meshreaderobj):
        """
        Constructor for PostProcessorWriter Base Class

        Parameters
        ----------
        meshreaderobj : amfe.io.mesh.MeshReader
            Mesh Reader object that can parse the mesh belonging to the data that will be postprocessed

        Returns
        -------
        writer : PostProcessorWriter
        """
        self._meshreader = meshreaderobj

    # -------------- BUILDER METHODS --------------------------------------------------

    def write_field(self, name, field_type, t, data, index, mesh_entity_type):
        """
        Parameters
        ----------
        name : str
            Name for the field to write
        field_type : PostProcessDataType
            Data Type
        t : ndarray
            Timesteps
        data : ndarray
            ndarray to write
        index : ndarray
            ndarray with indices of the mesh entities in the mesh reader object
        mesh_entity_type : MeshEntityType
            type of mesh entities

        Returns
        -------
        None
        """
        return

    def return_result(self):
        """
        Returns the result of the build process. This method is called after the build process has been done.

        """
        return
