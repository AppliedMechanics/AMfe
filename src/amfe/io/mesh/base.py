# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#


import logging
from abc import ABC, abstractmethod


__all__ = ['MeshReader',
           'MeshConverter']


class MeshReader(ABC):
    """
    Abstract super class for all mesh readers.

    TASKS
    -----
    - Read line by line a stream (or file).
    - Call mesh converter functions for each line.

    NOTES
    -----
    PLEASE FOLLOW THE BUILDER PATTERN!
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        return

    @abstractmethod
    def parse(self, builder):
        pass


class MeshConverter:
    """
    Super class for all mesh converters.
    """

    def __init__(self, *args, **kwargs):
        return

    def build_no_of_nodes(self, no):
        """
        Build number of nodes (optional)

        This function usually is optional. It can be used to enhance performance
        of the building process. This function can be used to preallocate arrays
        that contain the node coordinates

        Parameters
        ----------
        no : int
            number of nodes in the mesh

        Returns
        -------
        None
        """
        pass

    def build_no_of_elements(self, no):
        """
        Build number of elements (optional)

        This function usually is optional. It can be used to enhance performance
        of the building process. This function can be used to preallocate arrays
        that contain the element information

        Parameters
        ----------
        no : int
            number of elements in the mesh

        Returns
        -------
        None
        """
        pass

    def build_node(self, idx, x, y, z):
        """
        Builds a node

        Parameters
        ----------
        idx : int
            ID of the node
        x : float
            X coordinate of the node
        y : float
            Y coordinate of the node
        z : float
            Z coordinate of the node

        Returns
        -------
        None
        """
        pass

    def build_element(self, idx, etype, nodes):
        """
        Builds an  element

        Parameters
        ----------
        idx : int
            ID of an element
        etype : str
            valid amfe elementtype (shape) string
        nodes : iterable
            iterable of ints describing the connectivity of the element

        Returns
        -------
        None
        """
        pass

    def build_group(self, name, nodeids, elementids):
        """
        Builds a group, i.e. a collection of nodes and elements

        Parameters
        ----------
        name: str
            Name identifying the node group.
        nodeids: list
            List with node ids.
        elementids: list
            List with element ids.

        Returns
        -------
        None
        """
        pass

    def build_mesh_dimension(self, dim):
        """
        Builds the dimensino of the mesh (optional)
        If this method has not been called during build process, a mesh dimension
        of 3 is assumed

        Parameters
        ----------
        dim : int {2, 3}
            dimension of the mesh

        Returns
        -------
        None
        """
        pass

    def build_tag(self, tag_name, values2elements, dtype=None, default=None):
        """
        Builds a tag with following dict given in tag_dict

        Parameters
        ----------
        tag_name: str
            tag name
        values2elements: dict
            dict with following format:
            { tagvalue1 : [elementids],
              tagvalue2 : [elementids],
                           ...
            }

        dtype: { int, float, object }
            data type for this tag. Only int, float or object is allowed

        default:
            default value for elementids with no tagvalue

        Returns
        -------
        None
        """
        pass

    def return_mesh(self):
        """
        Returns the Mesh or the file pointer or 0

        Returns
        -------
        Object
        """
        return None
