# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

"""
Mesh module of AMfe.

This module provides a mesh class that handles the mesh information: nodes, mesh topology, element shapes, groups, ids.
"""


import numpy as np
import pandas as pd

__all__ = [
    'Mesh'
]

# Describe Element shapes, that can be used in AMfe
# 2D volume elements:
element_2d_set = {'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8', }
# 3D volume elements:
element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}
# 2D boundary elements
boundary_2d_set = {'straight_line', 'quadratic_line'}
# 3D boundary elements
boundary_3d_set = {'straight_line', 'quadratic_line', 'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}


class Mesh:
    """
    Class for handling the mesh operations.

    Attributes
    ----------
    nodes : ndarray
        Array of x-y-z coordinates of the nodes in reference configuration. Dimension is
        (no_of_nodes, 2) for 2D problems and (no_of_nodes, 3) for 3D problems.
        z-direction is dropped for 2D problems!
    nodeid2idx : dict
        Dictionary with key = node-id: value = row id in self.nodes array for getting nodes coordinates X
    connectivity : list
        List of node-rowindices of self.nodes belonging to one element.
    el_df : pandas.DataFrame
        DataFrame with element information
    groups : list
        List of groups containing ids (not row indices!)

    Notes
    -----
    GETTER CLASSES NAMING CONVENTION
    We use the following naming convention for function names:
      get_<node|element><ids|idxs>_by_<groups|ids|idxs>
               |            |     |        |
               |            |     |        - Describe which entity is passed groups, ids or row indices
               |            |     - 'by' keyword
               |            - describes weather ids or row indices are returned
                - describes weather nodes or elements are returned

    """
    def __init__(self, dimension=3):
        """
        Parameters
        ----------
        dimension : int
            describes the dimension of the mesh (2 or 3)

        Returns
        -------
        None
        """
        # -- GENERAL INFORMATION --
        self._dimension = dimension

        # -- NODE INFORMATION --
        # node coordinates as np.array
        self.nodes = np.empty((0, dimension), dtype=float)
        # map from nodeid to idx in nodes array
        self.nodeid2idx = dict([])

        # -- ELEMENT INFORMATION --
        # connectivity for volume elements and list of shape information of each element
        # list of elements containing rowidx of nodes array of connected nodes in each element
        self.connectivity = list()

        # the same for boundary elements
        self.boundary_connectivity = list()

        # Pandas dataframe for elements:
        self.el_df = pd.DataFrame(columns=('shape', 'is_boundary', 'connectivity_idx'))

        # group dict with names mapping to element ids or node ids, respectively
        self.groups = dict()

    @property
    def no_of_nodes(self):
        """
        Returns the number of nodes

        Returns
        -------
        no_of_nodes: int
            Number of nodes of the whole mesh.
        """
        return self.nodes.shape[0]

    @property
    def no_of_elements(self):
        """
        Returns the number of volume elements

        Returns
        -------
        no_of_elements : int
            Number of volume elements in the mesh
        """
        return len(self.el_df[self.el_df['is_boundary'] != True].index)

    @property
    def no_of_boundary_elements(self):
        """
        Returns the number of boundary elements

        Returns
        -------
        no_of_elements : int
            Number of boundary elements in the mesh
        """
        return len(self.el_df[self.el_df['is_boundary'] == True].index)

    @property
    def dimension(self):
        """
        Returns the dimension of the mesh

        Returns
        -------
        dimension : int
            Dimension of the mesh
        """
        return self._dimension

    @dimension.setter
    def dimension(self, dim):
        """
        Sets the dimension of the mesh

        Attention: The dimension should not be modified except you know what you are doing.

        Parameters
        ----------
        dim : int
            Dimension of the mesh

        Returns
        -------
        None

        """
        self._dimension = dim

    @property
    def nodes_voigt(self):
        """
        Returns the nodes in voigt notation

        Returns
        -------
        nodes_voigt : ndarray
            Returns the nodes in voigt-notation
        """
        return self.nodes.reshape(-1)

    def get_elementidxs_by_groups(self, groups):
        """
        Returns elementindices of the ele_shape/boundary_shape property belonging to groups

        Parameters
        ----------
        groups : list
            groupnames as strings in a list

        Returns
        -------
            indices of the elements in the connectivity array
        """
        elementids = list()
        for group in groups:
            elementids.extend(self.groups[group]['elements'])
        return self.el_df.loc[elementids, 'connectivity_idx'].values

    def get_nodeidxs_by_groups(self, groups):
        """
        Returns nodeindieces of the nodes property belonging to a group

        Parameters
        ----------
        groups : list
            contains the groupnames as strings

        Returns
        -------
        nodeidxs : ndarray

        """
        nodeids = []
        elementids = []
        for group in groups:
            if self.groups[group]['elements'] is not None:
                elementids.extend(self.groups[group]['elements'])
            if self.groups[group]['nodes'] is not None:
                nodeids.extend(self.groups[group]['nodes'])
        nodeidxs = np.array([self.nodeid2idx[nodeid] for nodeid in nodeids], dtype=int)
        nodes = np.hstack((self.connectivity[idx] for idx in self.el_df.loc[elementids, 'connectivity_idx'].values))
        nodes = np.unique(nodes)
        nodes = np.unique(np.hstack((nodes, np.array(nodeidxs))))
        return nodes

    def get_ele_shapes_by_idxs(self, elementidxes):
        """
        Returns list of element_shapes for elementidxes

        Parameters
        ----------
        elementidxes : list
            list of tuples (0/1, idx), where 0 = volume element, 1 = boundary_element

        Returns
        -------
        ele_shapes : list
            list of element_shapes as string
        """
        return [self.el_df.loc[self.el_df.connectivity_idx == idx, 'shape'].values[0] for idx in elementidxes]

    def get_nodeidxs_by_all(self):
        """
        Returns all nodeidxs

        Returns
        -------
        nodeidxs : ndarray
            returns all nodeidxs
        """
        return np.arange(self.no_of_nodes, dtype=np.int)

    def get_nodeids_by_nodeidxs(self, nodeidxs):
        """

        Parameters
        ----------
        nodeidxs : list
            rowindices of node array

        Returns
        -------
        id : list
            IDs of the corresponding nodes
        """
        keys = list(self.nodeid2idx.keys())
        values = list(self.nodeid2idx.values())
        return [keys[values.index(idx)] for idx in nodeidxs]
