#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
AMfe mesh converter for I/O module.
"""

import numpy as np
import pandas as pd

from amfe.io.mesh.base import MeshConverter
from amfe.io.mesh.constants import VOLUME_ELEMENTS_3D

__all__ = [
    'AmfePostprocessMeshConverter'
    ]


class AmfePostprocessMeshConverter(MeshConverter):
    """
    Converter for Postprocessing Meshes in Postprocessors

    This Converter is usually used by Postprocessorconverters to read the mesh.
    It is usually instatiated automatically within the Postprocessorconverters.

    Attributes
    ----------
    _currentnodeid : int
        Integer describing the current local node id. This is needed to build the node array.
    _dimension : int
        desribes the dimension of the mesh
    _el_df_indices : list
        list of elementids
    _el_df_eleshapes : list
        list of strings describing the shapes of the elements
    _el_df_connectivity : list
        list of ndarrays desribing the connectivity of the elements (regarding real elementids)
    _groups : dict
        dict with groups
    _nodes : ndarray
        ndarray that contains the node coordinates (rows: nodes, columns: x,y,z-coordinates)
    _no_of_nodes : int
        number of nodes that will be returned
    _no_of_elements : int
        number of elements that will be returned
    """
    def __init__(self, verbose=False):
        super().__init__()
        self._verbose = verbose
        self._dimension = None
        self._no_of_nodes = None
        self._no_of_elements = None
        self._nodes = np.empty((0, 4), dtype=float)
        self._currentnodeid = 0
        self._groups = dict()
        self._tags = dict()
        # df information
        self._el_df_indices = list()
        self._el_df_eleshapes = list()
        self._el_df_connectivity = list()
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
        # This function is only used for preallocation
        # It is not necessary to call, but useful if information about no_of_nodes exists
        self._no_of_nodes = no
        if self._nodes.shape[0] == 0:
            self._nodes = np.zeros((no, 4), dtype=float)
        return

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
        # This function is not used
        # If someone wants to improve performance he/she can add preallocation functionality for elements
        self._no_of_elements = no
        # preallocation...
        return

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
        self._dimension = dim
        return

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
        # amfeid is the row-index in nodes array
        amfeid = self._currentnodeid
        # Check if preallocation has been done so far
        if self._no_of_nodes is not None:
            # write node in preallocated array
            self._nodes[amfeid, :] = [idx, x, y, z]
        else:
            # append node if array is not preallocated with full node dimension
            self._nodes = np.append(self._nodes, np.array([idx, x, y, z], dtype=float, ndmin=2), axis=0)
        self._currentnodeid += 1
        return

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
        # update df information
        self._el_df_connectivity.append(np.array(nodes, dtype=int))
        self._el_df_indices.append(idx)
        self._el_df_eleshapes.append(etype)
        return

    def build_group(self, name, nodeids=None, elementids=None):
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
        # append group information
        group = {name: {'nodes': nodeids, 'elements': elementids}}
        self._groups.update(group)
        return

    def build_tag(self, tag_dict):
        """
        Builds a tag with following dict given in tag_dict

        Parameters
        ----------
        tag_dict : dict
            dict with following format:
            { tagname1 : { tagvalue1 : [elementids],
                           tagvalue2 : [elementids],
                           ...
                         },
              tagname2 : { tagvalue1 : [elementids],
                           tagvalue2 : [elementids]
                           ...
                         },
              ...
            }

        Returns
        -------
        None
        """
        # append tag information
        self._tags.update(tag_dict)
        return None

    def return_mesh(self):
        """
        Returns the Mesh as dict container

        This function must be called after the building proccess is done.

        Returns
        -------
        meshcontainer : dict
            Meshcontainer described with a dict with following keys:
            'nodes': nodes_df, (nodes dataframe)
            'elements': el_df, (elements dataframe)
            'groups': self._groups, (groups)
            'dimension': self._dimension (dimension)
            The tags are included in the elements dataframe
        """
        # Check dimension of model
        if self._dimension is None:
            if not VOLUME_ELEMENTS_3D.intersection(set(self._el_df_eleshapes)):
                # No 3D element in eleshapes, thus:
                self._dimension = 2
            else:
                self._dimension = 3
        # If dimension = 2 cut the z coordinate
        x = self._nodes[:, 1]
        y = self._nodes[:, 2]
        no_of_nodes = self._nodes.shape[0]
        if self._dimension == 2:
            z = np.zeros(no_of_nodes)
        else:
            z = self._nodes[:, 3]
        iloc = np.arange(no_of_nodes)
        nodes_df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'iloc': iloc}, index=np.array(self._nodes[:, 0], dtype=int))

        # write properties
        # The iconnectivity is the row based connectivity (pointing to row indices in a nodes array insted of nodeids)
        iconnectivity = np.arange(len(self._el_df_indices))
        data = {'shape': self._el_df_eleshapes,
                'connectivity': self._el_df_connectivity,
                'iconnectivity': iconnectivity}
        el_df = pd.DataFrame(data, index=self._el_df_indices)

        # Write tags into the dataframe
        for tag_name, tag_value_dict in self._tags.items():
            el_df[tag_name] = None
            if tag_value_dict is not None:
                for tag_value, elem_list in tag_value_dict.items():
                    try:
                        el_df.loc[elem_list, (tag_name)] = tag_value
                    except:
                        temp_list = el_df[tag_name].tolist()
                        for elem in elem_list:
                            temp_list[elem] = tag_value
                        el_df[tag_name] = temp_list

        # Building the meshcontainer for return
        meshcontainer = {'nodes': nodes_df,
                         'elements': el_df,
                         'groups': self._groups,
                         'dimension': self._dimension,
                         }
        return meshcontainer
