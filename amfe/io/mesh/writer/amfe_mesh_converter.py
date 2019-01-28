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
from amfe.io.mesh.constants import VOLUME_ELEMENTS_2D, VOLUME_ELEMENTS_3D, BOUNDARY_ELEMENTS_2D, BOUNDARY_ELEMENTS_3D
from amfe.mesh import Mesh

__all__ = [
    'AmfeMeshConverter'
    ]


class AmfeMeshConverter(MeshConverter):
    """
    Converter for AMfe meshes.

    Examples
    --------
    Convert GiD json file to AMfe mesh:

    >>> from amfe.io.mesh.reader import GidJsonMeshReader
    >>> from amfe.io.mesh.writer import AmfeMeshConverter
    >>> filename = '/path/to/your/file.json'
    >>> converter = AmfeMeshConverter()
    >>> reader = GidJsonMeshReader(filename)
    >>> reader.parse(converter)
    >>> converter.return_mesh()

    """

    def __init__(self, verbose=False):
        super().__init__()
        self._verbose = verbose
        self._mesh = Mesh()
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
        self._el_df_is_boundary = list()
        return

    def build_no_of_nodes(self, no):
        # This function is only used for preallocation
        # It is not necessary to call, but useful if information about no_of_nodes exists
        self._no_of_nodes = no
        if self._nodes.shape[0] == 0:
            self._nodes = np.zeros((no, 4), dtype=float)
        return

    def build_no_of_elements(self, no):
        # This function is not used
        # If someone wants to improve performance he/she can add preallocation functionality for elements
        self._no_of_elements = no
        # preallocation...
        return

    def build_mesh_dimension(self, dim):
        self._dimension = dim
        return

    def build_node(self, idx, x, y, z):
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
        # update df information
        self._el_df_connectivity.append(np.array(nodes, dtype=int))
        self._el_df_indices.append(idx)
        self._el_df_eleshapes.append(etype)
        return

    def build_group(self, name, nodeids=None, elementids=None):
        # append group information
        group = {name: {'nodes': nodeids, 'elements': elementids}}
        self._groups.update(group)
        return

    def build_tag(self, tag_dict):
        # append tag information
        self._tags.update(tag_dict)
        return None

    def return_mesh(self):
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
        if self._dimension == 2:
            self._mesh.nodes_df = pd.DataFrame({'x': x, 'y': y}, index=np.array(self._nodes[:, 0], dtype=int))
        else:
            z = self._nodes[:, 3]
            self._mesh.nodes_df = pd.DataFrame({'x': x, 'y': y, 'z': z}, index=np.array(self._nodes[:, 0], dtype=int))

        # divide in boundary and volume elements
        if self._dimension == 3:
            volume_element_set = VOLUME_ELEMENTS_3D
            boundary_element_set = BOUNDARY_ELEMENTS_3D
        elif self._dimension == 2:
            volume_element_set = VOLUME_ELEMENTS_2D
            boundary_element_set = BOUNDARY_ELEMENTS_2D
        else:
            raise ValueError('Dimension must be 2 or 3')

        # write properties
        self._mesh.dimension = self._dimension

        self._el_df_is_boundary = len(self._el_df_connectivity)*[False]
        for index, shape in enumerate(self._el_df_eleshapes):
            if shape in boundary_element_set:
                self._el_df_is_boundary[index] = True
        data = {'shape': self._el_df_eleshapes,
                'is_boundary': self._el_df_is_boundary,
                'connectivity': self._el_df_connectivity}
        self._mesh.el_df = pd.DataFrame(data, index=self._el_df_indices)

        self._mesh.groups = self._groups
        
        for tag_name, tag_dict in self._tags.items():
            self._mesh.insert_tag(tag_name, tag_dict)

        return self._mesh
