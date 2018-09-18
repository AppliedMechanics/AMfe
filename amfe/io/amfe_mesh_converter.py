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

from .mesh_converter import MeshConverter
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

    >>> from amfe.io import GidJsonMeshReader, AmfeMeshConverter
    >>> filename = '/path/to/your/file.json'
    >>> converter = AmfeMeshConverter()
    >>> reader = GidJsonMeshReader(filename, converter)
    >>> mymesh = reader.parse()

    """

    element_2d_set = {'Tri6', 'Tri3', 'Quad4', 'Quad8', }
    element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}

    boundary_2d_set = {'straight_line', 'quadratic_line'}
    boundary_3d_set = {'straight_line', 'quadratic_line', 'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}

    def __init__(self, verbose=False):
        super().__init__()
        self._verbose = verbose
        self._mesh = Mesh()
        self._dimension = None
        self._no_of_nodes = None
        self._no_of_elements = None
        self._nodes = np.empty((0, 3), dtype=float)
        self._currentnodeid = 0
        self._currentelementid = 0
        self._connectivity = list()
        self._groups = dict()
        # mapping from reader-nodeid to amfe-nodeid
        self._nodeid2idx = dict()
        # df information
        self._el_df_indices = list()
        self._el_df_eleshapes = list()
        self._el_df_connectivity_idx = list()
        self._el_df_is_boundary = list()
        return

    def build_no_of_nodes(self, no):
        # This function is only used for preallocation
        # It is not necessary to call, but useful if information about no_of_nodes exists
        self._no_of_nodes = no
        if self._nodes.shape[0] == 0:
            self._nodes = np.zeros((no, 3), dtype=float)
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
            self._nodes[amfeid, :] = [x, y, z]
        else:
            # append node if array is not preallocated with full node dimension
            self._nodes = np.append(self._nodes, np.array([x, y, z], dtype=float, ndmin=2), axis=0)
        # add mapping information
        self._nodeid2idx.update({idx: amfeid})
        # increment row-index counter
        self._currentnodeid += 1
        return

    def build_element(self, idx, etype, nodes):
        # append connectivity information
        self._connectivity.append(np.array(nodes, dtype=int))
        # update df information
        self._el_df_connectivity_idx.append(self._currentelementid)
        self._el_df_indices.append(idx)
        self._el_df_eleshapes.append(etype)
        # append element type information
        # Hint: The differentiation between volume and boundary elements will be performed after all
        # elements have been read
        # increment row-index counter
        self._currentelementid += 1
        return

    def build_group(self, name, nodeids=None, elementids=None):
        # append group information
        group = {name: {'nodes': nodeids, 'elements': elementids}}
        self._groups.update(group)
        return

    def return_mesh(self):
        # Check dimension of model
        if self._dimension is None:
            if not self.element_3d_set.intersection(set(self._el_df_eleshapes)):
                # No 3D element in eleshapes, thus:
                self._dimension = 2
            else:
                self._dimension = 3
        # If dimension = 2 cut the z coordinate
        if self._dimension == 2:
            self._mesh.nodes = self._nodes[:, :self._dimension]
        # set the node mapping information
        self._mesh.nodeid2idx = self._nodeid2idx
        # divide in boundary and volume elements
        currentidx = 0
        currentboundaryidx = 0
        if self._dimension == 3:
            volume_element_set = self.element_3d_set
            boundary_element_set = self.boundary_3d_set
        elif self._dimension == 2:
            volume_element_set = self.element_2d_set
            boundary_element_set = self.boundary_2d_set
        else:
            raise ValueError('Dimension must be 2 or 3')

        # write properties
        self._mesh.dimension = self._dimension
        self._mesh.connectivity = [np.array([self._nodeid2idx[nodeid] for nodeid in element])
                                   for index, element in enumerate(self._connectivity)]
        self._el_df_is_boundary = len(self._connectivity)*[False]
        for index, shape in enumerate(self._el_df_eleshapes):
            if shape in boundary_element_set:
                self._el_df_is_boundary[index] = True
        data = {'connectivity_idx': self._el_df_connectivity_idx,
                'shape': self._el_df_eleshapes,
                'is_boundary': self._el_df_is_boundary}
        self._mesh.el_df = pd.DataFrame(data, index=self._el_df_indices)

        self._mesh.groups = self._groups
        return self._mesh
