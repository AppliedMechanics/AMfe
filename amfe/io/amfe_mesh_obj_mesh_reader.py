#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
AMfe mesh object reader for I/O module.
"""

from .mesh_reader import MeshReader
from .amfe_mesh_converter import AmfeMeshConverter

__all__ = [
    'AmfeMeshObjMeshReader'
    ]


class AmfeMeshObjMeshReader(MeshReader):
    """
    Reader for AMfe mesh objects.
    """

    def __init__(self, meshobj=None, builder=AmfeMeshConverter()):
        super().__init__()
        self._builder = builder
        self._meshobj = meshobj
        return

    def parse(self):
        # build dimension
        self._builder.build_mesh_dimension(self._meshobj.dimension)
        self._builder.build_no_of_nodes(self._meshobj.no_of_nodes)
        self._builder.build_no_of_elements(self._meshobj.no_of_elements + self._meshobj.no_of_boundary_elements)
        # build nodes
        nodeid2idx = self._meshobj.nodeid2idx
        if self._meshobj.dimension == 2:
            for nodeid in nodeid2idx:
                self._builder.build_node(nodeid,
                                         self._meshobj.nodes[nodeid2idx[nodeid], 0],
                                         self._meshobj.nodes[nodeid2idx[nodeid], 1],
                                         0.0)
        else:
            for nodeid in nodeid2idx:
                self._builder.build_node(nodeid,
                                         self._meshobj.nodes[nodeid2idx[nodeid], 0],
                                         self._meshobj.nodes[nodeid2idx[nodeid], 1],
                                         self._meshobj.nodes[nodeid2idx[nodeid], 2])
        # build elements
        elementid2idx = self._meshobj.eleid2idx
        connectivitylist = (self._meshobj.connectivity, self._meshobj.boundary_connectivity)
        eleshapeslist = (self._meshobj.ele_shapes, self._meshobj.boundary_ele_shapes)
        for elementid in elementid2idx:
            vol_boundary_flag, idx = elementid2idx[elementid]
            etype = eleshapeslist[vol_boundary_flag][idx]
            connectivity = connectivitylist[vol_boundary_flag][idx]
            connectivity = self._meshobj.get_nodeids_by_nodeidxs(connectivity)
            self._builder.build_element(elementid, etype, connectivity)
        # build groups
        for group in self._meshobj.groups:
            self._builder.build_group(group,
                                      self._meshobj.groups[group]['nodes'],
                                      self._meshobj.groups[group]['elements'])
        return self._builder.return_mesh()
