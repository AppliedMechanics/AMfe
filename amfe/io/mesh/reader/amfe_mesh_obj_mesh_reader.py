#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
AMfe mesh object reader for I/O module.
"""

from amfe.io.mesh.base import MeshReader

__all__ = [
    'AmfeMeshObjMeshReader'
    ]


class AmfeMeshObjMeshReader(MeshReader):
    """
    Reader for AMfe mesh objects.
    """

    def __init__(self, meshobj=None):
        super().__init__()
        self._meshobj = meshobj
        return

    def parse(self, builder, verbose=False):
        """

        Parameters
        ----------
        builder : MeshConverter
            MeshConverter object that builds the mesh
        verbose : bool
            If True, verbose mode is activated

        Returns
        -------
        None
        """
        # build dimension
        builder.build_mesh_dimension(self._meshobj.dimension)
        builder.build_no_of_nodes(self._meshobj.no_of_nodes)
        builder.build_no_of_elements(self._meshobj.no_of_elements + self._meshobj.no_of_boundary_elements)
        # build nodes
        if self._meshobj.dimension == 2:
            for index, row in self._meshobj.nodes_df.iterrows():
                builder.build_node(index, row['x'], row['y'], 0.0)
        else:
            for index, row in self._meshobj.nodes_df.iterrows():
                builder.build_node(index, row['x'], row['y'], row['z'])

        # build elements
        for elementid, element in self._meshobj.el_df.iterrows():
            etype = element['shape']
            connectivity = list(element['connectivity'])
            builder.build_element(elementid, etype, connectivity)
        # build groups
        for group in self._meshobj.groups:
            builder.build_group(group,
                                self._meshobj.groups[group]['nodes'],
                                self._meshobj.groups[group]['elements'])
        return
