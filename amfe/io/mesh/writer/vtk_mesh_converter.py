#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all mesh converter for I/O module.
"""

from enum import Enum
import numpy as np
import vtk
from os.path import splitext
import logging

from amfe.io.mesh.base import MeshConverter

__all__ = [
    'VtkMeshConverter'
    ]


class VtkPreallocation(Enum):
    PREALLOCATED = 1
    NOTPREALLOCATED = 2
    UNKNOWN = 3


class VtkMeshConverter(MeshConverter):
    """
    Super class for all mesh converters.
    """

    amfe2vtk = {'straight_line': vtk.vtkLine,
                'quadratic_line': vtk.vtkQuadraticEdge,
                'Tri3': vtk.vtkTriangle,
                'Tri6': vtk.vtkQuadraticTriangle,
                'Quad4': vtk.vtkQuad,
                'Quad8': vtk.vtkQuadraticQuad,
                'Tet4': vtk.vtkTetra,
                'Tet10': vtk.vtkQuadraticTetra,
                'Hexa8': vtk.vtkHexahedron,
                'Hexa20': vtk.vtkQuadraticHexahedron,
                'Prism6': vtk.vtkWedge }

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._vtknodes = vtk.vtkPoints()
        self._vtkelements = vtk.vtkUnstructuredGrid()
        self._nodes = np.empty((0, 4))
        self._elements = []
        self._nodespreallocation = VtkPreallocation.UNKNOWN
        self._elementspreallocation = VtkPreallocation.UNKNOWN
        self._eleid2cell = dict()
        self._tags = dict()
        self._groups = dict()
        self.logger = logging.getLogger('amfe.io.VtkMeshConverter')

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
        if self._nodespreallocation is VtkPreallocation.UNKNOWN:
            self._vtknodes.SetNumberOfPoints(no)
            self._nodespreallocation = VtkPreallocation.PREALLOCATED

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
        if self._elementspreallocation is VtkPreallocation.UNKNOWN:
            self._vtkelements.Allocate(no, 1)
            self._elementspreallocation = VtkPreallocation.PREALLOCATED

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
        if self._nodespreallocation is VtkPreallocation.PREALLOCATED:
            self._vtknodes.InsertPoint(idx, (x, y, z))
        else:
            self._nodes = np.concatenate((self._nodes, np.array([idx, x, y, z], ndmin=2, dtype=float)), axis=0)
            self._nodespreallocation = VtkPreallocation.NOTPREALLOCATED

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
        cell = self.amfe2vtk[etype]()
        for i, node in enumerate(nodes):
            cell.GetPointIds().SetId(i, node)
            self._eleid2cell.update({idx: cell})
        if self._elementspreallocation is VtkPreallocation.PREALLOCATED:
            cellid = self._vtkelements.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        else:
            if self._elementspreallocation is VtkPreallocation.UNKNOWN:
                self._vtkelements.Allocate(1, 1)
                self._elementspreallocation = VtkPreallocation.NOTPREALLOCATED
            cellid = self._vtkelements.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        self._eleid2cell.update({idx: cellid})

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
        self._groups.update({name: {'elements': elementids, 'nodes': nodeids}})

    def build_material(self, material):
        pass

    def build_partition(self, partition):
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
        self._tags.update(tag_dict)

    def return_mesh(self):
        """
        Returns the Mesh or the file pointer

        Returns
        -------
        Object
        """
        if self._nodespreallocation is not VtkPreallocation.PREALLOCATED:
            self._vtknodes.SetNumberOfPoints(self._nodes.shape[0])
            for node in self._nodes:
                self._vtknodes.InsertPoint(int(node[0]), node[1:])

        self._vtkelements.SetPoints(self._vtknodes)

        for tagname, tag_dict in self._tags.items():
            vtkarray = vtk.vtkIntArray()
            vtkarray.SetNumberOfComponents(1)
            vtkarray.SetNumberOfTuples(self._vtkelements.GetNumberOfCells())
            vtkarray.FillComponent(0, 0)
            vtkarray.SetName(tagname)
            for tagvalue, eleids in tag_dict.items():
                vtkids = [self._eleid2cell[eleid] for eleid in eleids]
                for vtkid in vtkids:
                    vtkarray.SetTuple1(vtkid, int(tagvalue))
            self._vtkelements.GetCellData().AddArray(vtkarray)

        for groupname, groupdict in self._groups.items():
            nodeids = groupdict['nodes']
            elementids = groupdict['elements']
            elementids = [self._eleid2cell[eleid] for eleid in elementids]

            if len(elementids) > 0:
                # Allocate vtk elements array
                vtkelements = vtk.vtkIntArray()
                vtkelements.SetNumberOfComponents(1)
                vtkelements.SetNumberOfTuples(self._vtkelements.GetNumberOfCells())
                vtkelements.SetName(groupname + '_elements')
                vtkelements.FillComponent(0, 0)
                for elementid in elementids:
                    vtkelements.SetTuple1(elementid, 1)
                self._vtkelements.GetCellData().AddArray(vtkelements)

            if len(nodeids) > 0:
                vtknodes = vtk.vtkIntArray()
                vtknodes.SetNumberOfComponents(1)
                vtknodes.SetNumberOfTuples(self._vtknodes.GetNumberOfPoints())
                vtknodes.SetName(groupname + '_nodes')
                vtknodes.FillComponent(0, 0)
                for nodeid in nodeids:
                    vtknodes.SetTuple1(nodeid, 1)
                self._vtkelements.GetPointData().AddArray(vtknodes)

        filename, file_extension = splitext(self._filename)
        if file_extension == '.vtu':
            vtkwriter = vtk.vtkXMLUnstructuredGridWriter()
        elif file_extension == '.vtk':
            vtkwriter = vtk.vtkUnstructuredGridWriter()
        else:
            self.logger.warning('No file extension given, choose \'vtk\' format')
            self._filename = self._filename + '.vtk'
            vtkwriter = vtk.vtkUnstructuredGridWriter()

        vtkwriter.SetInputData(self._vtkelements)
        vtkwriter.SetFileName(self._filename)
        vtkwriter.Write()
        return 0
