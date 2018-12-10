# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from enum import Enum
import numpy as np
import xml.etree.ElementTree as ET
import h5py
from os.path import basename

from .mesh_converter import MeshConverter
from .tools import prettify_xml

__all__ = [
    'XdmfMeshConverter'
    ]


class XdmfMeshConverter(MeshConverter):
    """
    Super class for all mesh converters.
    """
    class Preallocation(Enum):
        PREALLOCATED = 1
        NOTPREALLOCATD = 2
        UNKNOWN = 0

    ELEMENTS = {'Tri3': {'no_of_nodes': 3, 'xdmf_name': 'Triangle'},
                'Tri6': {'no_of_nodes': 6, 'xdmf_name': 'Triangle_6'},
                'Quad4': {'no_of_nodes': 4, 'xdmf_name': 'Quadrilateral'},
                'Quad8': {'no_of_nodes': 8, 'xdmf_name': 'Quadrilateral_8'},
                'Hex8': {'no_of_nodes': 8, 'xdmf_name': 'Hexahedron'},
                'Hex20': {'no_of_nodes': 20, 'xdmf_name': 'Hexahedron_20'},
                'Tet4': {'no_of_nodes': 4, 'xdmf_name': 'Tetrahedron'},
                'Tet10': {'no_of_nodes': 10, 'xdmf_name': 'Tetrahedron_10'},
                'straight_line': {'no_of_nodes': 2, 'xdmf_name': 'Polyline'},
                'quadratic_line': {'no_of_nodes': 3, 'xdmf_name': 'Edge_3'},
                }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'filename' in kwargs:
            self._filename = kwargs['filename']
        else:
            self._filename = args[0]
        self._root = ET.Element('Xdmf', {'Version': '2.0'})
        self._nodes = np.empty((0, 3), dtype=float)
        self._nodes_current_row = 0
        self._node_preallocation = self.Preallocation.UNKNOWN
        self._nodeids2row = dict()
        self._elements = dict()

    def build_no_of_nodes(self, no):
        if self._node_preallocation == self.Preallocation.UNKNOWN:
            self._nodes = np.zeros((no, 3))

    def build_no_of_elements(self, no):
        pass

    def build_node(self, id, x, y, z):
        if self._node_preallocation == self.Preallocation.PREALLOCATED:
            self._nodes[self._nodes_current_row, :] = [float(x), float(y), float(z)]
            if id not in self._nodeids2row:
                self._nodeids2row.update({id: self._nodes_current_row})
                self._nodes_current_row += 1
            else:
                raise ValueError('Nodeid already in nodeids2row')
        else:
            self._nodes = np.append(self._nodes, np.array([x, y, z], ndmin=2, dtype=float), axis=0)
            self._nodeids2row.update({id: self._nodes_current_row})
            self._nodes_current_row += 1
            if self._node_preallocation == self.Preallocation.UNKNOWN:
                self._node_preallocation = self.Preallocation.NOTPREALLOCATD

    def build_element(self, eid, etype, nodes):
        connectivity = np.array(nodes, ndmin=2, dtype=int)
        if etype not in self._elements:
            self._elements.update({etype: np.empty((0, self.ELEMENTS[etype]['no_of_nodes']))})
        self._elements[etype] = np.append(self._elements[etype], connectivity, 0)

    def build_group(self, name, nodeids, elementids):
        """

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

    def build_material(self, material):
        pass

    def build_partition(self, partition):
        pass

    def build_mesh_dimension(self, dim):
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
        pass

    def return_mesh(self):

        hdf5filename = self._filename + '.hdf5'
        with h5py.File(hdf5filename, 'w') as hdf_fp:
            group_mesh = hdf_fp.create_group('mesh')
            group_mesh.create_dataset('nodes', self._nodes.shape, float, self._nodes)
            group_elements = group_mesh.create_group('topology')
            for etype in self._elements:
                self._elements[etype] = np.vectorize(self._nodeids2row.get)(self._elements[etype])
                group_elements.create_dataset(etype, self._elements[etype].shape,
                                              int, self._elements[etype])

        with open(self._filename + '.xdmf', 'wb') as xdmf_fp:
            prologue = '<?xml version="1.0" ?>\n'
            doctype = '<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n'

            #xdmf_fp.write(prologue.encode('UTF-8'))
            # xdmf_fp.write(doctype)
            relative_hdf5_path = basename(self._filename)
            root = ET.Element('Xdmf', {'Version': '3.0'})
            domain = ET.SubElement(root, 'Domain')  # , {'Type': 'Uniform'})
            collection = ET.SubElement(domain, 'Grid', {'GridType': 'Collection', 'CollectionType': 'spatial'})
            for etype, arr in self._elements.items():
                grid = ET.SubElement(collection, 'Grid', {'Name': 'mesh', 'GridType': 'Uniform'})
                no_of_elements = arr.shape[0]
                topology = ET.SubElement(grid, 'Topology', {'NumberOfElements': str(no_of_elements),
                                                        'TopologyType': self.ELEMENTS[etype]['xdmf_name']})
                elementdata = ET.SubElement(topology, 'DataItem', {'Dimensions': '{} {}'.format(arr.shape[0],
                                                                                                arr.shape[1]),
                                                                   'Format': 'HDF', 'NumberType': 'Int'})
                elementdata.text = relative_hdf5_path + '.hdf5:/mesh/topology/' + etype
                geometry = ET.SubElement(grid, 'Geometry', {'GeometryType': 'XYZ'})
                nodedata = ET.SubElement(geometry, 'DataItem', {'Dimensions': '{} {}'.format(
                    self._nodes.shape[0], self._nodes.shape[1]), 'Format': 'HDF', 'NumberType': 'Float'})
                nodedata.text = relative_hdf5_path + '.hdf5:/mesh/nodes'
            prettify_xml(root)
            tree = ET.ElementTree(root)
            tree.write(xdmf_fp, 'UTF-8', True)
