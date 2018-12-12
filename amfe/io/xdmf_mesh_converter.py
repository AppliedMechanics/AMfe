# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from enum import Enum
import numpy as np
import pandas as pd
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
        NOTPREALLOCATED = 2
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
        self._tag_dict = dict()
        self._connectivity = list()
        self._ele_indices = list()
        self._eleshapes = list()

    def build_no_of_nodes(self, no):
        if self._node_preallocation == self.Preallocation.UNKNOWN:
            self._nodes = np.zeros((no, 4))
            self._node_preallocation = self.Preallocation.PREALLOCATED

    def build_no_of_elements(self, no):
        pass

    def build_node(self, idx, x, y, z):
        if self._node_preallocation == self.Preallocation.PREALLOCATED:
            self._nodes[self._nodes_current_row, :] = [float(idx), float(x), float(y), float(z)]
        else:
            if self._node_preallocation == self.Preallocation.UNKNOWN:
                self._nodes = np.empty((0, 4), dtype=float)
                self._node_preallocation = self.Preallocation.NOTPREALLOCATED
            self._nodes = np.append(self._nodes, np.array([idx, x, y, z], ndmin=2, dtype=float), axis=0)
            self._nodeids2row.update({id: self._nodes_current_row})
            self._nodes_current_row += 1

    def build_element(self, eid, etype, nodes):
        self._connectivity.append(np.array(nodes, ndmin=2, dtype=int))
        self._ele_indices.append(eid)
        self._eleshapes.append(etype)

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
        self._tag_dict.update(tag_dict)

    def return_mesh(self):
        # Make a pd Dataframe for the nodes
        # If dimension = 2 cut the z coordinate
        x = self._nodes[:, 1]
        y = self._nodes[:, 2]
        z = self._nodes[:, 3]
        no_of_nodes = self._nodes.shape[0]

        nodes_df = pd.DataFrame({'row': np.arange(no_of_nodes), 'x': x, 'y': y, 'z': z}, index=np.array(self._nodes[:, 0], dtype=int))

        # make a pd Dataframe for the elements
        data = {'shape': self._eleshapes,
                'connectivity': self._connectivity}
        el_df = pd.DataFrame(data, index=self._ele_indices)
        # introduce row values for each shape. The row values will map to the row entries in an separate array for each
        # elementtype
        etypes_in_el_df = el_df['shape'].unique()
        el_df['row'] = None
        for etype in etypes_in_el_df:
            el_df.loc[el_df['shape'] == etype, 'row'] = np.arange(sum(el_df['shape'] == etype))

        # Function change connectivity ids to row ids in nodes array:
        def change_connectivity(arr):
            return np.array([nodes_df.loc[node, 'row'] for node in arr], ndmin=2, dtype=int)
        el_df['connectivity'] = el_df['connectivity'].apply(change_connectivity)

        tag_names = list()

        for tag_name, tag_dict in self._tag_dict.items():
            el_df[tag_name] = None
            currentscalar = -1
            name2scalars = dict()
            if tag_dict is not None:
                for tag_value, elem_list in tag_dict.items():
                    # Check if tag_value is string
                    # if it is a string map the string to a scalar because xdmf does not support strings
                    if isinstance(tag_value, str):
                        name2scalars.update({tag_value: currentscalar})
                        tag_value = currentscalar
                        currentscalar -= 1
                    try:
                        el_df.loc[elem_list, (tag_name)] = tag_value
                    except:
                        temp_list = el_df[tag_name].tolist()
                        for elem in elem_list:
                            temp_list[elem] = tag_value
                        el_df[tag_name] = temp_list
            tag_names.extend([tag_name])
            self._tag_dict[tag_name].update({'name2scalars': name2scalars})

        # write hdf5 file
        hdf5filename = self._filename + '.hdf5'
        with h5py.File(hdf5filename, 'w') as hdf_fp:
            with open(self._filename + '.xdmf', 'wb') as xdmf_fp:
                # create mesh group
                group_mesh = hdf_fp.create_group('mesh')
                group_mesh.create_dataset('nodes', nodes_df[['x', 'y', 'z']].values.shape, float,
                                          nodes_df[['x', 'y', 'z']].values)
                group_elements = group_mesh.create_group('topology')
                group_tags = group_mesh.create_group('tags')
                for tag in tag_names:
                    tag_group = group_tags.create_group(tag)

                relative_hdf5_path = basename(self._filename)
                root = ET.Element('Xdmf', {'Version': '3.0'})
                domain = ET.SubElement(root, 'Domain')  # , {'Type': 'Uniform'})
                collection = ET.SubElement(domain, 'Grid', {'GridType': 'Collection', 'CollectionType': 'spatial'})

                # write topology for each element
                for etype in el_df['shape'].unique():
                    connectivity_of_current_etype = el_df[el_df['shape'] == etype]['connectivity'].values
                    no_of_elements_of_current_etype = len(connectivity_of_current_etype)
                    no_of_nodes_per_element = connectivity_of_current_etype[0].shape[1]
                    connectivity_array = np.concatenate(connectivity_of_current_etype, axis=0)
                    if no_of_elements_of_current_etype > 0:
                        group_elements.create_dataset(etype, (no_of_elements_of_current_etype, no_of_nodes_per_element),
                                                      int, connectivity_array)

                    grid = ET.SubElement(collection, 'Grid', {'Name': 'mesh', 'GridType': 'Uniform'})
                    topology = ET.SubElement(grid, 'Topology',
                                             {'NumberOfElements': str(no_of_elements_of_current_etype),
                                              'TopologyType': self.ELEMENTS[etype]['xdmf_name']})
                    elementdata = ET.SubElement(topology, 'DataItem',
                                                {'Dimensions': '{} {}'.format(no_of_elements_of_current_etype,
                                                                              no_of_nodes_per_element),
                                                 'Format': 'HDF', 'NumberType': 'Int'})
                    elementdata.text = relative_hdf5_path + '.hdf5:/mesh/topology/' + etype
                    geometry = ET.SubElement(grid, 'Geometry', {'GeometryType': 'XYZ'})
                    nodedata = ET.SubElement(geometry, 'DataItem', {'Dimensions': '{} {}'.format(
                        no_of_nodes, 3), 'Format': 'HDF', 'NumberType': 'Float'})
                    nodedata.text = relative_hdf5_path + '.hdf5:/mesh/nodes'

                    for tag in tag_names:
                        # write values for each element type
                        tag_values = el_df[el_df['shape'] == etype][tag].values
                        group_tags.create_dataset('{}/{}'.format(tag, etype), tag_values.shape, dtype=float,
                                                  data=tag_values.astype(float))

                        attribute = ET.SubElement(grid, 'Attribute', {'Name': tag, 'Center': 'Cell',
                                                                      'AttributeType': 'Scalar'})
                        attribute_data = ET.SubElement(attribute, 'DataItem',
                                                       {'Dimensions': '{} 1'.format(no_of_elements_of_current_etype),
                                                        'Format': 'HDF'})
                        attribute_data.text = relative_hdf5_path + '.hdf5:/mesh/tags/{}/{}'.format(tag, etype)

                prettify_xml(root)
                tree = ET.ElementTree(root)
                tree.write(xdmf_fp, 'UTF-8', True)
