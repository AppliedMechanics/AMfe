# -*- coding: utf-8 -*-
"""
Tests for testing io module
"""

from unittest import TestCase
from os.path import join, abspath, dirname
from os import makedirs
import os
import numpy as np
import pandas as pd
import h5py
import pickle
from numpy.testing import assert_allclose, assert_array_equal, assert_array_almost_equal
from pandas.testing import assert_frame_equal

from amfe.component import StructuralComponent
from amfe.material import KirchhoffMaterial
from amfe.solver import AmfeSolution, AmfeSolutionHdf5

# Import Mesh Reader
from amfe.io.mesh.reader import GidAsciiMeshReader, GidJsonMeshReader, GmshAsciiMeshReader, AmfeMeshObjMeshReader, \
    Hdf5MeshReader

# Import Mesh Writer
from amfe.io.mesh.writer import AmfeMeshConverter, VtkMeshConverter, Hdf5MeshConverter, write_xdmf_mesh_from_hdf5
from amfe.io.mesh.base import MeshConverter

# Import Postprocessing Tools
from amfe.io.postprocessing import *
from amfe.io.postprocessing.tools import *
# Import Postprocessing Reader
from amfe.io.postprocessing.reader import AmfeHdf5PostProcessorReader, AmfeSolutionReader

# Import Postprocessingwriter
from amfe.io.postprocessing.writer import Hdf5PostProcessorWriter
from amfe.io.postprocessing.base import PostProcessorWriter

from amfe.mesh import Mesh

from tests.tools import CustomDictAssertTest
from tests.io_tools import load_object, create_amfe_obj, clean_test_outputs


class DummyMeshConverter(MeshConverter):
    def __init__(self):
        super().__init__()
        self.no_of_elements = None
        self.no_of_nodes = None
        self.nodes = []
        self.elements = []
        self.groups = []
        self.materials = []
        self.partitions = []
        self.dimension = None
        self.mesh = None
        self.tags = dict()

    def build_no_of_nodes(self, no):
        self.no_of_nodes = no

    def build_no_of_elements(self, no):
        self.no_of_elements = no

    def build_node(self, nodeid, x, y, z):
        self.nodes.append((nodeid, x, y, z))

    def build_element(self, eleid, etype, nodes):
        self.elements.append((eleid, etype, nodes))

    def build_group(self, name, nodeids=None, elementids=None):
        self.groups.append((name, nodeids, elementids))

    def build_mesh_dimension(self, dim):
        self.dimension = dim

    def build_tag(self, tag_name, values2elements, dtype=None, default=None):
        self.tags.update({tag_name: {'values2elements': values2elements,
                                      'dtype': dtype,
                                      'default': default}})

    def return_mesh(self):
        return self


class DummyPostProcessorWriter(PostProcessorWriter):
    def __init__(self, meshreaderobj):
        super().__init__(meshreaderobj)
        self._meshreader = meshreaderobj
        self._fields = dict()

    def write_field(self, name, field_type, t, data, index, mesh_entity_type):
        fielddict = {'data_type': field_type,
                     'timesteps': t,
                     'index': index,
                     'mesh_entity_type': mesh_entity_type,
                     'data': data
                     }
        if name in fielddict:
            raise ValueError('Field already written')
        self._fields.update({name: fielddict})

    def return_result(self):
        return self._fields


class IOTest(TestCase):
    def setUp(self):
        directory = join(dirname(abspath(__file__)), '.results')
        if os.path.exists(directory):
            clean_test_outputs(directory)
        self.custom_asserter = CustomDictAssertTest()

    def tearDown(self):
        pass

    def test_gidascii_to_dummy(self):
        # Desired nodes
        nodes_desired = [(1, 1.34560000e-02, 3.56167570e-02, 0.00000000e+00),
                         (2, 1.02791191e+00, 3.91996620e-02, 1.23863900e-03),
                         (3, 6.35836584e-02, 1.05638658e+00, 8.97892300e-03),
                         (4, 1.05296566e+00, 1.04992142e+00, 5.77563650e-03),
                         (5, 2.04236782e+00, 4.27825670e-02, 2.47727800e-03),
                         (6, 2.04234766e+00, 1.04345626e+00, 2.57235000e-03)]
        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        elements_desired = [(1, 'Tri3', [5, 6, 4]),
                            (2, 'Tri3', [4, 3, 2]),
                            (3, 'Tri3', [4, 2, 5]),
                            (4, 'Tri3', [1, 2, 3])]
        dimension_desired = 3

        # -------------------------------------------------------
        # OTHER INFORMATION NOT AVAILABLE FROM ASCII MESH
        # THEREFORE ONLY nodes, elements and dimension is tested
        # -------------------------------------------------------

        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', 'gid_ascii_4_tets.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidAsciiMeshReader(file)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh.nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh.elements[i], element)
        # Check mesh dimension
        self.assertEqual(mesh.dimension, dimension_desired)

    def test_gidjson_to_dummy(self):
        # Desired nodes
        nodes_desired = [(1, 1.345600000e-02, 3.561675700e-02, 0.000000000e+00),
                         (2, 5.206839561e-01, 3.740820950e-02, 6.193195000e-04),
                         (3, 3.851982918e-02, 5.460016703e-01, 4.489461500e-03),
                         (4, 5.457667372e-01, 5.477935420e-01, 6.984401105e-04),
                         (5, 1.027911912e+00, 3.919966200e-02, 1.238639000e-03),
                         (6, 6.358365836e-02, 1.056386584e+00, 8.978923000e-03),
                         (7, 1.040469476e+00, 5.445628213e-01, 1.301993398e-03),
                         (8, 5.582746582e-01, 1.053154002e+00, 7.377279750e-03),
                         (9, 1.052965658e+00, 1.049921420e+00, 5.775636500e-03),
                         (10, 1.535139868e+00, 4.099111450e-02, 1.857958500e-03),
                         (11, 1.547697432e+00, 5.463542738e-01, 1.921312898e-03),
                         (12, 1.547656658e+00, 1.046688838e+00, 4.173993250e-03),
                         (13, 2.042367825e+00, 4.278256700e-02, 2.477278000e-03),
                         (14, 2.042357741e+00, 5.431194119e-01, 2.524814000e-03),
                         (15, 2.042347658e+00, 1.043456257e+00, 2.572350000e-03)]
        elements_desired = [(1, 'Tri6', [13, 15, 9, 14, 12, 11]),
                            (2, 'Tri6', [9, 6, 5, 8, 4, 7]),
                            (3, 'Tri6', [9, 5, 13, 7, 10, 11]),
                            (4, 'Tri6', [1, 5, 6, 2, 4, 3]),
                            (5, 'quadratic_line', [5, 13, 10]),
                            (6, 'quadratic_line', [1, 5, 2]),
                            (7, 'quadratic_line', [6, 1, 3]),
                            (8, 'quadratic_line', [9, 6, 8]),
                            (9, 'quadratic_line', [13, 15, 14]),
                            (10, 'quadratic_line', [15, 9, 12])
                            ]
        dimension_desired = 2
        groups_desired = [
            ('left', [], [2, 4]),
            ('right', [], [1, 3]),
            ('left_boundary', [], [7]),
            ('right_boundary', [], [9]),
            ('top_boundary', [], [8, 10]),
            ('left_dirichlet', [1, 3, 6], [])
        ]
        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', 'gid_json_4_tets.json')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidJsonMeshReader(file)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()
        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh.nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh.elements[mesh.elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh.dimension, dimension_desired)
        self.assertEqual(mesh.groups, groups_desired)
        self.assertEqual(mesh.no_of_nodes, 15)
        self.assertEqual(mesh.no_of_elements, 10)

    def test_dummy_to_amfe(self):
        # Desired nodes
        self.set_dummy_input()

        converter = AmfeMeshConverter()

        self.run_build_commands(converter, dim=2)

        mesh = converter.return_mesh()

        # CHECK NODES

        nodes_desired = np.array([[node[1], node[2]] for node in self.nodes_input])
        assert_allclose(mesh.nodes, nodes_desired)

        # CHECK CONNECTIVITIES
        # connectivity_desired = [np.array(element[2]) for element in elements_input[:]]
        for element in self.elements_input:
            assert_array_equal(mesh.get_connectivity_by_elementids([element[0]])[0], np.array(element[2], dtype=int))

        # CHECK DIMENSION
        self.assertEqual(mesh.dimension, 2)

        # CHECK NODE MAPPING
        for node in self.nodes_input:
            nodeid = node[0]
            node_actual = mesh.nodes_df.loc[nodeid]
            self.assertAlmostEqual(node_actual['x'], node[1])
            self.assertAlmostEqual(node_actual['y'], node[2])

        # CHECK ELESHAPES AND ELEMENTMAPPING IN DATAFRAME
        indices = list(np.arange(1, 11))
        data = {'shape': ['Tri6', 'Tri6', 'Tri6', 'Tri6', 'quadratic_line', 'quadratic_line',
                          'quadratic_line', 'quadratic_line', 'quadratic_line', 'quadratic_line'],
                'is_boundary': [False, False, False, False, True, True, True, True, True, True],
                'connectivity': [element[2] for element in self.elements_input],
                **self.tags_desired}
        el_df_desired = pd.DataFrame(data, index=indices)
        for tagname, dtype in zip(self.tags_desired, self.tags_dtypes_desired):
            if dtype == int:
                dtype = pd.Int64Dtype()
            el_df_desired[tagname] = el_df_desired[tagname].astype(dtype)

        assert_frame_equal(mesh.el_df, el_df_desired)

        # CHECK GROUPS
        groups_desired = dict()
        for group in self.groups_input:
            groups_desired.update({group[0]: {'nodes': group[1], 'elements': group[2]}})
        self.assertEqual(mesh.groups, groups_desired)

    def set_dummy_input(self):
        # Desired nodes
        self.nodes_input = [(1, 1.345600000e-02, 3.561675700e-02, 0.000000000e+00),
                            (2, 5.206839561e-01, 3.740820950e-02, 6.193195000e-04),
                            (3, 3.851982918e-02, 5.460016703e-01, 4.489461500e-03),
                            (4, 5.457667372e-01, 5.477935420e-01, 6.984401105e-04),
                            (50, 1.027911912e+00, 3.919966200e-02, 1.238639000e-03),
                            (6, 6.358365836e-02, 1.056386584e+00, 8.978923000e-03),
                            (7, 1.040469476e+00, 5.445628213e-01, 1.301993398e-03),
                            (8, 5.582746582e-01, 1.053154002e+00, 7.377279750e-03),
                            (9, 1.052965658e+00, 1.049921420e+00, 5.775636500e-03),
                            (10, 1.535139868e+00, 4.099111450e-02, 1.857958500e-03),
                            (11, 1.547697432e+00, 5.463542738e-01, 1.921312898e-03),
                            (12, 1.547656658e+00, 1.046688838e+00, 4.173993250e-03),
                            (13, 2.042367825e+00, 4.278256700e-02, 2.477278000e-03),
                            (14, 2.042357741e+00, 5.431194119e-01, 2.524814000e-03),
                            (15, 2.042347658e+00, 1.043456257e+00, 2.572350000e-03)]
        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        self.elements_input = [(1, 'Tri6', [13, 15, 9, 14, 12, 11]),
                               (2, 'Tri6', [9, 6, 50, 8, 4, 7]),
                               (3, 'Tri6', [9, 50, 13, 7, 10, 11]),
                               (4, 'Tri6', [1, 50, 6, 2, 4, 3]),
                               (5, 'quadratic_line', [50, 13, 10]),
                               (6, 'quadratic_line', [1, 50, 2]),
                               (7, 'quadratic_line', [6, 1, 3]),
                               (8, 'quadratic_line', [9, 6, 8]),
                               (9, 'quadratic_line', [13, 15, 14]),
                               (10, 'quadratic_line', [15, 9, 12])
                               ]
        self.groups_input = [
            ('left', [], [2, 4]),
            ('right', [], [1, 3]),
            ('left_boundary', [], [7]),
            ('right_boundary', [], [9]),
            ('top_boundary', [], [8, 10]),
            ('left_dirichlet', [1, 3, 6], [])
        ]

        self.tags_input = {'domain': {'values2elements': {1: [2, 4], 2: [1, 3]},
                                      'dtype': int,
                                      'default': 0},
                           'weight': {'values2elements': {0.0: [1, 2], 2.0: [3, 4]},
                                      'dtype': float,
                                      'default': 1.0}
                           }

        self.tags_input_with_default = {'domain': {'values2elements': {1: [2, 4], 2: [1, 3], 0: [5, 6, 7, 8, 9, 10]},
                                                   'dtype': int,
                                                   'default': None},
                                        'weight': {'values2elements': {0.0: [1, 2], 2.0: [3, 4], 1.0: [5, 6, 7, 8, 9, 10]},
                                                   'dtype': float,
                                                   'default': None}
                                        }

        self.tags_desired = {'domain': [2, 1, 2, 1, 0, 0, 0, 0, 0, 0],
                             'weight': [0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
        self.tags_dtypes_desired = [int, float]

    def run_build_commands(self, converter, dim=3):
        for node in self.nodes_input:
            converter.build_node(node[0], node[1], node[2], node[3])
        for element in self.elements_input:
            converter.build_element(element[0], element[1], element[2])
        for group in self.groups_input:
            converter.build_group(group[0], group[1], group[2])
        for key, tag_dict in self.tags_input.items():
            converter.build_tag(key, tag_dict['values2elements'], tag_dict['dtype'], tag_dict['default'])
        converter.build_mesh_dimension(dim)

    def test_dummy_to_hdf5_and_xdmf(self):
        self.set_dummy_input()

        filename = join('.results', 'hdf5_dummy')
        if not os.path.exists('.results/'):
            os.makedirs('.results')
        hdf5filename = filename + '.hdf5'
        if os.path.isfile(hdf5filename):
            os.remove(hdf5filename)

        converter = Hdf5MeshConverter(hdf5filename)
        self.run_build_commands(converter)
        converter.return_mesh()

        write_xdmf_mesh_from_hdf5(filename + '.xdmf', filename + '.hdf5', '/mesh')

    def test_hdf5_to_dummy(self):

        self.set_dummy_input()

        h5filename = join('.results', 'hdf5_dummy.hdf5')

        meshreader = Hdf5MeshReader(h5filename, '/mesh')
        builder = DummyMeshConverter()
        meshreader.parse(builder)
        amfemesh = builder.return_mesh()

        # compare nodes
        for actual, desired in zip(amfemesh.nodes, self.nodes_input):
            self.assertEqual(actual, desired)
        # compare elements
        for actual, desired in zip(amfemesh.elements, self.elements_input):
            self.assertEqual(actual, desired)
        # compare tags
        self.assertEqual(amfemesh.tags, self.tags_input_with_default)

    def test_dummy_to_vtu(self):
        self.set_dummy_input()

        filename = join('.results', 'vtk_dummy.vtu')
        if not os.path.exists('.results'):
            makedirs('.results')

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk(self):
        self.set_dummy_input()

        filename = join('.results', 'vtk_dummy.vtk')
        if not os.path.exists('.results'):
            makedirs('.results')

        converter = VtkMeshConverter(filename=filename)
        # Run build commands
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_wrong_fileextension(self):
        self.set_dummy_input()

        filename = join('.results', 'vtk_dummy.abc')
        if not os.path.exists('.results'):
            makedirs('.results')

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_with_preallocation(self):
        self.set_dummy_input()

        filename = join('.results', 'vtk_dummy.vtk')
        if not os.path.exists('.results'):
            makedirs('.results')

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        converter.build_no_of_nodes(len(self.nodes_input))
        converter.build_no_of_elements(len(self.elements_input))
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_with_preallocation_too_late(self):
        self.set_dummy_input()

        filename = join('.results', 'vtk_dummy.vtk')
        if not os.path.exists('.results'):
            makedirs('.results')

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        self.run_build_commands(converter)
        converter.build_no_of_nodes(len(self.nodes_input))
        converter.build_no_of_elements(len(self.elements_input))
        converter.return_mesh()

    def test_gmshascii_to_dummy(self):
        # Desired nodes
        nodes_desired = [(1, 0.0, 0.0, 0.0),
                         (2, 2.0, 0.0, 0.0),
                         (3, 2.0, 1.0, 0.0),
                         (4, 0.0, 1.0, 0.0),
                         (5, 0.999999999997388, 0.0, 0.0),
                         (6, 1.000000000004118, 1.0, 0.0),
                         (7, 0.5000000000003766, 0.5, 0.0),
                         (8, 1.500000000000857, 0.5, 0.0)]

        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        elements_desired = [(1, 'straight_line', [2, 3]),
                            (2, 'straight_line', [4, 1]),
                            (3, 'Tri3', [8, 6, 5]),
                            (4, 'Tri3', [5, 6, 7]),
                            (5, 'Tri3', [4, 7, 6]),
                            (6, 'Tri3', [2, 8, 5]),
                            (7, 'Tri3', [2, 3, 8]),
                            (8, 'Tri3', [1, 7, 4]),
                            (9, 'Tri3', [1, 5, 7]),
                            (10, 'Tri3', [3, 6, 8])]

        dimension_desired = 2
        groups_desired = [('right_boundary', [], [1]),
                          ('left_boundary', [], [2]),
                          ('volume', [], [3, 4, 5, 6, 7, 8, 9, 10])]

        tags_desired = {'physical_group': {'values2elements': {2: [1], 3: [2], 1: [3, 4, 5, 6, 7, 8, 9, 10]},
                                           'dtype': int,
                                           'default': 0},
                        'elementary_model_entity': {
                                            'values2elements': {2: [1], 4: [2], 1: [3, 4, 5, 6, 7, 8, 9, 10]},
                                            'dtype': int,
                                            'default': 0}
                                           }
        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', 'gmsh_ascii_8_tets.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GmshAsciiMeshReader(file)
        # Parse dummy mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh.nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh.elements[mesh.elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh.dimension, dimension_desired)
        self.assertEqual(mesh.groups, groups_desired)
        self.assertEqual(mesh.no_of_nodes, 8)
        self.assertEqual(mesh.no_of_elements, 10)
        self.assertEqual(mesh.tags, tags_desired)

    def test_gmshascii_to_dummy_hexa20(self):
        # Desired nodes
        # old ordering: [5, 42, 60, 30, 21, 45, 75, 65, 44, 32, 22,
        # 63, 47, 64, 76, 67, 49, 66, 77, 78]
        element_57_desired = (57, 'Hexa20', [5, 42, 60, 30, 21, 45, 75, 65, 44, 63, 64,
                                             32, 49, 77, 78, 66, 22, 47, 76, 67])

        dimension_desired = 3
        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', 'gmsh_ascii_v2_hexa20.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GmshAsciiMeshReader(file)
        # Parse dummy mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check elements
        self.assertEqual(mesh.elements[mesh.elements.index(element_57_desired)], element_57_desired)
        # Check mesh dimension
        self.assertEqual(mesh.no_of_nodes, 81)
        self.assertEqual(mesh.no_of_elements, 64)
        self.assertEqual(mesh.dimension, dimension_desired)

    def test_gmshascii_to_dummy_tet10(self):

        element_65_desired = (65, 'Tet10', [61, 9, 45, 72, 84, 85, 86, 87, 79, 88])
        dimension_desired = 3
        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', 'gmsh_ascii_v2_tet10.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GmshAsciiMeshReader(file)
        # Parse dummy mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check elements
        self.assertEqual(mesh.elements[mesh.elements.index(element_65_desired)], element_65_desired)
        # Check mesh dimension
        self.assertEqual(mesh.no_of_nodes, 113)
        self.assertEqual(mesh.no_of_elements, 112)
        self.assertEqual(mesh.dimension, dimension_desired)

    def test_gmshascii_to_dummy_physical_surfaces_and_partitions(self):
        # Desired nodes
        nodes_desired = [(1, 0.0, 0.0, 0.0),
                         (2, 0.0, 5.0, 0.0),
                         (3, 5.0, 5.0, 0.0),
                         (4, 5.0, 0.0, 0.0),
                         (5, 10.0, 0.0, 0.0),
                         (6, 10.0, 5.0, 0.0),
                         (7, 0.0, 10.0, 0.0),
                         (8, 10.0, 10.0, 0.0),
                         (9, 2.499999999996199, 0.0, 0.0),
                         (10, 5, 2.499999999996199, 0),
                         (11, 2.5, 5, 0),
                         (12, 0, 2.5, 0),
                         (13, 7.176360840382222, 0, 0),
                         (14, 7.176360840382227, 5, 0),
                         (15, 5, 2.5, 0),
                         (16, 2.499999999996199, 5, 0),
                         (17, 7.176360840382222, 5, 0),
                         (18, 5, 10, 0),
                         (19, 0, 7.176360840382227, 0),
                         (20, 2.5, 2.5, 0),
                         (21, 1.2499999999981, 1.25, 0),
                         (22, 1.25, 3.75, 0),
                         (23, 3.7499999999981, 1.2499999999981, 0),
                         (24, 3.750000000001901, 3.749999999998099, 0),
                         (25, 8.22303669484285, 2.552255592639131, 0),
                         (26, 6.540735812640291, 1.811426457748249, 0),
                         (27, 6.3023563126068, 3.480526007198198, 0),
                         (28, 2.842729718643214, 7.449413862174849, 0),
                         (29, 6.088180420191112, 7.499999999999999, 0),
                         (30, 8.316135315143335, 6.875, 0),
                         (31, 1.335682429659853, 6.156443675639268, 0)]

        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        elements_desired = [(1, 'straight_line', [2, 12]),
                            (2, 'straight_line', [12, 1]),
                            (3, 'straight_line', [5, 6]),
                            (4, 'straight_line', [6, 8]),
                            (5, 'straight_line', [7, 19]),
                            (6, 'straight_line', [19, 2]),
                            (7, 'Tri3', [24, 23, 10]),
                            (8, 'Tri3', [20, 23, 24]),
                            (9, 'Tri3', [4, 23, 9]),
                            (10, 'Tri3', [3, 24, 10]),
                            (11, 'Tri3', [9, 23, 20]),
                            (12, 'Tri3', [9, 20, 21]),
                            (13, 'Tri3', [1, 21, 12]),
                            (14, 'Tri3', [12, 21, 20]),
                            (15, 'Tri3', [11, 20, 24]),
                            (16, 'Tri3', [11, 22, 20]),
                            (17, 'Tri3', [12, 20, 22]),
                            (18, 'Tri3', [3, 11, 24]),
                            (19, 'Tri3', [2, 22, 11]),
                            (20, 'Tri3', [2, 12, 22]),
                            (21, 'Tri3', [4, 10, 23]),
                            (22, 'Tri3', [1, 9, 21]),
                            (23, 'Tri3', [5, 25, 13]),
                            (24, 'Tri3', [6, 14, 25]),
                            (25, 'Tri3', [14, 27, 25]),
                            (26, 'Tri3', [13, 25, 26]),
                            (27, 'Tri3', [4, 13, 26]),
                            (28, 'Tri3', [4, 26, 15]),
                            (29, 'Tri3', [25, 27, 26]),
                            (30, 'Tri3', [3, 15, 27]),
                            (31, 'Tri3', [3, 27, 14]),
                            (32, 'Tri3', [15, 26, 27]),
                            (33, 'Tri3', [5, 6, 25]),
                            (34, 'Tri3', [7, 28, 18]),
                            (35, 'Tri3', [3, 28, 16]),
                            (36, 'Tri3', [7, 19, 28]),
                            (37, 'Tri3', [3, 29, 28]),
                            (38, 'Tri3', [8, 29, 30]),
                            (39, 'Tri3', [3, 17, 29]),
                            (40, 'Tri3', [8, 18, 29]),
                            (41, 'Tri3', [19, 31, 28]),
                            (42, 'Tri3', [2, 16, 31]),
                            (43, 'Tri3', [16, 28, 31]),
                            (44, 'Tri3', [6, 30, 17]),
                            (45, 'Tri3', [2, 31, 19]),
                            (46, 'Tri3', [17, 30, 29]),
                            (47, 'Tri3', [18, 28, 29]),
                            (48, 'Tri3', [6, 8, 30])]

        dimension_desired = 2
        groups_desired = [('x_dirichlet-line', [], [1, 2, 5, 6]),
                          ('x_neumann', [], [3, 4]),
                          ('surface_left', [], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
                          ('surface_right', [], [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]),
                          ('surface_top', [], [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48])]

        tags_desired = {'no_of_mesh_partitions': {'values2elements': {1: [1, 2, 5, 6, 7, 9, 10, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                                                  2: [3, 4, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22]},
                                                  'dtype': int,
                                                  'default': 0},
                        'partition_id': {'values2elements': {2: [1, 2, 4, 5, 6, 13, 14, 16, 17, 19, 20, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                                         1: [3, 7, 8, 9, 10, 11, 12, 15, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]},
                                         'dtype': int,
                                         'default': 0},
                        'partitions_neighbors': {'values2elements': {(): [1, 2, 5, 6, 7, 9, 10, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
                                                (2,): [3, 8, 11, 12, 15, 18, 22],
                                                (1,): [4, 13, 14, 16, 17, 19]},
                                                'dtype': object,
                                                'default': ()},
                        'elementary_model_entity': {'values2elements': {4: [1, 2],
                                                    6: [3],
                                                    11: [4],
                                                    13: [5, 6],
                                                    1: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                                    2: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                                                    3: [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]},
                                                    'dtype': int,
                                                    'default': 0},
                        'physical_group': {'values2elements': {8: [1, 2, 5, 6],
                                           9: [3, 4],
                                           5: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                                           6: [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
                                           7: [34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
                                           },
                                           'dtype': int,
                                           'default': 0}
                        }

        # Define input file path
        here = dirname(abspath(__file__))
        file = join(here, 'meshes', '3_surfaces_2_partitions_mesh.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GmshAsciiMeshReader(file)
        # Parse dummy mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertEqual(mesh.nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh.elements[mesh.elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh.dimension, dimension_desired)
        self.assertEqual(mesh.groups, groups_desired)
        self.custom_asserter.assert_dict_almost_equal(mesh.tags, tags_desired)
        self.assertEqual(mesh.no_of_nodes, 31)
        self.assertEqual(mesh.no_of_elements, 48)

    def test_amfemeshobj_to_dummy(self):
        # Desired nodes
        nodes_desired = [(1, 1.345600000e-02, 3.561675700e-02, 0.0),
                         (2, 5.206839561e-01, 3.740820950e-02, 0.0),
                         (3, 3.851982918e-02, 5.460016703e-01, 0.0),
                         (4, 5.457667372e-01, 5.477935420e-01, 0.0),
                         (5, 1.027911912e+00, 3.919966200e-02, 0.0),
                         (6, 6.358365836e-02, 1.056386584e+00, 0.0),
                         (7, 1.040469476e+00, 5.445628213e-01, 0.0),
                         (8, 5.582746582e-01, 1.053154002e+00, 0.0),
                         (9, 1.052965658e+00, 1.049921420e+00, 0.0),
                         (10, 1.535139868e+00, 4.099111450e-02, 0.0),
                         (11, 1.547697432e+00, 5.463542738e-01, 0.0),
                         (12, 1.547656658e+00, 1.046688838e+00, 0.0),
                         (13, 2.042367825e+00, 4.278256700e-02, 0.0),
                         (14, 2.042357741e+00, 5.431194119e-01, 0.0),
                         (15, 2.042347658e+00, 1.043456257e+00, 0.0)]
        elements_desired = [(1, 'Tri6', [13, 15, 9, 14, 12, 11]),
                            (2, 'Tri6', [9, 6, 5, 8, 4, 7]),
                            (3, 'Tri6', [9, 5, 13, 7, 10, 11]),
                            (4, 'Tri6', [1, 5, 6, 2, 4, 3]),
                            (5, 'quadratic_line', [5, 13, 10]),
                            (6, 'quadratic_line', [1, 5, 2]),
                            (7, 'quadratic_line', [6, 1, 3]),
                            (8, 'quadratic_line', [9, 6, 8]),
                            (9, 'quadratic_line', [13, 15, 14]),
                            (10, 'quadratic_line', [15, 9, 12])
                            ]
        dimension_desired = 2
        groups_desired = [
            ('left', [], [2, 4]),
            ('right', [], [1, 3]),
            ('left_boundary', [], [7]),
            ('right_boundary', [], [9]),
            ('top_boundary', [], [8, 10]),
            ('left_dirichlet', [1, 3, 6], [])
        ]

        tags_desired = {'domain': { 'values2elements': {2: [1, 3], 1: [2, 4], 0: [5, 6, 7, 8, 9, 10]},
                                    'dtype': int,
                                    'default': None
                                    },
                        'weight': { 'values2elements': {0.0: [1, 2], 2.0: [3, 4], 1.0: [5, 6, 7, 8, 9, 10]},
                                    'dtype': float,
                                    'default': None,
                                  }
                        }

        meshobj = create_amfe_obj()

        # Define Reader Object, parse with AmfeMeshConverter
        reader = AmfeMeshObjMeshReader(meshobj)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()
        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertEqual(mesh.nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh.elements[mesh.elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh.dimension, dimension_desired)
        self.assertEqual(mesh.groups, groups_desired)
        self.assertEqual(mesh.no_of_nodes, 15)
        self.assertEqual(mesh.no_of_elements, 10)
        self.assertEqual(mesh.tags, tags_desired)

    def test_gmsh_parser_with_2_partitions(self):

        here = dirname(abspath(__file__))
        msh_filename = join(here, 'meshes', '2_partitions_2quad_mesh.msh')
        reader_obj = GmshAsciiMeshReader(msh_filename)
        converter = AmfeMeshConverter()
        reader_obj.parse(converter)
        mesh_obj = converter.return_mesh()

        self.assertTrue('no_of_mesh_partitions' in mesh_obj.el_df)
        self.assertTrue('partition_id' in mesh_obj.el_df)
        self.assertTrue('partitions_neighbors' in mesh_obj.el_df)

        desired_list_1 = [1, 1, 2, 2]
        desired_list_2 = [1, 1, 1, 2]
        desired_list_3 = [(), (), (2,), (1,)]
        actual_list_1 = mesh_obj.el_df['no_of_mesh_partitions'].tolist()
        actual_list_2 = mesh_obj.el_df['partition_id'].tolist()
        actual_list_3 = mesh_obj.el_df['partitions_neighbors'].tolist()

        self.assertListEqual(actual_list_1, desired_list_1)
        self.assertListEqual(actual_list_2, desired_list_2)
        self.assertListEqual(actual_list_3, desired_list_3)

    def test_gmsh_parser_with_2_partitions_splitboundary(self):

        here = dirname(abspath(__file__))
        msh_filename = join(here, 'meshes', '2_partitions_2quad_mesh_splitboundary.msh')
        reader_obj = GmshAsciiMeshReader(msh_filename)
        converter = AmfeMeshConverter()
        reader_obj.parse(converter)
        mesh_obj = converter.return_mesh()

        self.assertTrue('no_of_mesh_partitions' in mesh_obj.el_df)
        self.assertTrue('partition_id' in mesh_obj.el_df)
        self.assertTrue('partitions_neighbors' in mesh_obj.el_df)

        desired_list_1 = [2, 2, 2, 2]
        desired_list_2 = [1, 2, 1, 2]
        desired_list_3 = [(2,), (1,), (2,), (1,)]

        actual_list_1 = mesh_obj.el_df['no_of_mesh_partitions'].tolist()
        actual_list_2 = mesh_obj.el_df['partition_id'].tolist()
        actual_list_3 = mesh_obj.el_df['partitions_neighbors'].tolist()

        self.assertListEqual(actual_list_1, desired_list_1)
        self.assertListEqual(actual_list_2, desired_list_2)
        self.assertListEqual(actual_list_3, desired_list_3)

    def test_gmsh_parser_with_8_partitions(self):

        here = dirname(abspath(__file__))
        msh_filename = join(here, 'meshes', 'retangule_5_by_2_quad_par_8_irreg.msh')
        reader_obj = GmshAsciiMeshReader(msh_filename)
        converter = AmfeMeshConverter()
        reader_obj.parse(converter)
        mesh_obj = converter.return_mesh()

        self.assertTrue('no_of_mesh_partitions' in mesh_obj.el_df)
        self.assertTrue('partition_id' in mesh_obj.el_df)
        self.assertTrue('partitions_neighbors' in mesh_obj.el_df)

        actual_list_1 = mesh_obj.el_df['no_of_mesh_partitions'].tolist()
        actual_list_2 = mesh_obj.el_df['partition_id'].tolist()
        actual_list_3 = mesh_obj.el_df['partitions_neighbors'].tolist()

        here = dirname(abspath(__file__))
        desired_list_1 = load_object(join(here, 'pickle_obj', 'l1.pkl'))
        desired_list_2 = load_object(join(here, 'pickle_obj', 'l2.pkl'))
        desired_list_3 = load_object(join(here, 'pickle_obj', 'l3.pkl'))

        def helper(x):
            if x is None:
                return 0
            return x

        def helper2(x):
            if x is None:
                return ()
            elif type(x) == int:
                return (x,)
            return x

        desired_list_1 = [helper(val) for val in desired_list_1]
        desired_list_2 = [helper(val) for val in desired_list_2]
        desired_list_3 = [helper2(val) for val in desired_list_3]

        self.assertListEqual(actual_list_1, desired_list_1)
        self.assertListEqual(actual_list_2, desired_list_2)
        self.assertListEqual(actual_list_3, desired_list_3)


class PostProcessorTest(TestCase):
    def setUp(self):
        directory = join(dirname(abspath(__file__)), '.results')
        if os.path.exists(directory):
            clean_test_outputs(directory)

    def tearDown(self):
        pass

    def _create_fields(self, dim=3):
        amfemesh = create_amfe_obj()
        self.meshreader = AmfeMeshObjMeshReader(amfemesh)

        self.timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
        no_of_nodes = amfemesh.no_of_nodes
        no_of_cells = amfemesh.no_of_elements
        no_of_dofs = no_of_nodes * dim
        # q = np.random.rand(no_of_dofs * len(timesteps)).reshape(no_of_dofs, len(timesteps))
        q = np.ones((no_of_dofs, len(self.timesteps)))
        q[:, 0] = q[:, 0] * 0.0
        q[:, 1] = q[:, 1] * 0.1
        q[:, 2] = q[:, 2] * 0.2
        q[:, 3] = q[:, 3] * 0.3
        q2 = -q

        s = np.arange(no_of_cells * len(self.timesteps)).reshape(no_of_cells, len(self.timesteps))
        volume_indices = amfemesh.el_df[amfemesh.el_df['is_boundary'] == False].index.values

        self.fields_desired = {'Nodefield1': {'data_type': PostProcessDataType.VECTOR, 'timesteps': self.timesteps,
                                              'data': q, 'index': amfemesh.nodes_df.index.values,
                                              'mesh_entity_type': MeshEntityType.NODE},
                               'Nodefield2': {'data_type': PostProcessDataType.VECTOR, 'timesteps': self.timesteps,
                                              'data': q2, 'index': amfemesh.nodes_df.index.values,
                                              'mesh_entity_type': MeshEntityType.NODE},
                               'Elementfield1': {'data_type': PostProcessDataType.SCALAR, 'timesteps': self.timesteps,
                                                 'data': s, 'index': volume_indices,
                                                 'mesh_entity_type': MeshEntityType.ELEMENT}
                               }
        self.fields_no_of_nodes = no_of_nodes
        self.fields_no_of_timesteps = len(self.timesteps)

    def test_hdf5_postprocessor_writer_and_reader(self):
        self._create_fields()

        filename = join('.results', 'hdf5postprocessing.hdf5')

        if os.path.isfile(filename):
            os.remove(filename)

        writer = Hdf5PostProcessorWriter(self.meshreader, filename, '/myresults')
        fields = self.fields_desired
        for fieldname in fields:
            field = fields[fieldname]
            if field['data_type'] == PostProcessDataType.VECTOR:
                data = field['data'].reshape(self.fields_no_of_nodes, 3, self.fields_no_of_timesteps)
            else:
                data = field['data']
            writer.write_field(fieldname, field['data_type'], field['timesteps'],
                               data, field['index'], field['mesh_entity_type'])

        self._create_fields()

        h5filename = join('.results', 'hdf5postprocessing.hdf5')

        postprocessorreader = AmfeHdf5PostProcessorReader(h5filename,
                                                          meshrootpath='/mesh',
                                                          resultsrootpath='/myresults')
        meshreader = Hdf5MeshReader(h5filename, '/mesh')
        postprocessorwriter = DummyPostProcessorWriter(meshreader)
        postprocessorreader.parse(postprocessorwriter)
        fields = postprocessorwriter.return_result()
        # Check no of fields:
        self.assertEqual(len(fields.keys()), len(self.fields_desired.keys()))
        # Check each field:
        for fieldname in self.fields_desired:
            field_actual = fields[fieldname]
            field_desired = self.fields_desired[fieldname]
            assert_array_equal(field_actual['timesteps'], field_desired['timesteps'])
            assert_array_equal(field_actual['data_type'], field_desired['data_type'])
            assert_array_equal(field_actual['data'], field_desired['data'])
            assert_array_equal(field_actual['index'], field_desired['index'])
            assert_array_equal(field_actual['mesh_entity_type'], field_desired['mesh_entity_type'])

    def test_write_xdmf_from_hdf5(self):
        self._create_fields()
        filename = join('.results', 'hdf5postprocessing.hdf5')
        with h5py.File(filename, mode='r') as hdf5_fp:
            filename = join('.results', 'hdf5postprocessing.xdmf')
            with open(filename, 'wb') as xdmf_fp:
                fielddict = self.fields_desired
                for key in fielddict:
                    fielddict[key].update({'hdf5path': '/myresults/{}'.format(key)})
                    timesteps = fielddict[key]['timesteps']
                # timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
                write_xdmf_from_hdf5(xdmf_fp, hdf5_fp, '/mesh/nodes', '/mesh/topology', timesteps, fielddict)

    def test_amfe_solution_reader(self):
        self._create_fields(2)

        amfesolution = AmfeSolution()
        sol = self.fields_desired['Nodefield1']
        for t, q in zip(sol['timesteps'], sol['data'].T):
            amfesolution.write_timestep(t, q, q, q)

        mesh = create_amfe_obj()
        meshcomponent = StructuralComponent(mesh)
        # Set a material to get a mapping
        material = KirchhoffMaterial()
        meshcomponent.assign_material(material, 'Tri6', 'S', 'shape')

        postprocessorreader = AmfeSolutionReader(amfesolution, meshcomponent)

        meshreader = AmfeMeshObjMeshReader(mesh)
        postprocessorwriter = DummyPostProcessorWriter(meshreader)
        postprocessorreader.parse(postprocessorwriter)
        fields_actual = postprocessorwriter.return_result()

        field_desired = sol
        q = field_desired['data']
        dofs_x = meshcomponent.mapping.get_dofs_by_nodeids(meshcomponent.mesh.nodes_df.index.values, ('ux'))
        dofs_y = meshcomponent.mapping.get_dofs_by_nodeids(meshcomponent.mesh.nodes_df.index.values, ('uy'))
        q_x = q[dofs_x, :]
        q_y = q[dofs_y, :]
        data = np.empty((0, 3, 4), dtype=float)
        for node in meshcomponent.mesh.get_nodeidxs_by_all():
                data = np.concatenate((data, np.array([[q_x[node], q_y[node], np.zeros(q_x.shape[1])]])), axis=0)
        field_desired['data'] = data
        # Check no of fields:
        self.assertEqual(len(fields_actual.keys()), 3)
        # Check each field:
        field_displacement_actual = fields_actual['displacement']
        assert_array_equal(field_displacement_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_displacement_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_displacement_actual['data'], field_desired['data'])
        assert_array_equal(field_displacement_actual['index'], field_desired['index'])
        assert_array_equal(field_displacement_actual['mesh_entity_type'], field_desired['mesh_entity_type'])
        field_velocity_actual = fields_actual['velocity']
        assert_array_equal(field_velocity_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_velocity_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_velocity_actual['data'], field_desired['data'])
        assert_array_equal(field_velocity_actual['index'], field_desired['index'])
        assert_array_equal(field_velocity_actual['mesh_entity_type'], field_desired['mesh_entity_type'])
        field_acceleration_actual = fields_actual['acceleration']
        assert_array_equal(field_acceleration_actual['timesteps'], field_desired['timesteps'])
        assert_array_equal(field_acceleration_actual['data_type'], field_desired['data_type'])
        assert_array_equal(field_acceleration_actual['data'], field_desired['data'])
        assert_array_equal(field_acceleration_actual['index'], field_desired['index'])
        assert_array_equal(field_acceleration_actual['mesh_entity_type'], field_desired['mesh_entity_type'])


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj

# Example for testing one certain test:
# if __name__ == '__main__':
#   io_obj = IOTest()
#   io_obj.test_gmsh_parser_with_8_partitions()
