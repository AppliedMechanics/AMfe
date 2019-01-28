# -*- coding: utf-8 -*-
"""
Tests for testing io module
"""

from unittest import TestCase
import os
import numpy as np
import pandas as pd
import h5py
import pickle
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from amfe.component import StructuralComponent
from amfe.material import KirchhoffMaterial
from amfe.solver import AmfeSolution, AmfeSolutionHdf5
# Import I/O tools
from amfe.io.tools import check_dir, amfe_dir

# Import Mesh Reader
from amfe.io.mesh.reader import GidAsciiMeshReader, GidJsonMeshReader, GmshAsciiMeshReader, AmfeMeshObjMeshReader,\
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


class DummyMeshConverter(MeshConverter):
    def __init__(self):
        super().__init__()
        self._no_of_elements = None
        self._no_of_nodes = None
        self._nodes = []
        self._elements = []
        self._groups = []
        self._materials = []
        self._partitions = []
        self._dimension = None
        self._mesh = None
        self._tags = dict()

    def build_no_of_nodes(self, no):
        self._no_of_nodes = no

    def build_no_of_elements(self, no):
        self._no_of_elements = no

    def build_node(self, nodeid, x, y, z):
        self._nodes.append((nodeid, x, y, z))

    def build_element(self, eleid, etype, nodes):
        self._elements.append((eleid, etype, nodes))

    def build_group(self, name, nodeids=None, elementids=None):
        self._groups.append((name, nodeids, elementids))

    def build_mesh_dimension(self, dim):
        self._dimension = dim

    def build_tag(self, tag_dict):
        self._tags.update(tag_dict)

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


def create_amfe_obj():
    # Define input file path
    meshobj = Mesh(dimension=2)

    nodes = np.array([[1.345600000e-02, 3.561675700e-02],
                              [5.206839561e-01, 3.740820950e-02],
                              [3.851982918e-02, 5.460016703e-01],
                              [5.457667372e-01, 5.477935420e-01],
                              [1.027911912e+00, 3.919966200e-02],
                              [6.358365836e-02, 1.056386584e+00],
                              [1.040469476e+00, 5.445628213e-01],
                              [5.582746582e-01, 1.053154002e+00],
                              [1.052965658e+00, 1.049921420e+00],
                              [1.535139868e+00, 4.099111450e-02],
                              [1.547697432e+00, 5.463542738e-01],
                              [1.547656658e+00, 1.046688838e+00],
                              [2.042367825e+00, 4.278256700e-02],
                              [2.042357741e+00, 5.431194119e-01],
                              [2.042347658e+00, 1.043456257e+00]], dtype=float)

    connectivity = [np.array([13, 15,  9, 14, 12, 11], dtype=int),
                            np.array([9, 6, 5, 8, 4, 7], dtype=int),
                            np.array([9,  5, 13,  7,  10, 11], dtype=int),
                            np.array([1, 5, 6, 2, 4, 3], dtype=int),
                            np.array([5, 13,  10], dtype=int),
                            np.array([1, 5, 2], dtype=int),
                            np.array([6, 1, 3], dtype=int),
                            np.array([9, 6, 8], dtype=int),
                            np.array([13, 15, 14], dtype=int),
                            np.array([15,  9, 12], dtype=int)]

    data = {'shape': ['Tri6', 'Tri6', 'Tri6', 'Tri6', 'quadratic_line',
                      'quadratic_line', 'quadratic_line', 'quadratic_line',
                      'quadratic_line', 'quadratic_line'],
            'connectivity': connectivity,
            'is_boundary': [False, False, False, False, True, True, True, True, True, True]
            }
    indices = list(np.arange(1, 11))

    meshobj.el_df = pd.DataFrame(data, index=indices)

    meshobj.groups = {'left': {'nodes': [], 'elements': [2, 4]},
                      'right': {'nodes': [], 'elements': [1, 3]},
                      'left_boundary': {'nodes': [], 'elements': [7]},
                      'right_boundary': {'nodes': [], 'elements': [9]},
                      'top_boundary': {'nodes': [], 'elements': [8, 10]},
                      'left_dirichlet': {'nodes': [1, 3, 6], 'elements': []}}

    nodeids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    meshobj.nodes_df = pd.DataFrame({'x': nodes[:, 0], 'y': nodes[:, 1]}, index=nodeids)
    return meshobj


class IOTest(TestCase):
    def setUp(self):
        pass

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
        file = amfe_dir('tests/meshes/gid_ascii_4_tets.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidAsciiMeshReader(file)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh._nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh._elements[i], element)
        # Check mesh dimension
        self.assertEqual(mesh._dimension, dimension_desired)

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
        file = amfe_dir('tests/meshes/gid_json_4_tets.json')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GidJsonMeshReader(file)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()
        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh._nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh._elements[mesh._elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh._dimension, dimension_desired)
        self.assertEqual(mesh._groups, groups_desired)
        self.assertEqual(mesh._no_of_nodes, 15)
        self.assertEqual(mesh._no_of_elements, 10)

    def test_dummy_to_amfe(self):
        # Desired nodes
        nodes_input = [(1, 1.345600000e-02, 3.561675700e-02, 0.000000000e+00),
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
        # Desired elements
        # (internal name of Triangle Nnode 3 is 'Tri3')
        elements_input = [(1, 'Tri6', [13, 15, 9, 14, 12, 11]),
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
        groups_input = [
            ('left', [], [2, 4]),
            ('right', [], [1, 3]),
            ('left_boundary', [], [7]),
            ('right_boundary', [], [9]),
            ('top_boundary', [], [8, 10]),
            ('left_dirichlet', [1, 3, 6], [])
        ]

        converter = AmfeMeshConverter()
        # Build nodes
        for node in nodes_input:
            converter.build_node(node[0], node[1], node[2], node[3])
        for element in elements_input:
            converter.build_element(element[0], element[1], element[2])
        for group in groups_input:
            converter.build_group(group[0], group[1], group[2])
        mesh = converter.return_mesh()

        # CHECK NODES

        nodes_desired = np.array([[node[1], node[2]] for node in nodes_input])
        assert_allclose(mesh.nodes, nodes_desired)

        # CHECK CONNECTIVITIES
        # connectivity_desired = [np.array(element[2]) for element in elements_input[:]]
        for element in elements_input:
            assert_array_equal(mesh.get_connectivity_by_elementids([element[0]])[0], np.array(element[2], dtype=int))

        # CHECK DIMENSION
        self.assertEqual(mesh.dimension, 2)

        # CHECK NODE MAPPING
        for node in nodes_input:
            nodeid = node[0]
            node_actual = mesh.nodes_df.loc[nodeid]
            self.assertAlmostEqual(node_actual['x'], node[1])
            self.assertAlmostEqual(node_actual['y'], node[2])

        # CHECK ELESHAPES AND ELEMENTMAPPING IN DATAFRAME
        indices = list(np.arange(1, 11))
        data = {'shape': ['Tri6', 'Tri6', 'Tri6', 'Tri6', 'quadratic_line', 'quadratic_line',
                          'quadratic_line', 'quadratic_line', 'quadratic_line', 'quadratic_line'],
                'is_boundary': [False, False, False, False, True, True, True, True, True, True],
                'connectivity': [element[2] for element in elements_input]}
        el_df_desired = pd.DataFrame(data, index=indices)

        assert_frame_equal(mesh.el_df, el_df_desired)

        # CHECK GROUPS
        groups_desired = dict()
        for group in groups_input:
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

        self.tags_input = {'domain': {1: [2, 4], 2: [1, 3]}}

    def run_build_commands(self, converter):
        for node in self.nodes_input:
            converter.build_node(node[0], node[1], node[2], node[3])
        for element in self.elements_input:
            converter.build_element(element[0], element[1], element[2])
        for group in self.groups_input:
            converter.build_group(group[0], group[1], group[2])
        converter.build_tag(self.tags_input)
        converter.build_mesh_dimension(3)

    def test_dummy_to_hdf5_and_xdmf(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/hdf5_dummy')
        if not os.path.exists(amfe_dir('results/tests')):
            os.makedirs(amfe_dir('results/tests'))

        converter = Hdf5MeshConverter(filename + '.hdf5')
        self.run_build_commands(converter)
        converter.return_mesh()

        write_xdmf_mesh_from_hdf5(filename + '.xdmf', filename + '.hdf5', '/mesh')

    def test_hdf5_to_dummy(self):

        self.set_dummy_input()
        h5filename = amfe_dir('results/tests/hdf5_dummy.hdf5')

        meshreader = Hdf5MeshReader(h5filename, '/mesh')
        builder = DummyMeshConverter()
        meshreader.parse(builder)
        amfemesh = builder.return_mesh()

        # compare nodes
        for actual, desired in zip(amfemesh._nodes, self.nodes_input):
            self.assertEqual(actual, desired)
        # compare elements
        for actual, desired in zip(amfemesh._elements, self.elements_input):
            self.assertEqual(actual, desired)
        # compare tags
        self.assertEqual(amfemesh._tags.keys(), self.tags_input.keys())
        for key in self.tags_input:
            actual = amfemesh._tags[key]
            desired = self.tags_input[key]
            self.assertEqual(actual.keys(), desired.keys())
            for desired_key_value in desired.keys():
                self.assertEqual(len(actual[desired_key_value]), len(desired[desired_key_value]))
                actual_set = set(actual[desired_key_value])
                desired_set = set(desired[desired_key_value])
                self.assertEqual(actual_set, desired_set)

    def test_dummy_to_vtu(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/vtk_dummy.vtu')
        check_dir(filename)

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/vtk_dummy.vtk')
        check_dir(filename)

        converter = VtkMeshConverter(filename=filename)
        # Run build commands
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_wrong_fileextension(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/vtk_dummy.abc')
        check_dir(filename)

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_with_preallocation(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/vtk_dummy.vtk')
        check_dir(filename)

        converter = VtkMeshConverter(filename=filename)
        # Build nodes
        converter.build_no_of_nodes(len(self.nodes_input))
        converter.build_no_of_elements(len(self.elements_input))
        self.run_build_commands(converter)
        converter.return_mesh()

    def test_dummy_to_vtk_with_preallocation_too_late(self):
        self.set_dummy_input()

        filename = amfe_dir('results/tests/vtk_dummy.vtk')
        check_dir(filename)

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

        # Define input file path
        file = amfe_dir('tests/meshes/gmsh_ascii_8_tets.msh')
        # Define Reader Object, initialized with AmfeMeshConverter
        reader = GmshAsciiMeshReader(file)
        # Parse dummy mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()

        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh._nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh._elements[mesh._elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh._dimension, dimension_desired)
        self.assertEqual(mesh._groups, groups_desired)
        self.assertEqual(mesh._no_of_nodes, 8)
        self.assertEqual(mesh._no_of_elements, 10)

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

        meshobj = create_amfe_obj()

        # Define Reader Object, parse with AmfeMeshConverter
        reader = AmfeMeshObjMeshReader(meshobj)
        # Parse mesh
        dummy = DummyMeshConverter()
        reader.parse(dummy)
        mesh = dummy.return_mesh()
        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertEqual(mesh._nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh._elements[mesh._elements.index(element)], element)
        # Check mesh dimension
        self.assertEqual(mesh._dimension, dimension_desired)
        self.assertEqual(mesh._groups, groups_desired)
        self.assertEqual(mesh._no_of_nodes, 15)
        self.assertEqual(mesh._no_of_elements, 10)

    def test_gmsh_parser_with_2_partitions(self):

        msh_filename = amfe_dir('tests/meshes/2_partitions_2quad_mesh.msh')
        reader_obj = GmshAsciiMeshReader(msh_filename)
        converter = AmfeMeshConverter()
        reader_obj.parse(converter)
        mesh_obj = converter.return_mesh()

        self.assertTrue('no_of_mesh_partitions' in mesh_obj.el_df)
        self.assertTrue('partition_id' in mesh_obj.el_df)
        self.assertTrue('partitions_neighbors' in mesh_obj.el_df)
        
        desired_list_1 = [1,1,2,2]
        desired_list_2 = [1,1,1,2]
        desired_list_3 = [None,None,-2,-1]
        actual_list_1 = mesh_obj.el_df['no_of_mesh_partitions'].tolist()
        actual_list_2 = mesh_obj.el_df['partition_id'].tolist()
        actual_list_3 = mesh_obj.el_df['partitions_neighbors'].tolist()
        
        self.assertListEqual(actual_list_1, desired_list_1)
        self.assertListEqual(actual_list_2, desired_list_2)
        self.assertListEqual(actual_list_3, desired_list_3)
        
    def test_gmsh_parser_with_8_partitions(self):

        msh_filename = amfe_dir('tests/meshes/retangule_5_by_2_quad_par_8_irreg.msh')
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

        desired_list_1 = load_object(amfe_dir('tests/pickle_obj/l1.pkl'))
        desired_list_2 = load_object(amfe_dir('tests/pickle_obj/l2.pkl'))
        desired_list_3 = load_object(amfe_dir('tests/pickle_obj/l3.pkl'))
        
        self.assertListEqual(actual_list_1, desired_list_1)
        self.assertListEqual(actual_list_2, desired_list_2)
        self.assertListEqual(actual_list_3, desired_list_3)


class PostProcessorTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _create_fields(self):
        amfemesh = create_amfe_obj()
        self.meshreader = AmfeMeshObjMeshReader(amfemesh)

        self.timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
        no_of_nodes = amfemesh.no_of_nodes
        no_of_cells = amfemesh.no_of_elements
        dim = 3
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

    def test_hdf5_postprocessor_writer_and_reader(self):
        self._create_fields()

        filename = amfe_dir('results/tests/hdf5postprocessing.hdf5')

        writer = Hdf5PostProcessorWriter(self.meshreader, filename, '/myresults')
        fields = self.fields_desired
        for fieldname in fields:
            field = fields[fieldname]
            writer.write_field(fieldname, field['data_type'], field['timesteps'], field['data'], field['index'],
                               field['mesh_entity_type'])

        self._create_fields()

        h5filename = amfe_dir('results/tests/hdf5postprocessing.hdf5')

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
        filename = amfe_dir('results/tests/hdf5postprocessing.hdf5')
        with h5py.File(filename, mode='r') as hdf5_fp:
            filename = amfe_dir('results/tests/hdf5postprocessing.xdmf')
            with open(filename, 'wb') as xdmf_fp:
                fielddict = self.fields_desired
                for key in fielddict:
                    fielddict[key].update({'hdf5path': '/myresults/{}'.format(key)})
                    timesteps = fielddict[key]['timesteps']
                # timesteps = np.arange(0, 0.8, 0.2)  # 4 timesteps
                write_xdmf_from_hdf5(xdmf_fp, hdf5_fp, '/mesh/nodes', '/mesh/topology', timesteps, fielddict)

    def test_amfe_solution_reader(self):
        self._create_fields()

        amfesolution = AmfeSolution()
        sol = self.fields_desired['Nodefield1']
        for t, q in zip(sol['timesteps'], sol['data'].T):
            amfesolution.write_timestep(t, q)

        mesh = create_amfe_obj()
        meshcomponent = StructuralComponent(mesh)
        # Set a material to get a mapping
        material = KirchhoffMaterial()
        meshcomponent.assign_material(material, 'Tri6', 'S', 'shape')


        postprocessorreader = AmfeSolutionReader(amfesolution, meshcomponent, False)

        #meshreader = AmfeMeshObjMeshReader(mesh)
        #postprocessorwriter = DummyPostProcessorWriter(meshreader)
        # postprocessorreader.parse(postprocessorwriter)
        # fields = postprocessorwriter.return_result()
        # Check no of fields:
#        self.assertEqual(len(fields.keys()), len(self.fields_desired.keys()))
        # Check each field:
        #for fieldname in self.fields_desired:
#            field_actual = fields[fieldname]
#            field_desired = self.fields_desired[fieldname]
#            assert_array_equal(field_actual['timesteps'], field_desired['timesteps'])
#            assert_array_equal(field_actual['data_type'], field_desired['data_type'])
#            assert_array_equal(field_actual['data'], field_desired['data'])
#            assert_array_equal(field_actual['index'], field_desired['index'])
#            assert_array_equal(field_actual['mesh_entity_type'], field_desired['mesh_entity_type'])


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
