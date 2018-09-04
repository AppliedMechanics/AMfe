# -*- coding: utf-8 -*-
"""
Tests for testing io module
"""

from unittest import TestCase

from amfe.io import GidAsciiMeshReader, MeshConverter
from amfe import amfe_dir


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

    def build_material(self, material):
        self._groups.append(material)

    def build_partition(self, partition):
        self._partitions.append(partition)

    def build_mesh_dimension(self, dim):
        self._dimension = dim

    def return_mesh(self):
        return self


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
        reader = GidAsciiMeshReader(file, DummyMeshConverter())
        # Parse mesh
        mesh = reader.parse()
        # Check nodes
        for i, node in enumerate(nodes_desired):
            self.assertAlmostEqual(mesh._nodes[i], node)
        # Check elements
        for i, element in enumerate(elements_desired):
            self.assertEqual(mesh._elements[i], element)
        # Check mesh dimension
        self.assertEqual(mesh._dimension, dimension_desired)
