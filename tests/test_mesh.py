# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_equal


class TestMesh(TestCase):
    def setUp(self):
        from amfe import Mesh
        self.testmesh = Mesh(dimension=2)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        connectivity = [np.array([4, 5, 2], dtype=np.int), np.array([2, 1, 4], dtype=np.int),
                        np.array([0, 1, 2, 3], dtype=np.int)]
        boundary_connectivity = [np.array([3, 0], dtype=np.int), np.array([4, 5], dtype=np.int)]
        eleid2idx = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0), 5: (1, 1)}
        nodeid2idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        ele_shapes = ['Tri3', 'Tri3', 'Quad4']
        boundary_ele_shapes = ['straight_line', 'straight_line']
        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5], 'nodes': [1, 2]}
                  }

        self.testmesh.nodes = nodes
        self.testmesh.connectivity = connectivity
        self.testmesh.boundary_connectivity = boundary_connectivity
        self.testmesh.ele_shapes = ele_shapes
        self.testmesh.boundary_ele_shapes = boundary_ele_shapes
        self.testmesh.nodeid2idx = nodeid2idx
        self.testmesh.eleid2idx = eleid2idx
        self.testmesh.groups = groups

    def tearDown(self):
        pass

    def test_no_of_properties(self):
        self.assertEqual(self.testmesh.no_of_nodes, 6)
        self.assertEqual(self.testmesh.no_of_elements, 3)
        self.assertEqual(self.testmesh.no_of_boundary_elements, 2)
        self.assertEqual(self.testmesh.dimension, 2)

    def test_nodes_voigt(self):
        desired = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0], dtype=np.float)
        assert_equal(self.testmesh.nodes_voigt, desired)

    def test_get_elementidxs_by_group(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right'])
        desired = [(0, 0), (0, 1)]
        self.assertEqual(actual, desired)

    def test_get_elementidxs_by_groups(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right', 'left_boundary'])
        desired = [(0, 0), (0, 1), (1, 0)]
        self.assertEqual(actual, desired)

    def test_get_nodeidxs_by_group(self):
        actual = self.testmesh.get_nodeidxs_by_groups(['left'])
        desired = np.array([0, 1, 2, 3], dtype=np.int)
        assert_equal(actual, desired)
        actual = set(self.testmesh.get_nodeidxs_by_groups(['right_boundary']))
        desired = set(np.array([4, 5, 0, 1]))
        assert_equal(actual, desired)

    def test_get_nodeidxs_by_groups(self):
        actual = self.testmesh.get_nodeidxs_by_groups(['left', 'right_boundary'])
        desired = np.array([0, 1, 2, 3, 4, 5])
        assert_equal(actual, desired)

    def test_get_ele_shapes_by_idxs(self):
        actual = self.testmesh.get_ele_shapes_by_idxs([(0, 1), (1, 1), (0, 2)])
        desired = ['Tri3', 'straight_line', 'Quad4']
        self.assertEqual(actual, desired)

    def test_get_nodeidxs_by_all(self):
        actual = self.testmesh.get_nodeidxs_by_all()
        desired = np.array([0, 1, 2, 3, 4, 5], dtype=np.int)
        assert_equal(actual, desired)
