# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from amfe.mapping import Mapping


class TestMapping(TestCase):
    def setUp(self):
        self.no_of_nodes = 4
        # Two 2D Tri3 elements:
        self.connectivity = np.array([np.array([0, 1, 2], dtype=int), np.array([1, 3, 2], dtype=int)], dtype=object)
        self.fields = ('ux', 'uy', 'T')
        # Thus: 'ux' = 0, 'uy' = 1, 'T' = 2
        # First element has all fields
        # Second element only has displacement field

        self.dofs_by_node = ((0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1))
        self.dofs_by_element = [((0, 1, 2), (0, 1, 2), (0, 1, 2)), ((0, 1), (0, 1), (0, 1))]

    def tearDown(self):
        pass

    def test_field2idx(self):
        mapping = Mapping(self.fields, self.no_of_nodes, self.connectivity,
                          self.dofs_by_node, self.dofs_by_element)
        field2idx_desired = {'ux': 0, 'uy': 1, 'T': 2}
        self.assertEqual(mapping.field2idx, field2idx_desired)

    def test_standard_mapping(self):
        mapping = Mapping(self.fields, self.no_of_nodes, self.connectivity,
                          self.dofs_by_node, self.dofs_by_element)
        nodes2global_desired = np.array([[0, 1, 2],
                                         [3, 4, 5],
                                         [6, 7, 8],
                                         [9, 10, -1]], dtype=int)
        assert_array_equal(mapping.nodes2global, nodes2global_desired)

        elements2global_desired = [np.array([[0, 1, 2],
                                             [3, 4, 5],
                                             [6, 7, 8]], dtype=int),
                                   np.array([[3, 4],
                                             [9, 10],
                                             [6, 7]], dtype=int)]
        for element_actual, element_desired in zip(mapping.elements2global, elements2global_desired):
            assert_array_equal(element_actual, element_desired)

    def test_get_global_by_local(self):
        mapping = Mapping(self.fields, self.no_of_nodes, self.connectivity,
                          self.dofs_by_node, self.dofs_by_element)
        mapping.nodes2global = np.array([[0, 1, 2],
                                         [3, 4, 5],
                                         [60, 70, 80],
                                         [9, 10, -1]], dtype=int)
        global_desired = 70
        self.assertEqual(mapping.get_dof_by_nodeidx(2, 'uy'), global_desired)
        global_desired = -1
        self.assertEqual(mapping.get_dof_by_nodeidx(3, 'T'), global_desired)

    def test_get_dofs_by_nodeidxs(self):
        mapping = Mapping(self.fields, self.no_of_nodes, self.connectivity,
                          self.dofs_by_node, self.dofs_by_element)
        mapping.nodes2global = np.array([[0, 1, 2],
                                         [3, 4, 5],
                                         [60, 70, 80],
                                         [9, 10, -1]], dtype=int)
        nodeidxs = (0, 2, 3)
        fields = ('ux', 'T')
        actual = mapping.get_dofs_by_nodeidxs(nodeidxs, fields)
        desired = np.array([0, 2, 60, 80, 9, -1])
        assert_array_equal(actual, desired)
