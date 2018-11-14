# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


from amfe.mapping import Mapping


class TestMapping(TestCase):
    def setUp(self):
        self.no_of_nodes = 4
        # Two 2D Tri3 elements:
        self.connectivity = np.array([np.array([1, 2, 3], dtype=int), np.array([2, 4, 3], dtype=int)], dtype=object)
        self.fields = ('ux', 'uy', 'T')
        # Thus: 'ux' = 0, 'uy' = 1, 'T' = 2
        # First element has all fields
        # Second element only has displacement field

        self.nodeids = [1, 2, 3, 4]

        # self.dofs_by_node = ((0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1))
        self.dofs_by_element = [(('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'T'),
                                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'T'),
                                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'T')),
                                (('N', 0, 'ux'), ('N', 0, 'uy'),
                                ('N', 1, 'ux'), ('N', 1, 'uy'),
                                ('N', 2, 'ux'), ('N', 2, 'uy'))]

    def tearDown(self):
        pass

    def test_standard_mapping(self):
        mapping = Mapping(self.fields, self.nodeids, self.connectivity,
                          self.dofs_by_element)

        nodal2global_desired = pd.DataFrame({'ux': {1: 0, 2: 3, 3: 6, 4: 9}, 'uy': {1: 1, 2: 4, 3: 7, 4: 10},
                                             'T': {1: 2, 2: 5, 3: 8, 4: -1}})

        assert_frame_equal(mapping.nodal2global, nodal2global_desired)

        elements2global_desired = [np.array([0, 1, 2,
                                             3, 4, 5,
                                             6, 7, 8], dtype=int),
                                   np.array([3, 4,
                                             9, 10,
                                             6, 7], dtype=int)]
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
