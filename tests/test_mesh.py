# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase, main
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_array_equal


class TestMesh(TestCase):
    def setUp(self):
        from amfe.mesh import Mesh
        self.testmesh = Mesh(dimension=2)
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]], dtype=np.float)
        connectivity = [np.array([5, 6, 3], dtype=np.int), np.array([3, 2, 5], dtype=np.int),
                        np.array([1, 2, 3, 4], dtype=np.int),
                        # boundary elements
                        np.array([4, 1], dtype=np.int), np.array([5, 6], dtype=np.int)]

        self._connectivity = connectivity

        data = {'shape': ['Tri3', 'Tri3', 'Quad4', 'straight_line', 'straight_line'],
                'is_boundary': [False, False, False, True, True],
                'connectivity': connectivity}
        indices = [1, 2, 3, 4, 5]
        el_df = pd.DataFrame(data, index=indices)

        x = nodes[:, 0]
        y = nodes[:, 1]
        nodeids = [1, 2, 3, 4, 5, 6]
        nodes_df = pd.DataFrame({'x': x, 'y': y}, index=nodeids)

        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5], 'nodes': [1, 2]}
                  }

        self.testmesh.nodes_df = nodes_df
        self.testmesh.groups = groups
        self.testmesh._el_df = el_df

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

    def test_connectivity(self):
        desireds = np.array(self._connectivity)
        actuals = self.testmesh.connectivity
        for actual, desired in zip(actuals, desireds):
            assert_array_equal(actual, desired)

    def test_get_connectivity_by_elementids(self):
        desireds = [np.array([5, 6, 3], dtype=int), np.array([4, 1], dtype=int)]
        for actual, desired in zip(self.testmesh.get_connectivity_by_elementids([1, 4]), desireds):
            assert_array_equal(actual, desired)

    def test_get_elementidxs_by_group(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right'])
        desired = np.array([0, 1], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementidxs_by_elementids(self):
        actual = self.testmesh.get_elementidxs_by_elementids([4, 1])
        desired = np.array([3, 0], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementids_by_elementidxs(self):
        actual = self.testmesh.get_elementids_by_elementidxs([3, 0])
        desired = np.array([4, 1], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementidxs_by_groups(self):
        actual = self.testmesh.get_elementidxs_by_groups(['right', 'left_boundary'])
        desired = np.array([0, 1, 3], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_elementids_by_groups(self):
        actual = self.testmesh.get_elementids_by_groups(['right', 'left_boundary'])
        desired = np.array([1, 2, 4], dtype=int)
        assert_array_equal(actual, desired)

    def test_get_nodeids_by_group(self):
        actual = self.testmesh.get_nodeids_by_groups(['left'])
        desired = np.array([0, 1, 2, 3], dtype=np.int)
        assert_equal(actual, desired)
        actual = set(self.testmesh.get_nodeids_by_groups(['right_boundary']))
        desired = set(np.array([4, 5, 0, 1]))
        assert_equal(actual, desired)

    def test_get_ele_shapes_by_idxs(self):
        actual = self.testmesh.get_ele_shapes_by_elementidxs([1, 4, 2])
        desired = np.array(['Tri3', 'straight_line', 'Quad4'], dtype=object)
        assert_equal(actual, desired)

    def test_get_ele_shapes_by_ids(self):
        actual = self.testmesh.get_ele_shapes_by_ids([2, 5, 3])
        desired = np.array(['Tri3', 'straight_line', 'Quad4'], dtype=object)
        assert_equal(actual, desired)

    def test_get_nodeidxs_by_all(self):
        actual = self.testmesh.get_nodeidxs_by_all()
        desired = np.array([0, 1, 2, 3, 4, 5], dtype=np.int)
        assert_equal(actual, desired)

    def test_get_nodeids_by_nodeidxs(self):
        actual = self.testmesh.get_nodeids_by_nodeidxs([3, 5, 2])
        desired = [4, 6, 3]
        assert_equal(actual, desired)
        actual = self.testmesh.get_nodeids_by_nodeidxs([3])
        desired = [4]
        assert_equal(actual, desired)

    def test_insert_tag(self):
        current_col_num = len(self.testmesh.el_df.columns)
        tag_to_add = 'partition_id'
        self.testmesh.insert_tag(tag_to_add)
        new_col_num = len(self.testmesh.el_df.columns)
        assert_equal(current_col_num + 1, new_col_num)
        self.assertTrue(tag_to_add in self.testmesh.el_df.columns)
        

    def test_remove_tag(self):
        tag_name = self.testmesh.el_df.columns[1]
        current_col_num = len(self.testmesh.el_df.columns)
        self.testmesh.remove_tag(tag_name)        
        new_col_num = len(self.testmesh.el_df.columns)
        assert_equal(current_col_num, new_col_num + 1)
        self.assertFalse(tag_name in self.testmesh.el_df.columns)
    
    def test_change_tag_values_by_dict(self):        
        desired_list_1 = [4,5]
        desired_list_2 = [1,2,3]
        tag_value_dict = {}
        tag_value_dict['False'] = desired_list_1
        tag_value_dict['True'] = desired_list_2
        self.testmesh.change_tag_values_by_dict('is_boundary',tag_value_dict)
        actual_list_1 = self.testmesh.el_df[self.testmesh.el_df['is_boundary'] == 'False'].index.tolist()
        actual_list_2 = self.testmesh.el_df[self.testmesh.el_df['is_boundary'] == 'True'].index.tolist()
        assert_equal(actual_list_1, desired_list_1)
        assert_equal(actual_list_2, desired_list_2)

    def test_replace_tag_values(self):
        current_key = 'Tri3' 
        new_key = 'Tri6'
        tag_name = 'shape'
        desired = [1, 2]
        self.testmesh.replace_tag_values(tag_name,current_key,new_key)
        actual = self.testmesh.el_df[self.testmesh.el_df[tag_name] == new_key].index.tolist()
        assert_equal(desired, actual)
        
    def test_get_elementids_by_tag(self):
        desired = [1,2]
        actual = self.testmesh.get_elementids_by_tag('shape','Tri3')
        assert_equal(desired, actual)

    def test_get_elementidxs_by_tag_value(self):
        desired = np.array([0, 1], dtype=int)
        actual = self.testmesh.get_elementidxs_by_tag('shape','Tri3')
        assert_equal(desired, actual)

if __name__ == '__main__':
    main()