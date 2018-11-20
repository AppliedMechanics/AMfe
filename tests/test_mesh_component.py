# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from amfe.io.amfe_mesh_converter import AmfeMeshConverter
from amfe.component import StructuralComponent
from amfe.element import Tri3, Quad4, LineLinearBoundary


class TestMeshComponent(TestCase):
    def setUp(self):

        nodes = [(1, (0.0, 0.0)), (2, (1.0, 0.0)), (3, (1.0, 1.0)), (4, (0.0, 1.0)), (5, (2.0, 0.0)), (6, (2.0, 1.0))]
        elements = [(1, 'Tri3', (5, 6, 3)), (2, 'Tri3', (3, 2, 5)), (3, 'Quad4', (1, 2, 3, 4)),
                    (4, 'straight_line', (4, 1)), (5, 'straight_line', (5,6))]
        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5], 'nodes': [1, 2]}
                  }
        converter = AmfeMeshConverter()
        for node in nodes:
            converter.build_node(node[0], node[1][0], node[1][1], 0.0)
        for element in elements:
            converter.build_element(element[0], element[1], element[2])
        for group in groups:
            converter.build_group(group, groups[group]['nodes'], groups[group]['elements'])

        self.testmesh = converter.return_mesh()

        class DummyMaterial:
            def __init__(self, name):
                self.name = name

        self.mat1 = DummyMaterial('steel')
        self.mat2 = DummyMaterial('rubber')

    def tearDown(self):
        pass

    def test_assign_material_by_eleids(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        eleids2 = np.array([3], dtype=int)
        component.assign_material(self.mat1, eleids1, '_eleids')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        self.assertIsInstance(ele_obj_actual[0], Tri3)
        self.assertIsInstance(ele_obj_actual[1], Tri3)
        self.assertEqual(ele_obj_actual[2], None)
        # check materials
        self.assertEqual(ele_obj_actual[0].material.name, 'steel')
        self.assertEqual(ele_obj_actual[1].material.name, 'steel')

        # assign second material
        component.assign_material(self.mat2, eleids2, '_eleids')
        ele_obj_actual = component.ele_obj
        # check each object
        self.assertIsInstance(ele_obj_actual[0], Tri3)
        self.assertIsInstance(ele_obj_actual[1], Tri3)
        self.assertIsInstance(ele_obj_actual[2], Quad4)
        # check materials
        self.assertEqual(ele_obj_actual[0].material.name, 'steel')
        self.assertEqual(ele_obj_actual[1].material.name, 'steel')
        self.assertEqual(ele_obj_actual[2].material.name, 'rubber')

        component = StructuralComponent(self.testmesh)
        eleids3 = np.array([3, 1], dtype=int)
        component.assign_material(self.mat1, eleids3, '_eleids')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        self.assertIsInstance(ele_obj_actual[0], Tri3)
        self.assertIsInstance(ele_obj_actual[2], Quad4)
        self.assertEqual(ele_obj_actual[1], None)
        # check materials
        self.assertEqual(ele_obj_actual[0].material.name, 'steel')
        self.assertEqual(ele_obj_actual[2].material.name, 'steel')

    def test_assign_material_by_groups(self):
        # 2 Groups
        #
        component = StructuralComponent(self.testmesh)
        # assign materials
        component.assign_material(self.mat1, ['left', 'right'], '_groups')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        self.assertIsInstance(ele_obj_actual[0], Tri3)
        self.assertIsInstance(ele_obj_actual[1], Tri3)
        self.assertIsInstance(ele_obj_actual[2], Quad4)
        # check materials
        self.assertEqual(ele_obj_actual[0].material.name, 'steel')
        self.assertEqual(ele_obj_actual[1].material.name, 'steel')
        self.assertEqual(ele_obj_actual[2].material.name, 'steel')

        # 1 Group
        #
        component = StructuralComponent(self.testmesh)
        # assign materials
        component.assign_material(self.mat1, ['left'])
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        self.assertEqual(ele_obj_actual[0], None)
        self.assertEqual(ele_obj_actual[1], None)
        self.assertIsInstance(ele_obj_actual[2], Quad4)
        # check materials
        self.assertEqual(ele_obj_actual[2].material.name, 'steel')

    def test_assign_neumann_condition_by_eleidxs(self):
        component = StructuralComponent(self.testmesh)
        eleids = [4, 5]
        eleidxs = self.testmesh.get_elementidxs_by_elementids(eleids)
        time_func = lambda t: 3.0*t
        component.assign_neumann_condition(time_func, (1, 0), eleidxs, tag='_eleidxs', shadow_area=False,
                                           name='TestCondition')
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df =component._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['ele_obj', 'connectivity_idx']].values
        self.assertIsInstance(neumann_obj_array[0, 0], LineLinearBoundary)
        self.assertEqual(neumann_obj_array[0, 1], 3)
        self.assertIsInstance(neumann_obj_array[1, 0], LineLinearBoundary)
        self.assertEqual(neumann_obj_array[1, 1], 4)
        self.assertEqual(neumann_obj_array.shape, (2, 2))

        neumann_df_actual = component._neumann_df
        df_dict = {'name': {0: 'TestCondition'},
                   'tag': {0: '_eleidxs'},
                   'property_names': {0: np.array([3, 4], dtype=int)},
                   'function': {0: time_func},
                   'direction': {0: (1, 0)},
                   'shadow_area': {0: np.array(False, dtype=bool)}}
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

    def test_assign_neumann_condition_by_groups(self):
        component = StructuralComponent(self.testmesh)
        group = 'left_boundary'
        time_func = lambda t: 3.0*t
        component.assign_neumann_condition(time_func, (1, 0), [group], tag='_groups', shadow_area=False,
                                           name='TestCondition')
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df = component._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['ele_obj', 'connectivity_idx']].values
        self.assertIsInstance(neumann_obj_array[0, 0], LineLinearBoundary)
        self.assertEqual(neumann_obj_array[0, 1], 3)
        self.assertEqual(neumann_obj_array.shape, (1, 2))

        neumann_df_actual = component._neumann_df
        df_dict = {
            'name': {0: 'TestCondition'},
            'tag': {0: '_groups'},
            'property_names': {0: np.array(['left_boundary'])},
            'function': {0: time_func},
            'direction': {0: (1, 0)},
            'shadow_area': {0: np.array(False, dtype=bool)}
        }
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)
