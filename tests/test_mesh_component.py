# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
from pandas.core.indexing import IndexingError

from amfe.io.amfe_mesh_converter import AmfeMeshConverter
from amfe.component import StructuralComponent
from amfe.element import Tri3, Quad4, LineLinearBoundary
from amfe.constraint.constraint import DirichletConstraint


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
        converter.build_mesh_dimension(2)

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
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        ele_obj_df = component._ele_obj_df
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 1], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 2], Tri3)
        # check materials
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 1].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 2].material.name, 'steel')
        # check mapping
        nodal2global_desired = pd.DataFrame({'ux': {1: -1, 2: 6, 3: 4, 4: -1, 5: 0, 6: 2},
                                             'uy': {1: -1, 2: 7, 3: 5, 4: -1, 5: 1, 6: 3}})
        elements2global_desired = [np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([4, 5, 6, 7, 0, 1], dtype=int)]
        assert_frame_equal(component._mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component._mapping.elements2global, elements2global_desired):
            assert_array_equal(actual, desired)

        # assign second material
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')
        ele_obj_df = component._ele_obj_df
        # check each object
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 1], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 2], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 3], Quad4)
        # check materials
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 1].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 2].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 3].material.name, 'rubber')
        nodal2global_desired = pd.DataFrame({'ux': {1: 8, 2: 6, 3: 4, 4: 10, 5: 0, 6: 2},
                                             'uy': {1: 9, 2: 7, 3: 5, 4: 11, 5: 1, 6: 3}})
        elements2global_desired = [np.array([0, 1, 2, 3, 4, 5], dtype=int),
                                   np.array([4, 5, 6, 7, 0, 1], dtype=int),
                                   np.array([8,  9,  6,  7,  4,  5, 10, 11], dtype=int)]
        assert_frame_equal(component._mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component._mapping.elements2global, elements2global_desired):
            assert_array_equal(actual, desired)

        component = StructuralComponent(self.testmesh)
        eleids3 = np.array([3, 1], dtype=int)
        component.assign_material(self.mat1, eleids3, 'S', '_eleids')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        ele_obj_df = component._ele_obj_df
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 1], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 3], Quad4)
        with self.assertRaises(IndexingError):
            temp = ele_obj_df['ele_obj'].loc['S', 2]
        # check materials
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 1].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 3].material.name, 'steel')
        nodal2global_desired = pd.DataFrame({'ux': {1: 6, 2: 8, 3: 4, 4: 10, 5: 0, 6: 2},
                                             'uy': {1: 7, 2: 9, 3: 5, 4: 11, 5: 1, 6: 3}})
        elements2global_desired = [np.array([0, 1, 2, 3, 4, 5], dtype=int),
                                   np.array([6,  7,  8,  9,  4,  5, 10, 11], dtype=int)]
        assert_frame_equal(component._mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component._mapping.elements2global, elements2global_desired):
            assert_array_equal(actual, desired)

    def test_assign_material_by_groups(self):
        # 2 Groups
        #
        component = StructuralComponent(self.testmesh)
        # assign materials
        component.assign_material(self.mat1, ['left', 'right'], 'S', '_groups')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        ele_obj_df = component._ele_obj_df
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 1], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 2], Tri3)
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['S', 3], Quad4)
        # check materials
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 1].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 2].material.name, 'steel')
        self.assertEqual(ele_obj_df['ele_obj'].loc['S', 3].material.name, 'steel')

        # 1 Group
        #
        component = StructuralComponent(self.testmesh)
        # assign materials
        component.assign_material(self.mat1, ['left'], 'T')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        ele_obj_df = component._ele_obj_df
        with self.assertRaises(IndexingError):
            temp = ele_obj_df['ele_obj'].loc['T', 1]
        with self.assertRaises(IndexingError):
            temp = ele_obj_df['ele_obj'].loc['T', 2]
        self.assertIsInstance(ele_obj_df['ele_obj'].loc['T', 3], Quad4)
        # check materials
        self.assertEqual(ele_obj_df['ele_obj'].loc['T', 3].material.name, 'steel')
        # check if mapping is proceeded

    def test_assign_neumann_by_eleids(self):
        component = StructuralComponent(self.testmesh)
        eleids = [4, 5]
        time_func = lambda t: 3.0*t
        direction = (1, 0)
        condition = component._neumann.create_fixed_direction_neumann(direction, time_func)
        component.assign_neumann('TestCondition', condition, tag='_eleids', tag_values=eleids)
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df =component._neumann._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['neumann_obj', 'fk_mesh']].values
        self.assertIsInstance(neumann_obj_array[0, 0]._boundary_element, LineLinearBoundary)
        self.assertEqual(neumann_obj_array[0, 1], 4)
        self.assertIsInstance(neumann_obj_array[1, 0]._boundary_element, LineLinearBoundary)
        self.assertEqual(neumann_obj_array[1, 1], 5)
        self.assertEqual(neumann_obj_array.shape, (2, 2))

        neumann_df_actual = component._neumann._neumann_df
        df_dict = {'name': {0: 'TestCondition'},
                   'neumann_obj': {0: condition},
                   'property_names': {0: np.array([4, 5], dtype=int)},
                   'tag': {0: '_eleids'}}
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

    def test_assign_neumann_by_groups(self):
        component = StructuralComponent(self.testmesh)
        group = 'left_boundary'
        time_func = lambda t: 3.0*t
        direction = (1, 0)
        condition = component._neumann.create_fixed_direction_neumann(direction, time_func)
        component.assign_neumann('TestCondition', condition, [group])
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df = component._neumann._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['neumann_obj', 'fk_mesh']].values
        self.assertIsInstance(neumann_obj_array[0, 0]._boundary_element, LineLinearBoundary)
        self.assertEqual(neumann_obj_array[0, 1], 4)
        self.assertEqual(neumann_obj_array.shape, (1, 2))

        neumann_df_actual = component._neumann._neumann_df
        df_dict = {
            'name': {0: 'TestCondition'},
            'tag': {0: '_groups'},
            'property_names': {0: np.array(['left_boundary'])},
            'neumann_obj': {0: condition}
        }
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)
        
    def test_assign_constraints_by_group(self):
        component = StructuralComponent(self.testmesh)
        component.assign_material(self.mat1, np.array([1, 2, 3]), 'S', '_eleids')
        group = 'left_boundary'
        dirichlet = component._constraints.create_dirichlet_constraint()
        component.assign_constraint('TestConstraint', dirichlet, [group], '_groups', 'elim')
        constraint_df = component._constraints._constraints_df

        self.assertIsInstance(constraint_df['constraint_obj'].values[0], DirichletConstraint)
        self.assertEqual(constraint_df['name'].values[0], 'TestConstraint')
        self.assertEqual(constraint_df['strategy'].values[0], 'elim')
        assert_array_equal(constraint_df['dofids'].values[0], np.array([8, 9, 10, 11]))
        
    def test_assign_constraints_by_eleids(self):
        component = StructuralComponent(self.testmesh)
        component.assign_material(self.mat1, np.array([1, 2, 3]), 'S', '_eleids')
        component._update_mapping()
        dirichlet = component._constraints.create_dirichlet_constraint()
        component.assign_constraint('TestConstraint', dirichlet, [4], '_eleids', 'elim')
        constraint_df = component._constraints._constraints_df
        
        self.assertIsInstance(constraint_df['constraint_obj'].values[0], DirichletConstraint)
        self.assertEqual(constraint_df['name'].values[0], 'TestConstraint')
        self.assertEqual(constraint_df['strategy'].values[0], 'elim')
        assert_array_equal(constraint_df['dofids'].values[0], np.array([8, 9, 10, 11]))
