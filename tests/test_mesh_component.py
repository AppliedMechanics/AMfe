# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal

from amfe.io.mesh.writer import AmfeMeshConverter
from amfe.component import StructuralComponent
from amfe.element import Tri3, Quad4, LineLinearBoundary
from amfe.assembly.assembly import Assembly
from amfe.constraint.constraint import DirichletConstraint


class TestMeshComponent(TestCase):
    def setUp(self):

        nodes = [(1, (0.0, 0.0)), (2, (1.0, 0.0)), (3, (1.0, 1.0)), (4, (0.0, 1.0)), (5, (2.0, 0.0)), (6, (2.0, 1.0))]
        elements = [(1, 'Tri3', (5, 6, 3)), (2, 'Tri3', (3, 2, 5)), (3, 'Quad4', (1, 2, 3, 4)),
                    (4, 'straight_line', (4, 1)), (5, 'straight_line', (5, 6))]
        groups = {'left': {'elements': [3], 'nodes': []},
                  'right': {'elements': [1, 2], 'nodes': [2, 3, 5, 6]},
                  'left_boundary': {'elements': [4], 'nodes': []},
                  'right_boundary': {'elements': [5], 'nodes': [1, 2]}
                  }
        tags = {'tag_boundaries': {1: [4], 2: [5]}}
        converter = AmfeMeshConverter()
        for node in nodes:
            converter.build_node(node[0], node[1][0], node[1][1], 0.0)
        for element in elements:
            converter.build_element(element[0], element[1], element[2])
        for group in groups:
            converter.build_group(group, groups[group]['nodes'], groups[group]['elements'])
        converter.build_tag(tags)
        converter.build_mesh_dimension(2)

        self.testmesh = converter.return_mesh()          

        class DummyMaterial:
            def __init__(self, name):
                self.name = name

        self.mat1 = DummyMaterial('steel')
        self.mat2 = DummyMaterial('rubber')

    def tearDown(self):
        pass

    def test_assembly_getter(self):
        component = StructuralComponent(self.testmesh)
        self.assertIsInstance(component.assembly, Assembly)

    def test_assembly_setter(self):
        component = StructuralComponent(self.testmesh)
        new_assembly = Assembly()
        component.assembly = new_assembly
        self.assertEqual(id(component.assembly), id(new_assembly))

    def test_no_of_elements(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        no_of_elements_actual = component.no_of_elements
        self.assertEqual(no_of_elements_actual, 2)

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
        mask1 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 1)
        mask2 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 2)
        self.assertEqual(len(ele_obj_df.loc[mask1, 'ele_obj'].values), 1)
        self.assertIsInstance(ele_obj_df.loc[mask1, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask2, 'ele_obj'].values[0], Tri3)
        # check materials
        self.assertEqual(ele_obj_df.loc[mask1, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask2, 'ele_obj'].values[0].material.name, 'steel')
        # check mapping
        nodal2global_desired = pd.DataFrame({'ux': {1: -1, 2: 6, 3: 4, 4: -1, 5: 0, 6: 2},
                                             'uy': {1: -1, 2: 7, 3: 5, 4: -1, 5: 1, 6: 3}})
        elements2global_desired = [np.array([0, 1, 2, 3, 4, 5], dtype=int), np.array([4, 5, 6, 7, 0, 1], dtype=int)]
        assert_frame_equal(component.mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component.mapping.elements2global, elements2global_desired):
            assert_array_equal(actual, desired)

        # assign second material
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')
        ele_obj_df = component._ele_obj_df
        # check each object
        mask1 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 1)
        mask2 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 2)
        mask3 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 3)
        self.assertIsInstance(ele_obj_df.loc[mask1, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask2, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask3, 'ele_obj'].values[0], Quad4)
        # check materials
        self.assertEqual(ele_obj_df.loc[mask1, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask2, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask3, 'ele_obj'].values[0].material.name, 'rubber')
        nodal2global_desired = pd.DataFrame({'ux': {1: 8, 2: 6, 3: 4, 4: 10, 5: 0, 6: 2},
                                             'uy': {1: 9, 2: 7, 3: 5, 4: 11, 5: 1, 6: 3}})
        elements2global_desired = [np.array([0, 1, 2, 3, 4, 5], dtype=int),
                                   np.array([4, 5, 6, 7, 0, 1], dtype=int),
                                   np.array([8,  9,  6,  7,  4,  5, 10, 11], dtype=int)]
        assert_frame_equal(component.mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component.mapping.elements2global, elements2global_desired):
            assert_array_equal(actual, desired)

        component = StructuralComponent(self.testmesh)
        eleids3 = np.array([3, 1], dtype=int)
        component.assign_material(self.mat1, eleids3, 'S', '_eleids')
        ele_obj_actual = component.ele_obj
        # check ele_obj is instance array
        self.assertIsInstance(ele_obj_actual, np.ndarray)
        # check each object
        ele_obj_df = component._ele_obj_df

        mask1 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 1)
        mask3 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 3)
        self.assertIsInstance(ele_obj_df.loc[mask1, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask3, 'ele_obj'].values[0], Quad4)
        with self.assertRaises(IndexError):
            mask2 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 2)
            temp = ele_obj_df.loc[mask2, 'ele_obj'].values[0]
        # check materials
        self.assertEqual(ele_obj_df.loc[mask1, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask3, 'ele_obj'].values[0].material.name, 'steel')
        nodal2global_desired = pd.DataFrame({'ux': {1: 0, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10},
                                             'uy': {1: 1, 2: 3, 3: 5, 4: 7, 5: 9, 6: 11}})
        elements2global_desired = [np.array([0,  1,  2,  3,  4,  5, 6, 7], dtype=int),
                                   np.array([8, 9, 10, 11, 4, 5], dtype=int)]
        assert_frame_equal(component.mapping.nodal2global, nodal2global_desired)
        for actual, desired in zip(component.mapping.elements2global, elements2global_desired):
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
        mask1 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 1)
        mask2 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 2)
        mask3 = (ele_obj_df['physics'] == 'S') & (ele_obj_df['fk_mesh'] == 3)
        self.assertIsInstance(ele_obj_df.loc[mask1, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask2, 'ele_obj'].values[0], Tri3)
        self.assertIsInstance(ele_obj_df.loc[mask3, 'ele_obj'].values[0], Quad4)

        # check materials
        # check materials
        self.assertEqual(ele_obj_df.loc[mask1, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask2, 'ele_obj'].values[0].material.name, 'steel')
        self.assertEqual(ele_obj_df.loc[mask3, 'ele_obj'].values[0].material.name, 'steel')

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
        with self.assertRaises(IndexError):
            mask1 = (ele_obj_df['physics'] == 'T') & (ele_obj_df['fk_mesh'] == 1)
            temp = ele_obj_df.loc[mask1, 'ele_obj'].values[0]
        with self.assertRaises(IndexError):
            mask2 = (ele_obj_df['physics'] == 'T') & (ele_obj_df['fk_mesh'] == 2)
            temp = ele_obj_df.loc[mask2, 'ele_obj'].values[0]
        mask3 = (ele_obj_df['physics'] == 'T') & (ele_obj_df['fk_mesh'] == 3)
        self.assertIsInstance(ele_obj_df.loc[mask3, 'ele_obj'].values[0], Quad4)
        # check materials
        self.assertEqual(ele_obj_df.loc[mask3, 'ele_obj'].values[0].material.name, 'steel')
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

    def test_assign_neumann_by_tags(self):
        component = StructuralComponent(self.testmesh)
        tagvalue = 1
        tagname = 'tag_boundaries'
        time_func = lambda t: 3.0*t
        direction = (1, 0)
        condition = component._neumann.create_fixed_direction_neumann(direction, time_func)
        component.assign_neumann('TestCondition', condition, [tagvalue], tagname)
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
            'tag': {0: tagname},
            'property_names': {0: np.array([1])},
            'neumann_obj': {0: condition}
        }
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

    def test_assign_neumann_by_groups_nonexistent_group(self):
        # Test, that nothing happens if group is wrong and switch is set True
        component = StructuralComponent(self.testmesh)
        group = 'wrong_group'
        time_func = lambda t: 3.0 * t
        direction = (1, 0)
        condition = component._neumann.create_fixed_direction_neumann(direction, time_func)
        component.assign_neumann('TestCondition', condition, [group], '_groups', True)
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df = component._neumann._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['neumann_obj', 'fk_mesh']].values
        self.assertEqual(neumann_obj_array.shape, (0, 2))

        neumann_df_actual = component._neumann._neumann_df
        neumann_df_desired = pd.DataFrame(columns=['name', 'tag', 'property_names', 'neumann_obj'])
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

        # Test, that error is thrown for a wrong group in the default case
        with self.assertRaises(ValueError): component.assign_neumann('TestCondition', condition, ['left_boundary',
                                                                                                  group])

        # Test in case of multiple passed groups, that only existing groups are handled and nothing happens for wrong
        # ones
        component = StructuralComponent(self.testmesh)
        time_func = lambda t: 3.0 * t
        direction = (1, 0)
        condition = component._neumann.create_fixed_direction_neumann(direction, time_func)
        component.assign_neumann('TestCondition', condition, ['right_boundary', 'wrong_group', 'left_boundary'],
                                 '_groups', True)
        # It must be set:
        #   - neumann_df
        #   - neumann_obj_df
        neumann_obj_df = component._neumann._neumann_obj_df
        neumann_obj_array = neumann_obj_df[['neumann_obj', 'fk_mesh']].values
        self.assertIsInstance(neumann_obj_array[0, 0]._boundary_element, LineLinearBoundary)
        self.assertEqual(neumann_obj_array[0, 1], 5)
        self.assertEqual(neumann_obj_array[1, 1], 4)
        self.assertEqual(neumann_obj_array.shape, (2, 2))

        neumann_df_actual = component._neumann._neumann_df
        df_dict = {
            'name': {0: 'TestCondition'},
            'tag': {0: '_groups'},
            'property_names': {0: np.array(['right_boundary', 'left_boundary'])},
            'neumann_obj': {0: condition}
        }
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

    def test_get_elementids_by_physics(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        eleids2 = np.array([3], dtype=int)
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')

        eleids_actual = component.get_elementids_by_physics('S')
        eleids_desired = np.array([1, 2, 3], dtype=int)

        assert_array_equal(eleids_actual, eleids_desired)

    def test_get_elementids_by_materials(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        eleids2 = np.array([3], dtype=int)
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')

        eleids_actual = component.get_elementids_by_materials(self.mat1)
        eleids_desired = np.array([1, 2], dtype=int)

        assert_array_equal(eleids_actual, eleids_desired)

    def test_get_physics(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        eleids2 = np.array([3], dtype=int)
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')

        physics_desired = 'S'
        physics_actual = component.get_physics()

        assert_array_equal(physics_desired, physics_actual)

    def test_get_materials(self):
        component = StructuralComponent(self.testmesh)
        eleids1 = np.array([1, 2], dtype=int)
        eleids2 = np.array([3], dtype=int)
        component.assign_material(self.mat1, eleids1, 'S', '_eleids')
        component.assign_material(self.mat2, eleids2, 'S', '_eleids')

        materials_desired = [self.mat1, self.mat2]
        materials_actual = component.get_materials()

        assert_array_equal(materials_desired, materials_actual)
