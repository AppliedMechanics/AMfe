# -*- coding: utf-8 -*-
'''
Test for checking Neumann conditions.
'''

from unittest import TestCase
import numpy as np
import scipy as sp
import pandas as pd
from pandas.testing import assert_frame_equal

from numpy.testing import assert_allclose, assert_almost_equal
from amfe.element import Tri3Boundary, Tri6Boundary, Quad4Boundary, Quad8Boundary, LineLinearBoundary
from amfe.element.boundary_element import BoundaryElement
from amfe.neumann import *


class DummyBoundary(BoundaryElement):
    def __init__(self):
        pass

    def f_mat(self, X, u):
        f_mat = np.array([[0, -1/3], [0, -1/3]])
        return f_mat


class NeumannTest(TestCase):
    def setUp(self):                
        self.test_boundary = DummyBoundary()
        self.test_direct = np.array([1, -1])
        self.time_func = lambda t: 2
        
    def tearDown(self):
        pass
    
    def test_fixed_direction_neumann(self):
        X = None
        u = None
        t = 0.0
        neumann = FixedDirectionNeumann(self.test_direct, self.time_func)
        neumann._boundary_element = self.test_boundary
        f_ext_actual = neumann.f_ext(X, u, t)
        desired_f = np.array([2/3, -2/3, 2/3, -2/3])
        np.testing.assert_allclose(f_ext_actual, desired_f, rtol=1E-6, atol=1E-7)
        
    def test_normal_following_neumann(self):
        X = None
        u = None
        t = 0.0
        neumann = NormalFollowingNeumann(self.time_func)
        neumann._boundary_element = self.test_boundary
        f_proj = neumann._f_proj(self.test_boundary.f_mat(X, u))
        f_ext_actual = neumann.f_ext(X, u, t)
        desired_f = np.array([0, -2/3, 0, -2/3])
        np.testing.assert_allclose(f_ext_actual, desired_f, rtol=1E-6, atol=1E-7)
        
    def test_projected_neumann(self):
        X = None
        u = None
        t = 0.0
        neumann = ProjectedAreaNeumann(self.test_direct, self.time_func)
        neumann._boundary_element = self.test_boundary
        f_ext_actual = neumann.f_ext(X, u, t)
        desired_f = np.array([0.47140452, -0.47140452, 0.47140452, -0.47140452])
        np.testing.assert_allclose(f_ext_actual, desired_f, rtol=1E-6, atol=1E-7)
        

class TestNeumannManager(TestCase):
    def setUp(self):
        self.neumann_man = NeumannManager()
        self.test_boundary = DummyBoundary()
        self.test_direct = np.array([1, -1])
        self.time_func = lambda t: 2
        
    def test_create_neumann(self):
        neumann = self.neumann_man.create_fixed_direction_neumann(direction=self.test_direct)
        self.assertIsInstance(neumann, FixedDirectionNeumann)
        neumann = self.neumann_man.create_normal_following_neumann()
        self.assertIsInstance(neumann, NormalFollowingNeumann)
        neumann = self.neumann_man.create_projected_area_neumann(direction=self.test_direct)
        self.assertIsInstance(neumann, ProjectedAreaNeumann)
    
    def test_assign_neumann_by_eleids(self):
        eleids = [2, 7]
        ele_shapes = ['Tri3', 'Quad4']
        time_func = lambda t: 3.0*t

        neumannbc = self.neumann_man.create_fixed_direction_neumann((1, 0), time_func)
        self.neumann_man.assign_neumann_by_eleids(neumannbc, eleids, ele_shapes, tag='_eleids',
                                                  property_names=eleids, name='TestCondition')
        neumann_obj_df = self.neumann_man.el_df
        neumann_obj_array = neumann_obj_df[['neumann_obj', 'fk_mesh']].values
        self.assertIsInstance(neumann_obj_array[0, 0], FixedDirectionNeumann)
        self.assertIsInstance(neumann_obj_array[0, 0]._boundary_element, Tri3Boundary)
        self.assertEqual(neumann_obj_array[0, 1], 2)
        self.assertIsInstance(neumann_obj_array[1, 0], FixedDirectionNeumann)
        self.assertIsInstance(neumann_obj_array[1, 0]._boundary_element, Quad4Boundary)
        self.assertEqual(neumann_obj_array[1, 1], 7)
        self.assertEqual(neumann_obj_array.shape, (2, 2))

        neumann_df_actual = self.neumann_man._neumann_df
        df_dict = {'name': {0: 'TestCondition'},
                   'tag': {0: '_eleids'},
                   'property_names': {0: np.array([2, 7], dtype=int)},
                   'neumann_obj': neumannbc}
        neumann_df_desired = pd.DataFrame.from_dict(df_dict)
        assert_frame_equal(neumann_df_actual, neumann_df_desired, check_like=True)

    def test_write_mapping(self):
        eleids = [2, 7]
        ele_shapes = ['Tri3', 'Quad4']
        time_func = lambda t: 3.0 * t

        neumannbc = self.neumann_man.create_fixed_direction_neumann((1, 0), time_func)
        self.neumann_man.assign_neumann_by_eleids(neumannbc, eleids, ele_shapes, tag='_eleids',
                                                  property_names=eleids, name='TestCondition')
        neumann_obj_df = self.neumann_man.el_df
        fk = 100
        local_id = neumann_obj_df.index.get_values()[0]
        self.neumann_man.write_mapping_key(fk, local_id)
        actual = self.neumann_man.el_df.loc[local_id, 'fk_mapping']
        self.assertEqual(actual, fk)
