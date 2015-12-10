# -*- coding: utf-8 -*-
'''
Test for checking the stiffness matrices. 
'''

import unittest
import numpy as np
import scipy as sp
import sys
sys.path.insert(0,'..')



from amfe import Tri3, Tri6, Quad4, Quad8, Tet4, Tet10
from amfe import material

def jacobian(func, X, u):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme. 
    
    '''
    ndof = X.shape[0]
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(X, u).copy()
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp)
        jac[:,i] = (f_tmp - f) / h
    return jac


class ElementTest(unittest.TestCase):
    
    def initialize_element(self, element, no_of_dofs):
        self.X = sp.rand(no_of_dofs)
        self.u = sp.rand(no_of_dofs)
        self.my_material = material.KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
        self.my_element = element(self.my_material)

    def jacobi_test_element(self, rtol=1E-4, atol=1E-6):
        K, f = self.my_element.k_and_f_int(self.X, self.u)
        K_finite_diff = jacobian(self.my_element.f_int, self.X, self.u)
        np.testing.assert_allclose(K, K_finite_diff, rtol=rtol, atol=atol)
        

class Tri3Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tri3, 6)
        
    def test_jacobi(self):
        self.jacobi_test_element()
    
    def test_mass(self):
        X = np.array([0,0,3,1,2,2.])
        u = np.zeros(6)
        M = self.my_element.m_int(X, u)
        np.testing.assert_almost_equal(np.sum(M), 4)
        
class Tri6Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tri6, 12)
    
    def test_jacobi(self):
        self.jacobi_test_element(rtol=1E-4, atol=1E-6)

class Quad4Test(ElementTest):
    def setUp(self):
        self.initialize_element(Quad4, 8)
    
    def test_jacobi(self):
        self.jacobi_test_element()


class Quad8Test(ElementTest):
    def setUp(self):
        self.initialize_element(Quad8, 16)
    
    def test_jacobi(self):
        self.jacobi_test_element(rtol=1E-4, atol=1E-6)

class Tet4Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tet4, 4*3)
    
    def test_jacobi(self):
        self.jacobi_test_element()

class Tet10Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tet10, 10*3)
    
    def test_jacobi(self):
        self.jacobi_test_element(rtol=1E-3, atol=1E-5)

    

if __name__ == '__main__':
    unittest.main()