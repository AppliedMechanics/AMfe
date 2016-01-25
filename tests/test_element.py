# -*- coding: utf-8 -*-
'''
Test for checking the stiffness matrices.
'''

import unittest
import numpy as np
import scipy as sp
import nose

from numpy.testing import assert_allclose, assert_almost_equal
import amfe
from amfe import Tri3, Tri6, Quad4, Quad8, Tet4, Tet10
from amfe import material

def jacobian(func, X, u, t):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme.

    '''
    ndof = X.shape[0]
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(X, u, t).copy()
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(X, u_tmp, t)
        jac[:,i] = (f_tmp - f) / h
    return jac

X_tri3 = np.array([0,0,3,1,2,2], dtype=float)
X_tri6 = np.array([0,0,3,1,2,2,1.5,0.5,2.5,1.5,1,1], dtype=float)
X_quad4 = np.array([0,0,1,0,1,1,0,1], dtype=float)
X_quad8 = np.array([0,0,1,0,1,1,0,1,0.5,0,1,0.5,0.5,1,0,0.5], dtype=float)
X_tet4 = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1], dtype=float)
X_tet10 = np.array([0.,  0.,  0.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  2.,  1.,
                    0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
                    1.,  0.,  1.,  1.])


class ElementTest(unittest.TestCase):
    '''Base class for testing the elements with the jacobian'''
    def initialize_element(self, element, X_def):
        no_of_dofs = len(X_def)
        self.X = X_def + 0.5*sp.rand(no_of_dofs)
        self.u = sp.rand(no_of_dofs)
        self.my_material = material.KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
        self.my_element = element(self.my_material)

    @nose.tools.nottest
    def jacobi_test_element(self, rtol=2E-4, atol=1E-6):
        K, f = self.my_element.k_and_f_int(self.X, self.u, t=0)
        K_finite_diff = jacobian(self.my_element.f_int, self.X, self.u, t=0)
        np.testing.assert_allclose(K, K_finite_diff, rtol=rtol, atol=atol)
    
    @nose.tools.nottest
    def check_python_vs_fortran(self):
        # python routine
        self.my_element._compute_tensors_python(self.X, self.u, t=0)
        K = self.my_element.K.copy()
        f = self.my_element.f.copy()
        # fortran routine
        self.my_element._compute_tensors(self.X, self.u, t=0)
        assert_almost_equal(self.my_element.K, K)
        assert_almost_equal(self.my_element.f, f)



class Tri3Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tri3, X_tri3)

    def test_jacobi(self):
        self.jacobi_test_element()

    def test_mass(self):
        X = np.array([0,0,3,1,2,2.])
        u = np.zeros(6)
        M = self.my_element.m_int(X, u)
        np.testing.assert_almost_equal(np.sum(M), 4)

class Tri6Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tri6, X_tri6)

    def test_jacobi(self):
        self.jacobi_test_element(rtol=1E-3)

class Quad4Test(ElementTest):
    def setUp(self):
        self.initialize_element(Quad4, X_quad4)

    def test_jacobi(self):
        self.jacobi_test_element()


class Quad8Test(ElementTest):
    def setUp(self):
        self.initialize_element(Quad8, X_quad8)

    def test_jacobi(self):
        self.jacobi_test_element(rtol=1E-3)

class Tet4Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tet4, X_tet4)

    def test_jacobi(self):
        self.jacobi_test_element()

class Tet10Test(ElementTest):
    def setUp(self):
        self.initialize_element(Tet10, X_tet10)

    def test_jacobi(self):
        self.jacobi_test_element(rtol=2E-3)

#%%
# Test the material consistency:
class MaterialTest3D(ElementTest):
    '''
    Test the material using the Tet4 Element and different materials;
    Perform a jacobian check to find out, if any inconsistencies are apparent.
    '''
    def setUp(self):
        self.initialize_element(Tet4, X_tet4)
        self.u *= 0.1

    def test_Mooney(self):
        A10, A01, kappa, rho = sp.rand(4)*1E3 + 100
        print('Material parameters A10, A01 and kappa:', A10, A01, kappa)
        my_material = material.MooneyRivlin(A10, A01, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element(rtol=1E-3)

    def test_Neo(self):
        mu, kappa, rho = sp.rand(3)*1E3 + 100
        print('Material parameters mu, kappa:', mu, kappa)
#        mu /= 4
        my_material = material.NeoHookean(mu, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element(rtol=5E-4)

class MaterialTest2D(ElementTest):
    '''
    Test the material using the Tri3 Element and different materials;
    Perform a jacobian check to find out, if any inconsistencies are apparent.
    '''
    def setUp(self):
        self.initialize_element(Tri3, X_tri3)
        self.u *= 0.1

    def test_Mooney(self):
        A10, A01, kappa, rho = sp.rand(4)*1E3 + 100
        my_material = material.MooneyRivlin(A10, A01, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element(rtol=5E-4)

    def test_Neo(self):
        mu, kappa, rho = sp.rand(3)*1E3 + 100
#        mu /= 4
        kappa *= 100
        my_material = material.NeoHookean(mu, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element()



class MaterialTest(unittest.TestCase):
    def setUp(self):
        mu, kappa, rho = sp.rand(3)
        A10 = mu/2
        A01 = 0
        F = sp.rand(3,3)
        self.E = 1/2*(F.T @ F - sp.eye(3))
        self.mooney = material.MooneyRivlin(A10, A01, kappa, rho)
        self.neo = material.NeoHookean(mu, kappa, rho)

    def test_Neo_vs_Mooney_S(self):
        S_mooney, Sv_mooney, C_mooney = self.mooney.S_Sv_and_C(self.E)
        S_neo, Sv_neo, C_neo = self.neo.S_Sv_and_C(self.E)
        np.testing.assert_allclose(S_mooney, S_neo)

    def test_Neo_vs_Mooney_Sv(self):
        S_mooney, Sv_mooney, C_mooney = self.mooney.S_Sv_and_C(self.E)
        S_neo, Sv_neo, C_neo = self.neo.S_Sv_and_C(self.E)
        np.testing.assert_allclose(Sv_mooney, Sv_neo)

    def test_Neo_vs_Mooney_C(self):
        S_mooney, Sv_mooney, C_mooney = self.mooney.S_Sv_and_C(self.E)
        S_neo, Sv_neo, C_neo = self.neo.S_Sv_and_C(self.E)
        np.testing.assert_allclose(C_mooney, C_neo)


class MaterialTest2dPlaneStress(unittest.TestCase):
    def setUp(self):
        F = sp.zeros((3,3))
        F[:2,:2] = sp.rand(2,2)
        F[2,2] = 1
        self.E = 1/2*(F.T @ F - sp.eye(3))
        A10, A01, kappa, rho = sp.rand(4)
        mu = A10*2
        self.mooney = material.MooneyRivlin(A10, A01, kappa, rho, plane_stress=False)
        self.neo = material.NeoHookean(mu, kappa, rho, plane_stress=False)

    def test_mooney_2d(self):
        E = self.E
        S, S_v, C = self.mooney.S_Sv_and_C(E)
        S2d, S_v2d, C2d = self.mooney.S_Sv_and_C_2d(E[:2, :2])
        np.testing.assert_allclose(S[:2, :2], S2d)
        np.testing.assert_allclose(C[np.ix_([0,1,-1], [0,1,-1])], C2d)


    def test_neo_2d(self):
        E = self.E
        S, S_v, C = self.neo.S_Sv_and_C(E)
        S2d, S_v2d, C2d = self.neo.S_Sv_and_C_2d(E[:2, :2])
        np.testing.assert_allclose(S[:2, :2], S2d)
        np.testing.assert_allclose(C[np.ix_([0,1,-1], [0,1,-1])], C2d)


#%%
def test_tri3_pressure():
    X = np.array([0,0,0,1,0,0,0,1,0])
    u = np.zeros_like(X)
    my_press_ele = amfe.Tri3Boundary(1, 'normal')
    K, f = my_press_ele.k_and_f_int(X, u)
    np.testing.assert_array_equal( K, np.zeros((9,9)))
    np.testing.assert_allclose(np.sum(f), 1/2)

def test_line_pressure():
    X = np.array([0,0,1,1])
    u = X
    my_press_ele = amfe.LineLinearBoundary(1, 'normal')
    K, f = my_press_ele.k_and_f_int(X, u)
    np.testing.assert_array_equal( K, np.zeros((4,4)))
    np.testing.assert_allclose(f, np.array([1,-1,1,-1]))

def test_line_pressure2():
    X = np.array([0,0,1,1])
    u = X
    my_press_ele = amfe.LineLinearBoundary(1, 'x')
    K, f = my_press_ele.k_and_f_int(X, u)
    np.testing.assert_array_equal(K, np.zeros((4,4)))
    np.testing.assert_allclose(f, -np.sqrt(2)*np.array([1,0,1,0]))

@nose.tools.nottest
def test_tri6_pressure():
    '''
    The tri6 pressure element is not implemented yet. That's why this test 
    is disabled! 
    '''
    my_material = amfe.KirchhoffMaterial(rho=1)
    my_tri6 = amfe.Tri6(my_material)
    X = np.array([0,0,2,0,0,2,1,0,1,1,0,1.])
    u = np.zeros(12)
    X += sp.rand(12)*0.2
    M = my_tri6.m_int(X, u)
    t = np.array([ 0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.])
    f_2d = M @ t
    f_1d = f_2d[np.ix_([1,3,5,7,9,11])]
    my_boundary = amfe.Tri6Boundary(val=1., direct='normal', full_integration=True)
    X_3D = np.array([0,0,0,2,0,0,0,2,0,1,0,0,1,1,0,0,1.,0])
    u_3D = np.zeros(3*6)
    K, f = my_boundary.k_and_f_int(X_3D, u_3D)
    np.testing.assert_equal(K, np.zeros((18,18)))
    np.testing.assert_allclose(f_1d, f[np.ix_([2,5,8,11,14,17])])





#%%

class TestB_matrix_compuation(unittest.TestCase):
    '''
    Check the validity of the B-Matrix routine
    '''
    def produce_numbers(self, ndim):

        # Routine for testing the compute_B_matrix_routine
        # Try it the hard way:
        B_tilde = sp.rand(4, ndim)
        F = sp.rand(ndim, ndim)
        S_v = sp.rand(ndim*(ndim+1)//2)

        if ndim == 2:
            S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        else:
            S = np.array([[S_v[0], S_v[5], S_v[4]],
                          [S_v[5], S_v[1], S_v[3]],
                          [S_v[4], S_v[3], S_v[2]]])

        B = amfe.element.compute_B_matrix(B_tilde, F)
        self.res1 = B.T @ S_v
        self.res2 = B_tilde @ S @ F.T

    def test_2d(self):
        self.produce_numbers(2)
        np.testing.assert_allclose(self.res1, self.res2.reshape(-1))

    def test_3d(self):
        self.produce_numbers(3)
        np.testing.assert_allclose(self.res1, self.res2.reshape(-1))


class Test_fortran_vs_python(ElementTest):
    '''
    Compare the python and fortran element computation routines.
    '''
    
    def test_tri3(self):
        self.initialize_element(Tri3, X_tri3)
        self.check_python_vs_fortran()
    
    def test_tri6(self):
        self.initialize_element(Tri6, X_tri6)
        self.check_python_vs_fortran()
    
    def test_tet4(self):
        self.initialize_element(Tet4, X_tet4)
        self.check_python_vs_fortran()
        
    def test_mass_tri6(self):
        self.initialize_element(Tri6, X_tri6)
        self.my_element._m_int_python(self.X, self.u, t=0)
        M_py = self.my_element.M.copy()
        M_f = self.my_element._m_int(self.X, self.u, t=0)
        assert_almost_equal(M_f, M_py)
        
        
if __name__ == '__main__':
    unittest.main()
