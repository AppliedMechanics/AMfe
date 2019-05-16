# -*- coding: utf-8 -*-
'''
Test for checking the stiffness matrices.
'''

import unittest
import numpy as np
import scipy as sp
import nose

from numpy.testing import assert_allclose, assert_almost_equal
from amfe.element import Tri3, Tri6, Quad4, Quad8, Tet4, Tet10, Hexa8, Hexa20, LinearBeam3D
from amfe.element import Tri3Boundary, Tri6Boundary, Quad4Boundary, Quad8Boundary, LineLinearBoundary
from amfe.element import compute_B_matrix
from amfe.material import KirchhoffMaterial, NeoHookean, MooneyRivlin, BeamMaterial


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


X_linear_beam = np.array([0, 0, 0, 2, 1, 0], dtype=float)
X_tri3 = np.array([0,0,3,1,2,2], dtype=float)
X_tri6 = np.array([0,0,3,1,2,2,1.5,0.5,2.5,1.5,1,1], dtype=float)
X_quad4 = np.array([0,0,1,0,1,1,0,1], dtype=float)
X_quad8 = np.array([0,0,1,0,1,1,0,1,0.5,0,1,0.5,0.5,1,0,0.5], dtype=float)
X_tet4 = np.array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1], dtype=float)
X_tet10 = np.array([0.,  0.,  0.,  2.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  2.,  1.,
                    0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  1.,  0.,
                    1.,  0.,  1.,  1.])
X_hexa8 = np.array([0,0,0, 1,0,0, 1,1,0, 0,1,0, 0,0,1, 1,0,1, 1,1,1, 0,1,1],
                   dtype=float)
X_hexa20  = np.array(
      [ 0. ,  0. ,  0. ,  1. ,  0. ,  0. ,  1. ,  1. ,  0. ,  0. ,  1. ,
        0. ,  0. ,  0. ,  1. ,  1. ,  0. ,  1. ,  1. ,  1. ,  1. ,  0. ,
        1. ,  1. ,  0.5,  0. ,  0. ,  1. ,  0.5,  0. ,  0.5,  1. ,  0. ,
        0. ,  0.5,  0. ,  0.5,  0. ,  1. ,  1. ,  0.5,  1. ,  0.5,  1. ,
        1. ,  0. ,  0.5,  1. ,  0. ,  0. ,  0.5,  1. ,  0. ,  0.5,  1. ,
        1. ,  0.5,  0. ,  1. ,  0.5])


class ElementTest(unittest.TestCase):
    '''Base class for testing the elements with the jacobian'''
    def initialize_element(self, element, X_def):
        no_of_dofs = len(X_def)
        self.X = X_def + 0.5*sp.rand(no_of_dofs)
        self.u = sp.rand(no_of_dofs)
        self.my_material = KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
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
        S = self.my_element.S.copy()
        E = self.my_element.E.copy()
        # fortran routine
        self.my_element._compute_tensors(self.X, self.u, t=0)
        assert_allclose(self.my_element.K, K)
        assert_allclose(self.my_element.f, f)
        assert_allclose(self.my_element.S, S)
        assert_allclose(self.my_element.E, E)


class LinearBeam3DTest(ElementTest):
    def setUp(self):
        no_of_dofs = 12
        self.X = X_linear_beam
        self.u = sp.rand(no_of_dofs)
        self.my_material = BeamMaterial(120, 80, 1000.0, 4.0, 23.0, 34.0, 132.0, (0.0, 0.0, 1E23))
        self.my_element = LinearBeam3D(self.my_material)

    def test_mass(self):
        # tests if the beam has the correct mass (tested with unit translation)
        X = X_linear_beam
        u = np.zeros(12)
        M = self.my_element.m_int(X, u)

        L = np.linalg.norm(X_linear_beam[3:6] - X_linear_beam[0:3])
        mass_theoretic = self.my_material.rho * self.my_material.crosssec * L
        # x translation:
        u[0] = 1.0
        u[6] = 1.0
        mass = u.T.dot(M).dot(u)
        assert_allclose(mass, mass_theoretic)

        # y translation:
        u = np.zeros(12)
        u[1] = 1.0
        u[7] = 1.0
        mass = u.T.dot(M).dot(u)
        assert_allclose(mass, mass_theoretic)

        # z translation:
        u = np.zeros(12)
        u[2] = 1.0
        u[8] = 1.0
        mass = u.T.dot(M).dot(u)
        assert_allclose(mass, mass_theoretic)


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

class Hexa8Test(ElementTest):
    def setUp(self):
        self.initialize_element(Hexa8, X_hexa8)

    def test_jacobi(self):
        self.jacobi_test_element(rtol=2E-3)

    def test_mass(self):
        my_material = KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
        my_element = Hexa8(my_material)
        M = my_element.m_int(X_hexa8, np.zeros_like(X_hexa8), t=0)
        np.testing.assert_almost_equal(np.sum(M), 3)

class Hexa20Test(ElementTest):
    def setUp(self):
        self.initialize_element(Hexa20, X_hexa20)

    def test_jacobi(self):
        self.jacobi_test_element(rtol=5E-3)

    def test_mass(self):
        my_material = KirchhoffMaterial(E=60, nu=1/4, rho=1, thickness=1)
        my_element = Hexa20(my_material)
        M = my_element.m_int(X_hexa20, np.zeros_like(X_hexa20), t=0)
        np.testing.assert_almost_equal(np.sum(M), 3)


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
        my_material = MooneyRivlin(A10, A01, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element(rtol=1E-3)

    def test_Neo(self):
        mu, kappa, rho = sp.rand(3)*1E3 + 100
        print('Material parameters mu, kappa:', mu, kappa)
#        mu /= 4
        my_material = NeoHookean(mu, kappa, rho)
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
        my_material = MooneyRivlin(A10, A01, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element(rtol=5E-4)

    def test_Neo(self):
        mu, kappa, rho = sp.rand(3)*1E3 + 100
#        mu /= 4
        kappa *= 100
        my_material = NeoHookean(mu, kappa, rho)
        self.my_element.material = my_material
        self.jacobi_test_element()



class MaterialTest(unittest.TestCase):
    def setUp(self):
        mu, kappa, rho = sp.rand(3)
        A10 = mu/2
        A01 = 0
        F = sp.rand(3,3)
        self.E = 1/2*(F.T @ F - sp.eye(3))
        self.mooney = MooneyRivlin(A10, A01, kappa, rho)
        self.neo = NeoHookean(mu, kappa, rho)

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
        self.mooney = MooneyRivlin(A10, A01, kappa, rho, plane_stress=False)
        self.neo = NeoHookean(mu, kappa, rho, plane_stress=False)

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
class BoundaryElementTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_tri3_pressure(self):
        X = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=float)
        u = np.zeros_like(X)
        f_mat_desired = np.array([[0, 0, -1/6], [0, 0, -1/6], [0, 0, -1/6]])
        my_press_ele = Tri3Boundary()
        f_mat = my_press_ele.f_mat(X, u)
        np.testing.assert_allclose(f_mat, f_mat_desired, rtol=1E-6, atol=1E-7)

    def test_line_pressure(self):
        X = np.array([0, 0, 1, 1], dtype=float)
        u = X
        f_mat_desired = np.array([[1, -1], [1, -1]], dtype=float)
        my_press_ele = LineLinearBoundary()
        f_mat = my_press_ele.f_mat(X, u)
        np.testing.assert_allclose(f_mat, f_mat_desired, rtol=1E-6, atol=1E-7)

    def test_tri6_pressure(self):
        X = np.array([0, 0, 2, 0, 0, 2, 1, 0, 1, 1, 0, 1.])
        f_mat_desired = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                  [0., 0., -8/3], [0., 0., -8/3], [0., 0., -8/3]])
        X_3D = np.zeros(3*6)
        X_3D[0::3] = X[0::2]
        X_3D[1::3] = X[1::2]
        u_3D = X_3D
        my_boundary = Tri6Boundary()
        f_mat = my_boundary.f_mat(X_3D, u_3D)
        np.testing.assert_allclose(f_mat, f_mat_desired, rtol=1E-6, atol=1E-7)

    def test_quad4_pressure(self):
        X = X_quad4
        f_mat_desired = np.array([[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]])
        X_3D = np.zeros(3*4)
        X_3D[0::3] = X[0::2]
        X_3D[1::3] = X[1::2]  # X_3D is a unit quad 4 element
        u_3D = X_3D  # this displacement vector leads to an area of 4 units
        my_boundary = Quad4Boundary()
        f_mat = my_boundary.f_mat(X_3D, u_3D)
        np.testing.assert_allclose(f_mat, f_mat_desired, rtol=1E-6, atol=1E-7)

    def test_quad8_pressure(self):
        X = X_quad8
        f_mat_desired = np.array([[0, 0, 1/3], [0, 0, 1/3], [0, 0, 1/3], [0, 0, 1/3],
                                  [0, 0, -4/3], [0, 0, -4/3], [0, 0, -4/3], [0, 0, -4/3]])
        X_3D = np.zeros(3*8)
        X_3D[0::3] = X[0::2]
        X_3D[1::3] = X[1::2]  # This is a unit Quad8
        u_3D = X_3D  # This displacement vector leads to an area of 4 units
        my_boundary = Quad8Boundary()
        f_mat = my_boundary.f_mat(X_3D, u_3D)
        np.testing.assert_allclose(f_mat, f_mat_desired, rtol=1E-6, atol=1E-7)

#%%

def test_name_tet4():
    assert('Tet4' == Tet4.name)

def test_name_tet10():
    assert('Tet10' == Tet10.name)

def test_name_quad4():
    assert('Quad4' == Quad4.name)

def test_name_quad8():
    assert('Quad8' == Quad8.name)

def test_name_tri3():
    assert('Tri3' == Tri3.name)

def test_name_tri6():
    assert('Tri6' == Tri6.name)

def test_name_hexa8():
    assert('Hexa8' == Hexa8.name)

def test_name_hexa20():
    assert('Hexa20' == Hexa20.name)

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

        B = compute_B_matrix(B_tilde, F)
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

    def test_mass_tri6(self):
        self.initialize_element(Tri6, X_tri6)
        self.my_element._m_int_python(self.X, self.u, t=0)
        M_py = self.my_element.M.copy()
        M_f = self.my_element._m_int(self.X, self.u, t=0)
        assert_almost_equal(M_f, M_py)

    def test_tet4(self):
        self.initialize_element(Tet4, X_tet4)
        self.check_python_vs_fortran()

    def test_tet10(self):
        self.initialize_element(Tet10, X_tet10)
        self.check_python_vs_fortran()

    def test_hexa8(self):
        self.initialize_element(Hexa8, X_hexa8)
        self.check_python_vs_fortran()

    def test_hexa20(self):
        self.initialize_element(Hexa20, X_hexa20)
        self.check_python_vs_fortran()

    def test_mass_hexa20(self):
        self.initialize_element(Hexa20, X_hexa20)
        self.my_element._m_int_python(self.X, self.u, t=0)
        M_py = self.my_element.M.copy()
        M_f = self.my_element._m_int(self.X, self.u, t=0)
        assert_almost_equal(M_f, M_py)


if __name__ == '__main__':
    unittest.main()
