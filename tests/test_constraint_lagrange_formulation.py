"""Test Routine for constraint manager"""

from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sphstack
from scipy.sparse import vstack as spvstack
from numpy.testing import assert_array_equal, assert_allclose

from amfe.constraint.constraint_formulation_lagrange_multiplier import SparseLagrangeMultiplierConstraintFormulation


class SparseLagrangeFormulationTest(TestCase):
    def setUp(self):
        M_mat = np.array([[1, -1, 0], [-1, 1.2, -1.5], [0, -1.5, 2]], dtype=float)
        K_mat = np.array([[2, -1, 0], [-1, 2, -1.5], [0, -1.5, 3]], dtype=float)
        D_mat = 0.2 * M_mat + 0.1 * K_mat

        self.M_unconstr = csr_matrix(M_mat)
        self.D_unconstr = csr_matrix(D_mat)
        self.K_unconstr = csr_matrix(K_mat)
        self.f_int_unconstr = np.array([1, 2, 3], dtype=float)
        self.f_ext_unconstr = np.array([3, 4, 5], dtype=float)

        def M(u, du, t):
            return self.M_unconstr

        def h(u, du, t):
            return self.f_ext_unconstr - self.f_int_unconstr

        def h_q(u, du, t):
            return -self.K_unconstr

        def h_dq(u, du, t):
            return -self.D_unconstr

        def g_holo(u, t):
            return np.array(u[0], dtype=float, ndmin=1)

        def B_holo(u, t):
            return csr_matrix(np.array([[1, 0, 0]], dtype=float, ndmin=2))

        self.no_of_constraints = 1
        self.no_of_dofs_unconstrained = 3
        self.M_func = M
        self.h_func = h
        self.h_q_func = h_q
        self.h_dq_func = h_dq
        self.g_holo_func = g_holo
        self.B_holo_func = B_holo

        self.formulation = SparseLagrangeMultiplierConstraintFormulation(self.no_of_dofs_unconstrained, self.M_func,
                                                                         self.h_func, self.B_holo_func, self.h_q_func,
                                                                         self.h_dq_func, self.g_holo_func)

    def tearDown(self):
        self.formulation = None

    def test_no_of_dofs_unconstrained(self):
        self.assertEqual(self.formulation.no_of_dofs_unconstrained,
                         self.no_of_dofs_unconstrained)

        self.formulation.no_of_dofs_unconstrained = 5
        self.assertEqual(self.formulation.no_of_dofs_unconstrained,
                         5)

    def test_dimension(self):
        self.assertEqual(self.formulation.dimension,
                         self.no_of_dofs_unconstrained + self.no_of_constraints)

    def test_update(self):
        # Just test if update function works
        self.formulation.update()

    def test_recover_u_du_ddu_lambda(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy() + 1.0
        ddx = dx.copy() + 1.0

        u, du, ddu = self.formulation.recover(x, dx, ddx, 5.0)
        assert_array_equal(u, x[:self.no_of_dofs_unconstrained])
        assert_array_equal(du, dx[:self.no_of_dofs_unconstrained])
        assert_array_equal(ddu, ddx[:self.no_of_dofs_unconstrained])

        u = self.formulation.u(x, 2.0)
        du = self.formulation.du(x, dx, 3.0)
        ddu = self.formulation.ddu(x, dx, ddx, 6.0)
        lagrange_multiplier = self.formulation.lagrange_multiplier(x, 6.0)
        assert_array_equal(u, x[:self.no_of_dofs_unconstrained])
        assert_array_equal(du, dx[:self.no_of_dofs_unconstrained])
        assert_array_equal(ddu, ddx[:self.no_of_dofs_unconstrained])
        assert_array_equal(lagrange_multiplier, x[self.no_of_dofs_unconstrained:])

    def test_M(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        M_desired = spvstack((sphstack((self.M_unconstr,
                                        csr_matrix((self.no_of_dofs_unconstrained,
                                                    self.no_of_constraints))), format='csr'),
                             sphstack((csr_matrix((self.no_of_constraints,
                                                   self.no_of_dofs_unconstrained)),
                                       csr_matrix((1, 1))), format='csr')),
                             format='csr')

        M_actual = self.formulation.M(x, dx, 0.0)
        assert_array_equal(M_actual.todense(), M_desired.todense())

    def test_F(self):
        x = np.arange(self.no_of_dofs_unconstrained + self.no_of_constraints,
                      dtype=float) + 1.0
        dx = x.copy() + 1.0
        u = x[:self.no_of_dofs_unconstrained]
        du = dx[:self.no_of_dofs_unconstrained]
        t = 0.0
        F_desired = np.concatenate((self.h_func(u, du, t)-self.B_holo_func(u, t).T.dot(x[self.no_of_dofs_unconstrained:]),
                                   -self.g_holo_func(u, t)))
        F_actual = self.formulation.F(x, dx, t)
        assert_array_equal(F_actual, F_desired)

    def test_D(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        D_desired = spvstack((sphstack((self.D_unconstr,
                                        csr_matrix((self.no_of_dofs_unconstrained,
                                                    self.no_of_constraints))), format='csr'),
                             sphstack((csr_matrix((self.no_of_constraints,
                                                   self.no_of_dofs_unconstrained)),
                                       csr_matrix((1, 1))), format='csr')),
                             format='csr')

        D_actual = self.formulation.D(x, dx, 0.0)
        assert_array_equal(D_actual.todense(), D_desired.todense())

    def test_K(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        K_desired = spvstack((sphstack((self.K_unconstr,
                                        self.B_holo_func(x[:self.no_of_dofs_unconstrained],
                                                         0.0).T), format='csr'),
                             sphstack((self.B_holo_func(x[:self.no_of_dofs_unconstrained], 0.0),
                                       csr_matrix((1, 1))), format='csr')),
                             format='csr')

        K_actual = self.formulation.K(x, dx, 0.0)
        assert_array_equal(K_actual.todense(), K_desired.todense())

    def test_scaling(self):
        x0 = np.zeros(self.formulation.dimension)
        x0[0] = 3.141
        dx0 = np.zeros(self.formulation.dimension)
        h0 = self.h_func(x0, dx0, 0.0)
        F0 = np.zeros(self.formulation.dimension)
        F0[:len(h0)] = h0

        # Test only constraint equation scaling
        self.formulation.set_options(scaling=2.0)
        F_scale_2 = self.formulation.F(x0, dx0, 0.0).copy()

        self.formulation.set_options(scaling=4.0)
        F_scale_4 = self.formulation.F(x0, dx0, 0.0).copy()

        assert_array_equal((F_scale_2-F0)/2.0, (F_scale_4-F0)/4.0)

        # Test B.T lambda scaling
        x0 = np.zeros(self.formulation.dimension)
        x0[-1] = 2.7

        self.formulation.set_options(scaling=2.0)
        F_scale_2 = self.formulation.F(x0, dx0, 0.0).copy()

        self.formulation.set_options(scaling=4.0)
        F_scale_4 = self.formulation.F(x0, dx0, 0.0).copy()

        assert_array_equal((F_scale_2-F0)/2.0, (F_scale_4-F0)/4.0)

        # Test stiffness matrix scaling
        K0 = self.K_unconstr
        K0_full = np.zeros((self.formulation.dimension, self.formulation.dimension))
        for i in range(self.K_unconstr.shape[0]):
            K0_full[i, :self.K_unconstr.shape[1]] = K0[i, :].toarray()[:]

        x0 = np.zeros(self.formulation.dimension)
        x0[-1] = 2.7
        x0[0] = 3.141

        self.formulation.set_options(scaling=2.0)
        K_scale_2 = self.formulation.K(x0, dx0, 0.0).copy()

        self.formulation.set_options(scaling=4.0)
        K_scale_4 = self.formulation.K(x0, dx0, 0.0).copy()

        assert_array_equal((K_scale_2 - K0_full) / 2.0, (K_scale_4 - K0_full) / 4.0)

    def test_penalty(self):
        scale1 = 2.0
        scale2 = 4.0

        x0 = np.zeros(self.formulation.dimension)
        x0[-1] = 2.7
        x0[0] = 3.141
        dx0 = np.zeros(self.formulation.dimension)

        B = self.B_holo_func(x0[:-1], 0.0)
        BTB = B.T.dot(B)
        KBTB = np.zeros((self.formulation.dimension, self.formulation.dimension))
        for i in range(BTB.shape[0]):
            KBTB[i, :BTB.shape[1]] = BTB.toarray()[i, :]
        BTg = B.T.dot(self.g_holo_func(x0[:-1], 0.0))
        FBTg = np.zeros(self.formulation.dimension)
        FBTg[:len(BTg)] = BTg

        self.formulation.set_options(penalty=scale1)
        K_pen_2 = self.formulation.K(x0, dx0, 0.0).copy()
        F_pen_2 = self.formulation.F(x0, dx0, 0.0).copy()

        self.formulation.set_options(penalty=scale2)
        K_pen_4 = self.formulation.K(x0, dx0, 0.0).copy()
        F_pen_4 = self.formulation.F(x0, dx0, 0.0).copy()

        assert_allclose(K_pen_2 - scale1 * KBTB, K_pen_4 - scale2 * KBTB)
        assert_allclose(F_pen_2 + scale1 * FBTg, F_pen_4 + scale2 * FBTg)


