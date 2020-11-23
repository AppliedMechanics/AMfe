"""Test Routine for constraint manager"""

from unittest import TestCase
import numpy as np
from scipy.sparse import csr_matrix
from numpy.testing import assert_array_equal
from amfe.constraint.constraint_formulation_boolean_elimination import BooleanEliminationConstraintFormulation


class BooleanEliminationFormulationTest(TestCase):
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

        def M_dense(u, du, t):
            return M(u, du, t).toarray()

        def h(u, du, t):
             return self.f_int_unconstr

        def p(u, du, t):
            return self.f_ext_unconstr

        def h_q(u, du, t):
            return self.K_unconstr

        def h_q_dense(u, du, t):
            return self.K_unconstr.toarray()

        def h_dq(u, du, t):
            return self.D_unconstr

        def h_dq_dense(u, du, t):
            return self.D_unconstr.toarray()

        # Assumption: Fixed Dirichlet Condition on first dof
        def g_holo(u, t):
            return np.array(u[0], dtype=float, ndmin=1)

        def B_holo(u, t):
            return csr_matrix(np.array([[1, 0, 0]], dtype=float, ndmin=2))

        def B_holo_dense(u, t):
            return B_holo(u, t).toarray()

        self.no_of_constraints = 1
        self.no_of_dofs_unconstrained = 3
        self.M_func = M
        self.M_func_dense = M_dense
        self.h_func = h
        self.p_func = p
        self.h_q_func = h_q
        self.h_dq_func = h_dq
        self.h_q_func_dense = h_q_dense
        self.h_dq_func_dense = h_dq_dense
        self.g_holo_func = g_holo
        self.B_holo_func = B_holo
        self.B_holo_func_dense = B_holo_dense

        self.formulation = BooleanEliminationConstraintFormulation(self.no_of_dofs_unconstrained, self.M_func,
                                                                   self.h_func, self.B_holo_func, self.p_func,
                                                                   self.h_q_func, self.h_dq_func,
                                                                   g_func=self.g_holo_func)

        self.formulation_dense = BooleanEliminationConstraintFormulation(self.no_of_dofs_unconstrained,
                                                                         self.M_func_dense,
                                                                         self.h_func, self.B_holo_func_dense,
                                                                         self.p_func,
                                                                         self.h_q_func_dense,
                                                                         self.h_dq_func_dense,
                                                                         g_func=self.g_holo_func)

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
                         self.no_of_dofs_unconstrained - self.no_of_constraints)

        self.assertEqual(self.formulation_dense.dimension,
                         self.no_of_dofs_unconstrained - self.no_of_constraints)

    def test_update(self):
        # Just test if update function works
        self.formulation.update()

    def test_recover_u_du_ddu(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy() + 1.0
        ddx = dx.copy() + 1.0

        u, du, ddu = self.formulation.recover(x, dx, ddx, 5.0)
        zero_array = np.array([0.0], ndmin=1)
        assert_array_equal(u, np.concatenate((zero_array, x)))
        assert_array_equal(du, np.concatenate((zero_array, dx)))
        assert_array_equal(ddu, np.concatenate((zero_array, ddx)))

        u = self.formulation.u(x, 2.0)
        du = self.formulation.du(x, dx, 3.0)
        ddu = self.formulation.ddu(x, dx, ddx, 6.0)
        assert_array_equal(u, np.concatenate((zero_array, x)))
        assert_array_equal(du, np.concatenate((zero_array, dx)))
        assert_array_equal(ddu, np.concatenate((zero_array, ddx)))

    def test_M(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        M_desired = self.M_unconstr[1:, :][:, 1:]

        M_actual = self.formulation.M(x, dx, 0.0)
        M_actual_dense = self.formulation_dense.M(x, dx, 0.0)
        assert_array_equal(M_actual.todense(), M_desired.todense())
        assert_array_equal(M_actual_dense, M_desired.todense())

    def test_f_int(self):
        x = np.arange(self.no_of_dofs_unconstrained - self.no_of_constraints,
                      dtype=float) + 1.0
        dx = x.copy() + 1.0

        zero_array = np.array([0.0], ndmin=1)
        u = np.concatenate((zero_array, x), axis=0)
        du = np.concatenate((zero_array, dx), axis=0)
        t = 0.0
        f_int_desired = self.h_func(u, du, t)[1:]
        f_int_actual = self.formulation.f_int(x, dx, t)
        f_int_actual_dense = self.formulation_dense.f_int(x, dx, t)
        assert_array_equal(f_int_actual, f_int_desired)
        assert_array_equal(f_int_actual_dense, f_int_desired)

    def test_f_ext(self):
        x = np.arange(self.no_of_dofs_unconstrained - self.no_of_constraints,
                      dtype=float) + 1.0
        dx = x.copy() + 1.0

        zero_array = np.array([0.0], ndmin=1)
        u = np.concatenate((zero_array, x), axis=0)
        du = np.concatenate((zero_array, dx), axis=0)
        t = 0.0
        f_ext_desired = self.p_func(u, du, t)[1:]
        f_ext_actual = self.formulation.f_ext(x, dx, t)
        f_ext_actual_dense = self.formulation_dense.f_ext(x, dx, t)
        assert_array_equal(f_ext_actual, f_ext_desired)
        assert_array_equal(f_ext_actual_dense, f_ext_desired)

    def test_D(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        D_desired = self.D_unconstr[1:, :][:, 1:]

        D_actual = self.formulation.D(x, dx, 0.0)
        assert_array_equal(D_actual.todense(), D_desired.todense())
        D_actual_dense = self.formulation_dense.D(x, dx, 0.0)
        assert_array_equal(D_actual_dense, D_desired.todense())

    def test_K(self):
        x = np.arange(self.formulation.dimension, dtype=float)
        dx = x.copy()

        K_desired = self.K_unconstr[1:, :][:, 1:]

        K_actual = self.formulation.K(x, dx, 0.0)
        assert_array_equal(K_actual.todense(), K_desired.todense())
        K_actual_dense = self.formulation_dense.K(x, dx, 0.0)
        assert_array_equal(K_actual_dense, K_desired.todense())

    def test_L(self):
        L_desired = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        L_actual = self.formulation.L
        self.assertIsInstance(L_actual, csr_matrix)
        assert_array_equal(L_actual.todense(), L_desired)

        L_actual_dense = self.formulation_dense.L
        assert_array_equal(L_actual_dense, L_desired)

    def test_L_without_constraints(self):
        def B(u, t):
            return csr_matrix((0, 3))

        def g(u, t):
            return np.array([], dtype=float, ndmin=1)

        formulation = BooleanEliminationConstraintFormulation(self.no_of_dofs_unconstrained, self.M_func,
                                                              self.h_func, B, self.p_func,
                                                              self.h_q_func, self.h_dq_func,
                                                              g_func=g)

        L_desired = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        L_actual = formulation.L
        assert_array_equal(L_actual.todense(), L_desired)
