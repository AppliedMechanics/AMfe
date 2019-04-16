"""Test Routine for constraint manager"""

from unittest import TestCase
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from pandas.testing import assert_frame_equal

from amfe.constraint.constraint_manager import ConstraintManager


class ConstraintManagerTest(TestCase):
    def setUp(self):
        self.cm = ConstraintManager(3)

        class DummyDirichletConstraint:

            NO_OF_CONSTRAINTS = 1

            def __init__(self, u_constr):
                self.u_constr = u_constr

            def after_assignment(self, dofids):
                return

            def constraint_func(self, X_local, u_local, t):
                return u_local - self.u_constr
            
            def jacobian(self, X_local, u_local, t):
                return np.array([1])

        self.diric_constraint = DummyDirichletConstraint(0)
        self.diric_constraint2 = DummyDirichletConstraint(0.5)

    def tearDown(self):
        self.cm = None

    def _add_two_dirichlet_constraints(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], [])
        self.cm.add_constraint('Dirich0', self.diric_constraint, [1], [])

    def test_no_of_constraint_definitions(self):
        self._add_two_dirichlet_constraints()
        self.assertEqual(self.cm.no_of_constraint_definitions, 1)

    def test_no_of_constraints(self):
        self._add_two_dirichlet_constraints()
        self.assertEqual(self.cm.no_of_constraints, 2)

    def test_no_of_dofs_unconstrained(self):
        self.assertEqual(self.cm.no_of_dofs_unconstrained, 3)
        self.cm.no_of_dofs_unconstrained = 1
        self.assertEqual(self.cm.no_of_dofs_unconstrained, 1)

    def test_add_constraint(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], [2])
        constraint_df_desired = pd.DataFrame({'name': 'Dirich0', 'constraint_obj': self.diric_constraint,
                                              'dofidxs': [np.array([2], dtype=int)],
                                              'Xidxs': [np.array([2], dtype=int)]})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired, check_like=True)
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], [1])
        constraint_df_desired = constraint_df_desired.append(pd.DataFrame({'name': 'Dirich1',
                                                                           'constraint_obj': self.diric_constraint,
                                                                           'dofidxs': [np.array([1], dtype=int)],
                                                                           'Xidxs': [np.array([1], dtype=int)]}),
                                                             ignore_index=True)
        print(self.cm._constraints_df)
        print(constraint_df_desired)
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired, check_like=True)
        
    def test_remove_constraint_by_name(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], [2])
        self.cm.add_constraint('Dirich0', self.diric_constraint, [1], [1])
        self.cm.remove_constraint_by_name('Dirich0')
        constraint_df_desired = pd.DataFrame({'name': [], 'constraint_obj': [],
                                              'dofidxs': [],
                                              'Xidxs': []}, dtype=object)
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired, check_like=True)

        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], [2])
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], [1])
        self.cm.remove_constraint_by_name('Dirich0')
        constraint_df_desired = pd.DataFrame({'name': 'Dirich1', 'constraint_obj': self.diric_constraint,
                                              'dofidxs': [np.array([1], dtype=int)],
                                              'Xidxs': [np.array([1], dtype=int)]})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired, check_like=True)

    def test_remove_constraint_by_dofidxs(self):
        self.cm.add_constraint('Dirich0', self.diric_constraint, [2], [2])
        self.cm.add_constraint('Dirich1', self.diric_constraint, [1], [1])
        self.cm.remove_constraint_by_dofidxs([2])
        constraint_df_desired = pd.DataFrame({'name': 'Dirich1', 'constraint_obj': self.diric_constraint,
                                              'dofidxs': [np.array([1], dtype=int)],
                                              'Xidxs': [np.array([1], dtype=int)]})
        assert_frame_equal(self.cm._constraints_df, constraint_df_desired, check_like=True)

    def _initalization_for_g_b_test(self):
        constraint_fixation = self.cm.create_dirichlet_constraint(U=lambda t: 0.0, dU=lambda t: 0.0,
                                                                  ddU=lambda t: 0.0)
        self.cm.add_constraint('Dirich0', constraint_fixation, [2], [2])
        self.cm.add_constraint('Dirich1', constraint_fixation, [1], [1])

    def test_g_and_B(self):
        self._initalization_for_g_b_test()
        u = np.array([0, 0, 0], dtype=float)
        X = np.arange(9).reshape(-1, 3)
        t = 0.0
        g, B = self.cm.g_and_B(X, u, t)
        B_desired = np.array([[0, 0, 1], [0, 1, 0]], dtype=float)
        g_desired = np.array([0, 0], dtype=float)
        assert_array_equal(B_desired, B.todense())
        assert_array_equal(g_desired, g)

    def test_g(self):
        self._initalization_for_g_b_test()
        u = np.array([0, 0, 0], dtype=float)
        X = np.arange(9).reshape(-1, 3)
        t = 0.0
        g = self.cm.g(X, u, t)

        g_desired = np.array([0, 0], dtype=float)

        assert_array_equal(g_desired, g)

    def test_B(self):
        self._initalization_for_g_b_test()
        u = np.array([0, 0, 0], dtype=float)
        X = np.arange(9).reshape(-1, 3)
        t = 0.0
        B = self.cm.B(X, u, t)
        B_desired = np.array([[0, 0, 1], [0, 1, 0]], dtype=float)
        assert_array_equal(B_desired, B.todense())


class PendulumConstraintManagerTest(TestCase):
        """
        Simple Pendulum test:

                  y ^ u_2
                    |
                    |   u_1
        Fixed:      o----->
                    \     x
                     \
                      \
                     L \ ^ u_4
                        \|
        Mass m:          O --> u_3
                        m

        """
        def setUp(self):
            # Length of the pendulum
            self.L = 2.0
            # Gravitation
            self.g = 10.0
            # Mass of the pendulum
            self.m = 1.0
            # X coordinates in reference configuration
            self.X = np.array([0.0, 0.0, 0.0, -self.L], dtype=float).reshape(-1, 2)
            # initial condition u_0: 90deg to the right
            self.u_0 = np.array([0.0, 0.0, self.L, self.L])
            # other initial conditions:
            self.du_0 = np.zeros(4)
            self.ddu_0 = np.zeros(4)
            # Create ConstraintManager
            self.cm = ConstraintManager(4)
            # create constraints (fixation and fixed distance as pendulum)
            self.constraint_fixation = self.cm.create_dirichlet_constraint(U=lambda t: 0.0, dU=lambda t: 0.0,
                                                                           ddU=lambda t: 0.0)
            self.constraint_pendulum = self.cm.create_fixed_distance_constraint()

        def tearDown(self):
            self.cm = None

        @staticmethod
        def _finite_difference(g_func, X, u, t, delta, B):
            for i in range(len(u)):
                delta_u = np.zeros_like(u)
                delta_u[i] = delta
                g_after_plus = g_func(X, u+delta_u, t)

                g_after_minus = g_func(X, u-delta_u, t)

                B[:, i] = (g_after_plus - g_after_minus) / (2 * delta)
            return B

        def test_pendulum(self):
            """
            Test that tests a pure lagrange formulation.
            """
            # Add constraints to manager
            self.cm.add_constraint('Fixation1', self.constraint_fixation,
                                   np.array([0], dtype=int),
                                   np.array([], dtype=int))
            self.cm.add_constraint('Fixation1', self.constraint_fixation,
                                   np.array([1], dtype=int),
                                   np.array([], dtype=int))
            self.cm.add_constraint('Pendulum', self.constraint_pendulum,
                                   np.array([0, 1, 2, 3], dtype=int),
                                   np.array([0, 1], dtype=int))
            # set time to zero
            t = 0.0

            # --- Test g and B ---
            g_actual, B_actual = self.cm.g_and_B(self.X, self.u_0, 0.0)
            g_desired = np.array([0, 0, 0], dtype=float)
            B_preallocated = np.zeros((3, 4))
            B_desired = self._finite_difference(self.cm.g, self.X, self.u_0, t, 0.000000001, B_preallocated)

            assert_array_equal(g_actual, g_desired)
            assert_allclose(B_actual.todense(), B_desired, rtol=1e-7)

            # Check if g function is zero for different positions
            # -90 deg
            u = np.array([0.0, 0.0, -self.L, self.L])
            g_actual = self.cm.g(self.X, u, 0.0,)
            assert_array_equal(g_actual, g_desired)

            u = np.array([0.0, 0.0, 0.0, -0.1*self.L])
            g_actual = self.cm.g(self.X, u, 0.0)
            self.assertEqual(g_actual[0], g_desired[0])
            self.assertEqual(g_actual[1], g_desired[1])
            self.assertGreater(g_actual[2], 0.0)

            # --- Test b ---
            # Define velocity in correct direction
            du = np.array([0.0, 0.0, 0.0, -1.0])
            B_actual = self.cm.B(self.X, self.u_0, 0.0)
            b_actual = self.cm.b(self.X, self.u_0, 0.0)

            assert_allclose(B_actual.dot(du) + b_actual, np.array([0.0, 0.0, 0.0]))

            # --- Test a ---
            # Define Acceleration in correct direction
            ddu_direction = np.array([0.0, 0.0, -1.0, -0.0])
            du = np.array([0.0, 0.0, 0.0, -1.0])
            # compute acceleration
            absolute_velocity = 1.0
            radius = self.L
            absolute_acceleration = absolute_velocity**2/radius
            ddu = ddu_direction*absolute_acceleration
            B_actual = self.cm.B(self.X, self.u_0, 0.0)
            a_actual = self.cm.a(self.X, self.u_0, du, 0.0)

            assert_allclose(B_actual.dot(ddu) + a_actual, np.array([0.0, 0.0, 0.0]), atol=1e-6)
