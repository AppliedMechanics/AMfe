# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase, skip
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from amfe.constraint.tools import *
from amfe.constraint.constraint import *


class TestConstraintTools(TestCase):
    def setUp(self):
        return

    def test_constraints_scaling_factor(self):
        K = np.diag([1, 2, 3])
        M = 2.0*K
        D = 3.0*K
        dt = 0.1

        actual_1 = constraints_scaling_factor(dt, K, M)
        actual_2 = constraints_scaling_factor(dt, K, M, D)
        mean_k = (1.0 + 2.0 + 3.0)/3
        mean_m = 2.0 * mean_k
        mean_d = 3.0 * mean_k
        desired_1 = mean_k + mean_m/(0.1*0.1)
        desired_2 = mean_k + mean_m/(0.1*0.1) + mean_d/0.1
        self.assertAlmostEqual(actual_1, desired_1)
        self.assertAlmostEqual(actual_2, desired_2)

    def test_validate_constraints_independent(self):
        B_independent = np.array([[1, 0, 0], [1, 1, 0]], dtype=float)
        B_dependent = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        B_dependent_tol = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 1e-5, 0]], dtype=float)

        actual_independent = validate_constraints_independent(B_independent)
        actual_dependent = validate_constraints_independent(B_dependent)
        actual_tol1 = validate_constraints_independent(B_dependent_tol, 1e-6)
        actual_tol2 = validate_constraints_independent(B_dependent_tol, 1e-12)

        self.assertTrue(actual_independent)
        self.assertFalse(actual_dependent)
        self.assertFalse(actual_tol1)
        self.assertTrue(actual_tol2)


class TestNonholonomicBase(TestCase):
    def setUp(self):
        self.constraint = NonholonomicConstraintBase()
        self.X = np.array([0.0])
        self.u = np.array([0.0])
        self.t = 0.0

    def tearDown(self):
        self.constraint = None

    def test_nonholonomic_constraint_signature(self):
        with self.assertRaises(NotImplementedError):
            self.constraint.B(self.X, self.u, self.t)
        with self.assertRaises(NotImplementedError):
            self.constraint.b(self.X, self.u, self.t)
        with self.assertRaises(NotImplementedError):
            self.constraint.a(self.X, self.u, self.u, self.t)


class TestHolonomicBase(TestCase):
    def setUp(self):
        self.constraint = HolonomicConstraintBase()
        self.X = np.array([0.0])
        self.u = np.array([0.0])
        self.t = 0.0

    def tearDown(self):
        self.constraint = None

    def test_holonomic_constraint_signature(self):
        with self.assertRaises(NotImplementedError):
            self.constraint.B(self.X, self.u, self.t)
        with self.assertRaises(NotImplementedError):
            self.constraint.b(self.X, self.u, self.t)
        with self.assertRaises(NotImplementedError):
            self.constraint.a(self.X, self.u, self.u, self.t)
        with self.assertRaises(NotImplementedError):
            self.constraint.g(self.X, self.u, self.t)


class TestDirichletConstraint(TestCase):
    def setUp(self):

        self.dofs = 4

        def U1(t):
            return 0

        def dU1(t):
            return 0

        def ddU1(t):
            return 0

        def U2(t):
            return t**2

        def dU2(t):
            return 3*t

        def ddU2(t):
            return 2

        self.U1 = U1
        self.dU1 = dU1
        self.ddU1 = ddU1
        self.U2 = U2
        self.dU2 = dU2
        self.ddU2 = ddU2

        self.constraint_1 = DirichletConstraint(U=lambda t: 0, dU=lambda t: 0, ddU=lambda t: 0)
        self.constraint_2 = DirichletConstraint(U=lambda t: t ** 2, dU=lambda t: 3 * t, ddU=lambda t: 2)
        # set parameters:
        self.X_local = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
        self.u_local = np.array([0.1, 0.04, 0.02, 0.01], dtype=float)
        self.du_local = np.array([0.0, 0.0, 0.1, 0.2], dtype=float)
        self.t = 2

    def tearDown(self):
        pass

    def test_constraint_func(self):
        # test constraint-functions
        constraint_desired_1 = np.array([0.1, 0.04, 0.02, 0.01], dtype=float)
        constraint_desired_2 = np.array([-3.9, -3.96, -3.98, -3.99], dtype=float)
        for X, u, desired_1, desired_2 in zip(self.X_local, self.u_local,
                                              constraint_desired_1, constraint_desired_2):
            constraint_1 = self.constraint_1.g(X, u, self.t)
            assert_array_equal(desired_1, constraint_1)

            constraint_2 = self.constraint_2.g(X, u, self.t)

            assert_array_equal(desired_2, constraint_2)

    def test_jacobian(self):
        # test jacobians
        B_desired = np.array([1], dtype=float)

        for X, u, in zip(self.X_local, self.u_local):
            B_1 = self.constraint_1.B(X, u, self.t)
            B_2 = self.constraint_2.B(X, u, self.t)
            assert_array_equal(B_1, B_desired)
            assert_array_equal(B_2, B_desired)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        for X, u, in zip(self.X_local, self.u_local):
            B_1 = self.constraint_1.B(X, u, self.t)
            B_2 = self.constraint_2.B(X, u, self.t)
            b_1 = self.constraint_1.b(X, u, self.t)
            b_2 = self.constraint_2.b(X, u, self.t)
            assert_allclose(B_1*self.dU1(self.t) + b_1, b_desired)
            assert_allclose(B_2*self.dU2(self.t) + b_2, b_desired)

    def test_a(self):
        a_desired = np.array([0], dtype=float)
        for X, u, in zip(self.X_local, self.u_local):
            B_1 = self.constraint_1.B(X, u, self.t)
            B_2 = self.constraint_2.B(X, u, self.t)
            a_1 = self.constraint_1.a(X, u, self.dU1(self.t), self.t)
            a_2 = self.constraint_2.a(X, u, self.dU2(self.t), self.t)
            assert_allclose(B_1 * self.ddU1(self.t) + a_1, a_desired)
            assert_allclose(B_2 * self.ddU2(self.t) + a_2, a_desired)


class TestFixedDistanceConstraint(TestCase):
    def setUp(self):
        self.dofs = 4
        # set parameters:
        self.X_local = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
        self.u_local_1 = np.array([0.1, 0.02, 0.1, 0.02], dtype=float)
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self.u_local_3 = np.array([0.0, 0.0, -2.0, 0.0], dtype=float)
        self.du_local = np.array([0.0, 0.0, 0.1, 0.2], dtype=float)
        self.t = 2

        self.constraint_1 = FixedDistanceConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        constraint_desired_1 = np.array(0.0, dtype=float)
        assert_array_equal(constraint_desired_1, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        constraint_desired_2 = np.array(0.0, dtype=float)
        assert_array_equal(constraint_desired_2, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        self.assertLess(constraint_3[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T)
        assert_allclose(J_u_2, J_u_desired_2.T)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if a rotation of the second node around the first node returns zero
        assert_allclose(B_1.dot(np.array([0.0, 0.0, -1.0, 1.0])) + b_1, b_desired)

    def test_a(self):
        a_desired = np.array([0], dtype=float)
        velocity = np.array([0.0, 0.0, -1.0, 1.0])
        B_1 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        a_1 = self.constraint_1.a(self.X_local, self.u_local_2, velocity, self.t)
        # Check if a rotation of the second node around the first node returns zero
        # The acceleration would be into the center (Zentripetal)
        direction = np.array([0.0, 0.0, -1.0, -1.0])/np.sqrt(2)
        absolute_velocity = np.linalg.norm(velocity)
        radius = np.linalg.norm(self.X_local[2:]-self.X_local[:2])
        absolute_acceleration = absolute_velocity**2/radius
        assert_allclose(B_1.dot(absolute_acceleration*direction) + a_1, a_desired, atol=1e-6)


class TestFixedDistanceToLineConstraint(TestCase):
    def setUp(self):
        """
                     o  -
                        ^
                        | fixed distance
                        v
        o------------o  -


        Returns
        -------

        """
        self.dofs = 9
        # set parameters:
        self.X_local = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0], dtype=float)
        # shift line to right
        self.u_local_1 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # zero
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        # turn point around line:
        self.u_local_3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0])
        # Turn line and point
        self.u_local_4 = np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0, -1.0, 0.0, 0.0])
        # violated constraint
        self.u_local_5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0])

        self.du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 2

        self.constraint_1 = FixedDistanceToLineConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_desired = np.array(0.0, dtype=float)

        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_3)

        constraint_4 = self.constraint_1.g(self.X_local, self.u_local_4,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_4)

        # violated constraint
        constraint_5 = self.constraint_1.g(self.X_local, self.u_local_5,
                                           self.t)
        self.assertLess(constraint_5[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.00001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T, atol=1e-9)
        assert_allclose(J_u_2, J_u_desired_2.T, atol=1e-9)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if a rotation of the node around the line returns zero
        du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)
        # Check if a same velocity of line and point returns zero
        du_local = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)

    def test_a(self):
        with self.assertRaises(NotImplementedError):
            a_1 = self.constraint_1.a(self.X_local, self.u_local_1, self.u_local_1, self.t)


class TestNodesCollinear2DConstraint(TestCase):
    def setUp(self):
        """

        o--------o----o


        Returns
        -------

        """
        self.dofs = 6
        # set parameters:
        self.X_local = np.array([0.0, 0.0, 2.0, 0.0, 1.0, 0.0], dtype=float)
        # shift line to right
        self.u_local_1 = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        # zero
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        # move point along line
        self.u_local_3 = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 0.0])
        # Turn line and point
        self.u_local_4 = np.array([0.0, 0.0, -2.0, -2.0, -1.0, -1.0])
        # violated constraint
        self.u_local_5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.1])

        self.du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 2

        self.constraint_1 = NodesCollinear2DConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_desired = np.array(0.0, dtype=float)

        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_3)

        constraint_4 = self.constraint_1.g(self.X_local, self.u_local_4,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_4)

        # violated constraint
        constraint_5 = self.constraint_1.g(self.X_local, self.u_local_5,
                                           self.t)
        self.assertLess(constraint_5[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.00001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T, atol=1e-9)
        assert_allclose(J_u_2, J_u_desired_2.T, atol=1e-9)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if constraint returns zero if node moves along line
        du_local = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)
        # Check if a same velocity of line and point returns zero
        du_local = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)

    def test_a(self):
        with self.assertRaises(NotImplementedError):
            a_1 = self.constraint_1.a(self.X_local, self.u_local_1, self.u_local_1, self.t)


class TestEqualDisplacementConstraint(TestCase):
    def setUp(self):
        """
                    o -->

        o -->


        Returns
        -------

        """
        self.dofs = 2
        # set parameters:
        self.X_local = np.array([], dtype=float)
        # shift to right
        self.u_local_1 = np.array([1.0, 1.0])
        # zero
        self.u_local_2 = np.array([0.0, 0.0], dtype=float)
        # violated constraint
        self.u_local_3 = np.array([1.0, 3.0])

        self.du_local = np.array([0.0, 0.0], dtype=float)
        self.t = 2

        self.constraint_1 = EqualDisplacementConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_desired = np.array(0.0, dtype=float)

        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        self.assertGreater(constraint_3[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.00001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T, atol=1e-9)
        assert_allclose(J_u_2, J_u_desired_2.T, atol=1e-9)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if constraint returns zero if node moves along line
        du_local = np.array([0.0, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)
        # Check if a same velocity of line and point returns zero
        du_local = np.array([3.0, 3.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)

    def test_a(self):
        a_1 = self.constraint_1.a(self.X_local, self.u_local_1, self.u_local_1, self.t)
        assert_array_equal(a_1, np.array([0.0], ndmin=1))


@skip("temporarily disabled")
class TestFixedDistanceToPlaneConstraint(TestCase):
    def setUp(self):
        """
                     o
                     |
                     |
                     |    plane    point in z direction
        o------------o


        Returns
        -------

        """
        self.dofs = 12
        # set parameters:
        self.X_local = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 5.0], dtype=float)
        # shift bottom line to right
        self.u_local_1 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # zero
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        # turn plane such that it is a yz-plane
        self.u_local_3 = np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0, -2.0, -1.0, 1.0, -5.0, 0.0, -5.0])
        # Move point parallel to plane
        self.u_local_4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 40.0, 0.0])
        # violated constraint
        self.u_local_5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1])

        self.du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 2

        self.constraint_1 = FixedDistanceToPlaneConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_desired = np.array(0.0, dtype=float)

        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_3)

        constraint_4 = self.constraint_1.g(self.X_local, self.u_local_4,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_4)

        # violated constraint
        constraint_5 = self.constraint_1.g(self.X_local, self.u_local_5,
                                           self.t)
        self.assertLess(constraint_5[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.000001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T, atol=1e-9)
        assert_allclose(J_u_2, J_u_desired_2.T, atol=1e-9)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if a move with same distance leads to zero
        du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 2.0, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)
        # Check if a same velocity of line and point returns zero
        du_local = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)

    def test_a(self):
        with self.assertRaises(NotImplementedError):
            a_1 = self.constraint_1.a(self.X_local, self.u_local_1, self.u_local_1, self.t)


class TestNodesCoplanarConstraint(TestCase):
    def setUp(self):
        """
                     o
                     |
               o     |
                     |    plane    point in z direction
        o------------o


        Returns
        -------

        """
        self.dofs = 12
        # set parameters:
        self.X_local = np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.5, 0.0], dtype=float)
        # shift bottom line to right
        self.u_local_1 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # zero
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        # turn plane such that it is a yz-plane
        self.u_local_3 = np.array([0.0, 0.0, 0.0, -2.0, -2.0, 0.0, -2.0, -1.0, 1.0, -1.0, -0.5, 0.5])
        # Move point parallel to plane
        self.u_local_4 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 40.0, 0.0])
        # violated constraint
        self.u_local_5 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1])

        self.du_local = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 2

        self.constraint_1 = NodesCoplanarConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_desired = np.array(0.0, dtype=float)

        constraint_1 = self.constraint_1.g(self.X_local, self.u_local_1,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_1)

        constraint_2 = self.constraint_1.g(self.X_local, self.u_local_2,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_2)

        constraint_3 = self.constraint_1.g(self.X_local, self.u_local_3,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_3)

        constraint_4 = self.constraint_1.g(self.X_local, self.u_local_4,
                                           self.t)
        assert_array_equal(constraint_desired, constraint_4)

        # violated constraint
        constraint_5 = self.constraint_1.g(self.X_local, self.u_local_5,
                                           self.t)
        self.assertGreater(constraint_5[0], 0.0)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        J_u_2 = self.constraint_1.B(self.X_local, self.u_local_2, self.t)
        delta = 0.000001
        J_u_desired_1 = np.zeros((len(self.u_local_1)))
        J_u_desired_2 = np.zeros((len(self.u_local_2)))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.g(self.X_local, self.u_local_1 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_1 - delta * u_local_delta,
                                                      self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.g(self.X_local, self.u_local_2 + delta * u_local_delta,
                                                    self.t)
                                - self.constraint_1.g(self.X_local, self.u_local_2 - delta * u_local_delta,
                                                      self.t)) / (2*delta)

        assert_allclose(J_u_1, J_u_desired_1.T, atol=1e-9)
        assert_allclose(J_u_2, J_u_desired_2.T, atol=1e-9)

    def test_b(self):
        b_desired = np.array([0], dtype=float)
        B_1 = self.constraint_1.B(self.X_local, self.u_local_1, self.t)
        b_1 = self.constraint_1.b(self.X_local, self.u_local_1, self.t)
        # Check if a the move of all points inside plane returns zero
        du_local = np.array([1.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.6, 0.7, 0.0, 0.1, 0.2, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)
        # Check if same velocity returns zero
        du_local = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
        assert_allclose(B_1.dot(du_local) + b_1, b_desired)

    def test_a(self):
        with self.assertRaises(NotImplementedError):
            a_1 = self.constraint_1.a(self.X_local, self.u_local_1, self.u_local_1, self.t)
