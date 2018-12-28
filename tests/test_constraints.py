# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from scipy.sparse import eye as speye


class TestDirichletConstraint(TestCase):
    def setUp(self):
        from amfe.constraint import DirichletConstraint
        self.dofs = 4
        self.constraint_1 = DirichletConstraint(self.dofs, U=lambda t: 0, dU=lambda t: 0, ddU=lambda t: 0)
        self.constraint_2 = DirichletConstraint(self.dofs, U=lambda t: t ** 2, dU=lambda t: 3 * t, ddU=lambda t: 2)
        # set parameters:
        self.X_local = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
        self.u_local = np.array([0.1, 0.04, 0.02, 0.01], dtype=float)
        self.du_local = np.array([0.0, 0.0, 0.1, 0.2], dtype=float)
        self.ddu_local = np.array([0.0, 0.0, 0.5, 0.7], dtype=float)
        self.t = 2

    def tearDown(self):
        pass

    def test_constraint_func(self):
        # test constraint-functions
        constraint_1 = self.constraint_1.constraint_func(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        constraint_desired_1 = np.array([0.1, 0.04, 0.02, 0.01], dtype=float)
        assert_array_equal(constraint_desired_1, constraint_1)

        constraint_2 = self.constraint_2.constraint_func(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        constraint_desired_2 = np.array([-3.9, -3.96, -3.98, -3.99], dtype=float)
        assert_array_equal(constraint_desired_2, constraint_2)

    def test_jacobian(self):
        # test jacobians
        J_u_1 = self.constraint_1.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        J_u_2 = self.constraint_2.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        J_u_desired = speye(4, 4)
        assert_array_equal(J_u_1.todense(), J_u_desired.todense())
        assert_array_equal(J_u_2.todense(), J_u_desired.todense())
        
        J_du_1 = self.constraint_1.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t, 'du')
        J_du_2 = self.constraint_2.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t, 'du')
        J_du_desired = np.zeros((4, 4))
        assert_array_equal(J_du_1.todense(), J_du_desired)
        assert_array_equal(J_du_2.todense(), J_du_desired)
        
        J_ddu_1 = self.constraint_1.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t, 'ddu')
        J_ddu_2 = self.constraint_2.jacobian(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t, 'ddu')
        J_ddu_desired = np.zeros((4, 4))
        assert_array_equal(J_ddu_1.todense(), J_ddu_desired)
        assert_array_equal(J_ddu_2.todense(), J_ddu_desired)


class TestFixedDistanceConstraint(TestCase):
    def setUp(self):
        from amfe.constraint import FixedDistanceConstraint
        self.dofs = 4
        # set parameters:
        self.X_local = np.array([5.0, 6.0, 7.0, 8.0], dtype=float)
        self.u_local_1 = np.array([0.1, 0.02, 0.1, 0.02], dtype=float)
        self.u_local_2 = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self.u_local_3 = np.array([0.0, 0.0, -2.0, 0.0], dtype=float)
        self.du_local = np.array([0.0, 0.0, 0.1, 0.2], dtype=float)
        self.ddu_local = np.array([0.0, 0.0, 0.5, 0.7], dtype=float)
        self.t = 2
        distance = np.sqrt((self.X_local[2]-self.X_local[0])**2 + (self.X_local[3] - self.X_local[1])**2)

        self.constraint_1 = FixedDistanceConstraint()

    def tearDown(self):
        pass

    def test_constraint_func_zero(self):
        # test constraint-functions
        constraint_1 = self.constraint_1.constraint_func(self.X_local, self.u_local_1, self.du_local, self.ddu_local,
                                                         self.t)
        constraint_desired_1 = np.array(0.0, dtype=float)
        assert_array_equal(constraint_desired_1, constraint_1)

        constraint_2 = self.constraint_1.constraint_func(self.X_local, self.u_local_1, self.du_local, self.ddu_local,
                                                         self.t)
        constraint_desired_2 = np.array(0.0, dtype=float)
        assert_array_equal(constraint_desired_2, constraint_2)

        constraint_3 = self.constraint_1.constraint_func(self.X_local, self.u_local_1, self.du_local, self.ddu_local,
                                                         self.t)
        constraint_desired_3 = np.array(0.0, dtype=float)
        assert_array_equal(constraint_desired_3, constraint_3)

    def test_jacobian(self):
        # test jacobians with finite differences:
        J_u_1 = self.constraint_1.jacobian(self.X_local, self.u_local_1, self.du_local, self.ddu_local, self.t)
        J_u_2 = self.constraint_1.jacobian(self.X_local, self.u_local_2, self.du_local, self.ddu_local, self.t)
        delta = 0.001
        J_u_desired_1 = np.zeros((len(self.u_local_1), 1))
        J_u_desired_2 = np.zeros((len(self.u_local_2), 1))
        for i, _ in enumerate(J_u_desired_1):
            u_local_delta = np.zeros(len(self.u_local_1))
            u_local_delta[i] = 1
            J_u_desired_1[i] = (self.constraint_1.constraint_func(self.X_local, self.u_local_1+delta*u_local_delta, self.du_local, self.ddu_local, self.t)
                                - self.constraint_1.constraint_func(self.X_local, self.u_local_1-delta*u_local_delta, self.du_local, self.ddu_local, self.t)) / (2*delta)
            J_u_desired_2[i] = (self.constraint_1.constraint_func(self.X_local, self.u_local_2+delta*u_local_delta, self.du_local, self.ddu_local, self.t)
                                - self.constraint_1.constraint_func(self.X_local, self.u_local_2-delta*u_local_delta, self.du_local, self.ddu_local, self.t)) / (2*delta)

        assert_allclose(J_u_1.todense(), J_u_desired_1.T)
        assert_allclose(J_u_2.todense(), J_u_desired_2.T)

        J_du_1 = self.constraint_1.jacobian(self.X_local, self.u_local_1, self.du_local, self.ddu_local, self.t, 'du')
        J_du_2 = self.constraint_1.jacobian(self.X_local, self.u_local_2, self.du_local, self.ddu_local, self.t, 'du')
        J_du_desired = np.zeros((1, 4))
        assert_array_equal(J_du_1.todense(), J_du_desired)
        assert_array_equal(J_du_2.todense(), J_du_desired)

        J_ddu_1 = self.constraint_1.jacobian(self.X_local, self.u_local_1, self.du_local, self.ddu_local, self.t, 'ddu')
        J_ddu_2 = self.constraint_1.jacobian(self.X_local, self.u_local_2, self.du_local, self.ddu_local, self.t, 'ddu')
        J_ddu_desired = np.zeros((1, 4))
        assert_array_equal(J_ddu_1.todense(), J_ddu_desired)
        assert_array_equal(J_ddu_2.todense(), J_ddu_desired)
