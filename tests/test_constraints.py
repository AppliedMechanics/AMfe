# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import eye as speye


class TestDirichletConstraint(TestCase):
    def setUp(self):
        from amfe.constraint import DirichletConstraint
        self.dofs = np.array([0, 1, 4, 5], dtype=int)
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

    def test_c_equation(self):
        # test c_equations
        c_equation_1 = self.constraint_1.c(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        c_equation_desired_1 = np.array([0.1, 0.04, 0.02, 0.01], dtype=float)
        assert_array_equal(c_equation_desired_1, c_equation_1)

        c_equation_2 = self.constraint_2.c(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        c_equation_desired_2 = np.array([-3.9, -3.96, -3.98, -3.99], dtype=float)
        assert_array_equal(c_equation_desired_2, c_equation_2)

    def test_b_vector(self):
        # test b_vectors
        b_vector_1 = self.constraint_1.b(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        b_vector_2 = self.constraint_2.b(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        b_desired = speye(4, 4)
        assert_array_equal(b_vector_1.todense(), b_desired.todense())
        assert_array_equal(b_vector_2.todense(), b_desired.todense())

    def test_slave_dofs(self):
        # test slave_dofs
        slave_dofs_desired = self.dofs
        assert_array_equal(self.constraint_1.slave_dofs(self.dofs), slave_dofs_desired)
        assert_array_equal(self.constraint_2.slave_dofs(self.dofs), slave_dofs_desired)

    def test_u_slave(self):
        # test u_slave
        u_s1 = self.constraint_1.u_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        u_s2 = self.constraint_2.u_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        u_s1_desired = np.array([0, 0, 0, 0], dtype=float)
        u_s2_desired = np.array([4, 4, 4, 4], dtype=float)
        assert_array_equal(u_s1, u_s1_desired)
        assert_array_equal(u_s2, u_s2_desired)

    def test_du_slave(self):
        # test u_slave
        du_s1 = self.constraint_1.du_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        du_s2 = self.constraint_2.du_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        du_s1_desired = np.array([0, 0, 0, 0], dtype=float)
        du_s2_desired = np.array([6, 6, 6, 6], dtype=float)
        assert_array_equal(du_s1, du_s1_desired)
        assert_array_equal(du_s2, du_s2_desired)

    def test_ddu_slave(self):
        # test u_slave
        ddu_s1 = self.constraint_1.ddu_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        ddu_s2 = self.constraint_2.ddu_slave(self.X_local, self.u_local, self.du_local, self.ddu_local, self.t)
        ddu_s1_desired = np.array([0, 0, 0, 0], dtype=float)
        ddu_s2_desired = np.array([2, 2, 2, 2], dtype=float)
        assert_array_equal(ddu_s1, ddu_s1_desired)
        assert_array_equal(ddu_s2, ddu_s2_desired)
