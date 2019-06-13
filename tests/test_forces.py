# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from unittest import TestCase
import numpy as np

from amfe.forces import *


class TestForces(TestCase):
    def setUp(self):
        self.f_max = 2

    def tearDown(self):
        pass

    def test_constant_force(self):
        t_test = np.array([0, 0.35, 1, 1.68])

        F = constant_force(self.f_max)

        for t in t_test:
            force_actual = F(t)

            self.assertEqual(force_actual, self.f_max)

    def test_linearly_increasing_force(self):
        t_test = np.array([0, 0.5, 0.99, 1, 1.01, 1.25, 1.75, 2, 2.1])
        force_desired = np.array([0, 0, 0, 0, 0.02, 0.5, 1.5, 2, 2])

        F = linearly_increasing_force(1, 2, self.f_max)

        for idx, t in enumerate(t_test):
            force_actual = F(t)

            self.assertAlmostEqual(force_actual, force_desired[idx])

    def test_triangular_force(self):
        t_test = np.array([0, 0.5, 0.99, 1, 1.01, 1.25, 1.75, 2, 2.1])
        force_desired = np.array([0, 1, 1.98, 2, 1.98, 1.5, 0.5, 0, 0])

        F = triangular_force(0, 1, 2, self.f_max)

        for idx, t in enumerate(t_test):
            force_actual = F(t)

            self.assertEqual(force_actual, force_desired[idx])

    def test_step_force(self):
        t_test = np.array([0, 0.99, 1, 1.01, 1.5, 1.738, 2, 2.01, 2.1])
        force_desired = np.array([0, 0, 2, 2, 2, 2, 2, 0, 0], dtype=float)

        F = step_force(1.0, self.f_max, 2.0)

        for idx, t in enumerate(t_test):
            force_actual = F(t)

            self.assertEqual(force_actual, force_desired[idx])
