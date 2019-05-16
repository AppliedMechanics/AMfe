#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Tests for nonlinear solver module
"""

from unittest import TestCase
import numpy as np
from numpy.testing import assert_

from amfe.solver.nonlinear_solver import NewtonRaphson


# The following function F for testing nonlinear solvers is similar to
# a test from the Scipy Package, distributed under BSD-3-License
# Copyright (c) 2003-2019 SciPy Developers.
# All rights reserved.
def F(x):
    x = np.asmatrix(x).T
    d = np.matrix(np.diag([3, 2, 1.5, 1, 0.5]))
    c = 0.01
    f = -d*x - c*float(x.T*x)*x
    return f
# End of Scipy function


def jac(x):
    d = np.matrix(np.diag([3, 2, 1.5, 1, 0.5]))
    c = 0.01
    j = -d - c*float()
    return j


class NonlinearTest(TestCase):
    def test_newton_raphson_solver(self):
        solver = NewtonRaphson()
        x0 = np.array([1,1,1,1,1], dtype=float)
        atol = 1e-6
        options = {'maxiter': 200, 'atol': atol}
        x, _ = solver.solve(F, x0, (), jac, options=options)
        assert_(np.absolute(F(x)).max() < atol)
