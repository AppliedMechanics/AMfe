#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#


from unittest import TestCase
import scipy.sparse.linalg
import numpy.linalg
import numpy

from amfe.linalg.linearsolvers import *


class TestPardisoSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[4, -2, 0, 0], [-2, 4 , -2, 0], [0, -2, 4, -1], [0, 0, -1, 1]], dtype=float)
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = PardisoLinearSolver()
        x1 = solver.solve(A, b)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        
        self.assertLess(res, 10 ** (-13))


class TestScipySparseSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[4, -2, 0, 0], [-2, 4 , -2, 0], [0, -2, 4, -1], [0, 0, -1, 1]], dtype = float)
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = ScipySparseLinearSolver()
        x1 = solver.solve(A, b)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)

        self.assertLess(res, 10 ** (-13))
        
class TestScipyConjugateGradientLinearSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[4, -2, 0, 0], [-2, 4 , -2, 0], [0, -2, 4, -1], [0, 0, -1, 1]], dtype=float)
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = ScipyConjugateGradientLinearSolver()
        x1 = solver.solve(A.todense(), b, numpy.array([0, 0, 0, 0]), 1e-5, 4)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)

        self.assertLess(res, 10 ** (-5))
        
        x1 = solver.solve(A.todense(), b, numpy.array([0, 0, 0, 0]), 1e-13, 4)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)

        self.assertLess(res, 10 ** (-13))
        
        x1 = solver.solve(A.todense(), b, numpy.array([0, 0, 0, 0]), 1e-1, 4)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)

        self.assertLess(res, 10 ** (-1))
        
class TestResidualbasedConjugateGradient(TestCase):
    def test_solve(self):
        A = numpy.array([[4, -2, 0, 0], [-2, 4 , -2, 0], [0, -2, 4, -1], [0, 0, -1, 1]], dtype=float)
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = ResidualbasedConjugateGradient()
        
        def residual(x):
            return A.dot(x)-b
        
        x1, _,_ = solver.solve(residual, numpy.array([0.0, 0.0, 0.0, 0.0]), 1e-5, 4)

        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        self.assertLess(res, 10 ** (-5))
        
        x1, _,_ = solver.solve(residual, numpy.array([0.0, 0.0, 0.0, 0.0]), 1e-13, 4)

        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        self.assertLess(res, 10 ** (-13))
        
        x1, _,_ = solver.solve(residual, numpy.array([0.0, 0.0, 0.0, 0.0]), 1e-1, 4)

        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        self.assertLess(res, 10 ** (-1))
