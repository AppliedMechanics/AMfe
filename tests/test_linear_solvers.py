import unittest
from unittest import TestCase
import scipy.sparse.linalg
import numpy.linalg
import numpy

from amfe.linalg.linearsolvers import PardisoSolver, ScipySparseSolver


class TestPardisoSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[0.3, 0.4, 0, 0], [0.4, 1.0 , 0.3, 0], [0, 0.8, 0.9, -0.5], [0, 0, -0.4, 0.8]])
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = PardisoSolver(A)
        x1 = solver.solve(b)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        self.assertLess(res, 10 ** (-13))


class TestScipySparseSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[0.3, 0.4, 0, 0], [0.4, 1.0 , 0.3, 0], [0, 0.8, 0.9, -0.5], [0, 0, -0.4, 0.8]])
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = ScipySparseSolver(A)
        x1 = solver.solve(b)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        self.assertLess(res, 10 ** (-13))

if __name__ == '__main__':
    unittest.main()
