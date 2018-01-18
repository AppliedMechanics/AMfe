import unittest
from unittest import TestCase
import scipy.sparse.linalg
import numpy.linalg
import numpy

from amfe.linalg.linearsolvers import PardisoSolver


class TestPardisoSolver(TestCase):
    def test_solve(self):
        A = numpy.array([[0.3, 0.4, 0, 0],[0.4,1 , 0.3, 0],[0, 0.8, 0.9, -0.5],[0, 0, -0.4, 0.8]])
        A = scipy.sparse.csr_matrix(A)
        b = numpy.array([1.4, 1.2, 0.8, 1.1])
        solver = PardisoSolver(A)
        x1 = solver.solve(b)
        x2 = scipy.sparse.linalg.spsolve(A, b)
        res1 = numpy.linalg.norm(A.dot(x1) - b)
        res2 = scipy.sparse.linalg.norm(A)
        print(x1)
        print(x2)
        print(res1)
        print(res2)
        res = numpy.linalg.norm(A.dot(x1) - b) / scipy.sparse.linalg.norm(A)
        del solver
        del x1
        self.assertLess(res, 10 ** (-13))

if __name__ == '__main__':
    unittest.main()
