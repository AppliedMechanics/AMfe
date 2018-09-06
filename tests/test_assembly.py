"""Test Routine for assembly"""


from unittest import TestCase
from scipy.sparse import csr_matrix
from numpy.random import randint
import numpy as np
from scipy import rand
from numpy.testing import assert_array_equal
from amfe.assembly.tools import get_index_of_csr_data, fill_csr_matrix


class AssemblyTest(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_index_of_csr_data(self):
        N = int(10)
        row = randint(0, N, N)
        col = randint(0, N, N)
        val = rand(N)
        A = csr_matrix((val, (row, col)))
        # print('row:', row, '\ncol:', col)
        for i in range(N):
            a = get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
            b = A[row[i], col[i]]
            #    print(val[i] - A.data[b])
            assert_array_equal(A.data[a], b)

    def test_fill_csr_matrix(self):
        # Test: Build matrix
        #           -                 -     -                 -
        #           | 2.0   0.0   3.0 |     | 1.0   0.0   1.0 |
        #           | 1.0   0.0   1.0 |     | 1.0   0.0   1.0 |
        #           | 4.0   1.0   5.0 |     | 1.0   1.0   1.0 |
        #           -                 -     -                 -

        k_indices = np.array([0, 2])
        K = np.array([[1.0, 2.0], [3.0, 4.0]])
        A_desired = csr_matrix(np.array([[2.0, 0.0, 3.0], [1.0, 0.0, 1.0], [4.0, 1.0, 5.0]], dtype=float))

        dummy = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        A_actual = csr_matrix(dummy)
        fill_csr_matrix(A_actual.indptr, A_actual.indices, A_actual.data, K, k_indices)

        assert_array_equal(A_actual.A, A_desired.A)
