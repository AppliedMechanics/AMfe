"""Test Routine for assembly"""


from unittest import TestCase
from scipy.sparse import csr_matrix
from numpy.random import randint
from scipy import rand
from numpy.testing import assert_array_equal
from amfe.assembly.tools import get_index_of_csr_data


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
