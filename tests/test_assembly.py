# -*- coding: utf-8 -*-
"""Test Routine for assembly"""

import numpy as np
import scipy as sp

# make amfe running
import sys
sys.path.insert(0,'..')
import amfe

from numpy.testing import assert_equal, assert_almost_equal




def test_index_getter():
    N = int(1E3)
    row = sp.random.randint(0, N, N)
    col = sp.random.randint(0, N, N)
    val = sp.rand(N)
    A = sp.sparse.csr_matrix((val, (row, col)))
    # print('row:', row, '\ncol:', col)
    for i in range(N):
        a = amfe.get_index_of_csr_data(row[i], col[i], A.indptr, A.indices)
        b = A[row[i], col[i]]
        #    print(val[i] - A.data[b])
        assert_equal(A.data[a], b)

