"""
Test routine to test some of the model reduction routines
"""

import numpy as np
import scipy as sp
import nose
from amfe import modal_assurance, principal_angles


def test_mac_diag_ones():
    '''
    Test mac criterion for getting ones on the diagonal, if the same matrix is
    given.
    '''
    N = 100
    n = 10
    A = np.random.rand(N, n)
    macvals = modal_assurance(A, A)
    np.testing.assert_allclose(np.diag(macvals), np.ones(n))

def test_mac_symmetric():
    '''
    '''
    N = 100
    n = 10
    A = np.random.rand(N, n)
    macvals = modal_assurance(A, A)
    result = macvals - macvals.T
    np.testing.assert_allclose(result, np.zeros((n, n)))

def test_mac_identity():
    N = 100
    n = 10
    A = np.random.rand(N, n)
    Q, __ = sp.linalg.qr(A, mode='economic')
    macvals = modal_assurance(Q, Q)
    np.testing.assert_allclose(macvals, np.eye(n), atol=1E-14)

def test_principal_angles():
    n_vec = 3
    n_dim = 5
    n_overlap = 2*n_vec - n_dim

    A = np.random.rand(n_dim, n_vec)
    B = np.random.rand(n_dim, n_vec)

    gamma, F1, F2 = principal_angles(A, B, principal_vectors=True)

    # test orthogonality of F1
    np.testing.assert_almost_equal(F1[:,:n_overlap].T @ F1[:,:n_overlap],
                                   np.eye(n_overlap))
    # test orthogonality of F2
    np.testing.assert_almost_equal(F2[:,:n_overlap].T @ F2[:,:n_overlap],
                                   np.eye(n_overlap))
    # test equality of F1 and F2 in the intersecting subspace
    np.testing.assert_almost_equal(F2[:,:n_overlap].T @ F1[:,:n_overlap],
                                   np.eye(n_overlap))
    # test principal angle cosines of intersecting subspace
    np.testing.assert_almost_equal(gamma[:n_overlap], np.ones(n_overlap))

