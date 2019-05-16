"""
Test routine to test some of the model reduction routines
"""


import numpy as np
from amfe.tools import principal_angles


def test_principal_angles():
    n_vec = 3
    n_dim = 5
    n_overlap = 2*n_vec - n_dim

    A = np.random.rand(n_dim, n_vec)
    B = np.random.rand(n_dim, n_vec)

    # sine-based method
    gamma, F1, F2 = principal_angles(V1=A, V2=B, unit=None, method='sin', principal_vectors=True)
    # test orthogonality of F1
    np.testing.assert_almost_equal(F1[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test orthogonality of F2
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F2[:,:n_overlap], np.eye(n_overlap))
    # test equality of F1 and F2 in the intersecting subspace
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test principal angle sines of intersecting subspace
    np.testing.assert_almost_equal(gamma[:n_overlap], np.zeros(n_overlap))

    # cosine-based method
    gamma, F1, F2 = principal_angles(V1=A, V2=B, unit=None, method='cos', principal_vectors=True)
    # test orthogonality of F1
    np.testing.assert_almost_equal(F1[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test orthogonality of F2
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F2[:,:n_overlap], np.eye(n_overlap))
    # test equality of F1 and F2 in the intersecting subspace
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test principal angle cosines of intersecting subspace
    np.testing.assert_almost_equal(gamma[:n_overlap], np.ones(n_overlap))

    # mixed sine- and cosine-based method
    gamma, F1, F2 = principal_angles(V1=A, V2=B, unit='deg', method=None, principal_vectors=True)
    # test orthogonality of F1
    np.testing.assert_almost_equal(F1[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test orthogonality of F2
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F2[:,:n_overlap], np.eye(n_overlap))
    # test equality of F1 and F2 in the intersecting subspace
    np.testing.assert_almost_equal(F2[:,:n_overlap].T@F1[:,:n_overlap], np.eye(n_overlap))
    # test principal angles of intersecting subspace
    np.testing.assert_almost_equal(gamma[:n_overlap], np.zeros(n_overlap))

