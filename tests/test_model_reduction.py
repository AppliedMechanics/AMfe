"""
Test routine to test some of the model reduction routines
"""

import numpy as np
import scipy as sp
import nose
from amfe import mac


def test_mac_diag_ones():
    '''
    Test mac criterion for getting ones on the diagonal, if the same matrix is
    given.
    '''
    N = 100
    n = 10
    A = np.random.rand(N, n)
    macvals = mac(A, A)
    np.testing.assert_allclose(np.diag(macvals), np.ones(n))

def test_mac_symmetric():
    '''
    '''
    N = 100
    n = 10
    A = np.random.rand(N, n)
    macvals = mac(A, A)
    result = macvals - macvals.T
    np.testing.assert_allclose(result, np.zeros((n, n)))

def test_mac_identity():
    N = 100
    n = 10
    A = np.random.rand(N, n)
    Q, __ = sp.linalg.qr(A, mode='economic')
    macvals = mac(Q, Q)
    np.testing.assert_allclose(macvals, np.eye(n), atol=1E-14)
