# -*- coding: utf-8 -*-
'''
Methods for the Quadratic Manifold approach.
'''

import numpy as np


__all__ = ['theta_orth_v']


def theta_orth_v(Theta, V, M, overwrite=False):
    '''
    Make third order tensor Theta fully mass orthogonal with respect to the
    basis V via a Gram-Schmid-process.

    Parameters
    ----------
    Theta : ndarray
        Third order Tensor describing the quadratic part of the basis
    V : ndarray
        Linear Basis
    M : ndarray or scipy.sparse matrix
        Mass Matrix
    overwrite : bool
        Flag for setting, if Theta should be overwritten in-place

    Returns
    -------
    Theta_orth : ndarray
        Third order tensor Theta mass orthogonalized, such that
        Theta_orth[:,i,j] is mass orthogonal to V[:,k]:
        :math:`\\theta_{ij}^T M V = 0`

    '''
    # Make sure, that V is M-orthogonal
    __, no_of_modes = V.shape
    V_M_space = M @ V
    np.testing.assert_allclose(V.T @ V_M_space, np.eye(no_of_modes), atol=1E-14)
    if overwrite:
        Theta_ret = Theta
    else:
        Theta_ret = Theta.copy()
    # inner product of Theta[:,j,k] with V[:,l] in the M-norm
    inner_prod = np.einsum('ijk, il -> jkl', Theta, V_M_space)
    for j in range(no_of_modes):
        for k in range(no_of_modes):
            Theta_ret[:,j,k] -= V @ inner_prod[j,k,:]
    return Theta_ret

