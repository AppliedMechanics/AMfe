# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
from scipy.linalg import inv, svdvals, qr, norm
from scipy.sparse import issparse


__all__ = ['coordinate_transform',
           'isboolean',
           'arnoldi',
           ]


def coordinate_transform(to, h_from):
    h__ij = h_from.T @ h_from
    h_ij = inv(h__ij)
    A = h_ij @ (h_from.T @ to)
    Ainv = inv(A)

    def transform(x):
        return np.tensordot(A, x, (A.ndim-1, 0))

    def inverse_transform(x):
        return np.tensordot(Ainv, x, (Ainv.ndim-1, 0))

    return transform, inverse_transform


def isboolean(A):
    """
    Checks if ndarray or sparse matrix A is Boolean

    Parameters
    ----------
    A : ndarray or csr_matrix or csc_matrix or lil_matrix
        matrix to check

    Returns
    -------
    flag : bool
        Flag if A is boolean. Returns True or False
    """
    if issparse(A):
        if not np.array_equal(A.data, A.data.astype(bool)):
            return False
    else:
        if not np.array_equal(A, A.astype(bool)):
            return False
    return True


def arnoldi(A, B, n, Vout=None, orthogonal=True):
    """
    Computes a Krylov subspace K(A, B) = span{B, AB, AAB, AAAB, ...}

    Parameters
    ----------
    A: LinearOperator, array_like
        Matrix or linear operator describing A
    B: array_like
        Matrix or linear operator describing B
    n: int
        integer n describing the number of Krylov vectors by the relation
        n*B.shape[1]
    Vout: array_like or None
        array where Krylov subspace shall be written to
    orthogonal: bool
        Flag if the resulting vectors shall be orthogonolalized or just normalized

    Returns
    -------
    V: numpy.array
        array containing the Krylov vectors
    """
    ndim = A.shape[0]
    no_of_inputs = B.size//ndim
    f = B.copy()
    if Vout is None:
        V = np.zeros((ndim, n * no_of_inputs))
    else:
        if Vout.shape[0] != ndim or Vout.shape[1] < n*no_of_inputs:
            raise ValueError('Dimension of Vout is wrong')
        V = Vout

    V[:, :no_of_inputs] = B.reshape((-1, no_of_inputs))
    for i in np.arange(n-1):
        b_new = A.dot(f)
        V[:, (i+1)*no_of_inputs:(i+2)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        if orthogonal:
            V[:, :(i+2) * no_of_inputs], R = qr(V[:, :(i + 2)*no_of_inputs], mode='economic')
            b_new = V[:, (i+1)*no_of_inputs:(i + 2)*no_of_inputs]
        else:
            b_new /= norm(b_new)
            V[:, (i+1)*no_of_inputs:(i+2)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        f = b_new
    sigmas = svdvals(V)

    print('Krylov Basis constructed. The singular values of the basis are', sigmas)

    return V
