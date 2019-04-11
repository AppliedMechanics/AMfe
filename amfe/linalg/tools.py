# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from numpy import tensordot, array_equal
from scipy.linalg import inv
from scipy.sparse import issparse


__all__ = ['coordinate_transform',
           'isboolean']


def coordinate_transform(to, h_from):
    h__ij = h_from.T @ h_from
    h_ij = inv(h__ij)
    A = h_ij @ (h_from.T @ to)
    Ainv = inv(A)

    def transform(x):
        return tensordot(A, x, (A.ndim-1, 0))

    def inverse_transform(x):
        return tensordot(Ainv, x, (Ainv.ndim-1, 0))

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
        if not array_equal(A.data, A.data.astype(bool)):
            return False
    else:
        if not array_equal(A, A.astype(bool)):
            return False
    return True
