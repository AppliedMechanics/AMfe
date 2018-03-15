# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains orthogonalizations
"""

import numpy as np


__all__ = [
    'm_orthogonalize',
]


def m_orthogonalize(V, M):
    """
    Returns M-orthonormalized vectors
    
    Parameters
    ----------
    V : numpy.ndarray
        Matrix with column vectors that shall be M-orthogonalized
    M : numpy.ndarray
        Matrix for orthogonalization metric such that V.T @ M @ V = I
    
    Returns
    -------
    Vn : numpy.ndarray
        Matrix with M-orthonormalized column vectors
    """
    for i in range(V.shape[1]-1):
        v = V[:, i]
        v = v/np.sqrt(v @ M @ v)
        V[:, i] = v
        weights = v @ M @ V[:, i + 1:]
        V[:, i + 1:] -= v.reshape((-1, 1)) * weights
    V[:,-1] = V[:,-1]/np.sqrt(V[:,-1].T @ M @ V[:,-1])
    return V
