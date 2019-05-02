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


def m_orthogonalize(Vin, M, Vout=None, niter=1):
    """
    Returns M-orthonormalized vectors
    
    Parameters
    ----------
    Vin : array_like
        Matrix with column vectors that shall be M-orthogonalized
    M : array_like
        Matrix for orthogonalization metric such that V.T @ M @ V = I
    Vout : array_like
        Matrix that can be filled with the orthogonoalized values,
        if None, Vin will be overwritten and used as output
    niter : int
        Number of Gram-Schmid runs for the orthogonalization. As the
        Gram-Schmid-procedure is not stable, more then one iteration are
        recommended.
    Returns
    -------
    Vn : numpy.ndarray
        Matrix with M-orthonormalized column vectors
    """
    if Vout is None:
        Vout = Vin
    else:
        Vout[:, :] = Vin[:, :]

    for iteration in range(niter):
        for i in range(Vin.shape[1]-1):
            v = Vout[:, i]
            v = v/np.sqrt(v.dot(M).dot(v))
            Vout[:, i] = v
            weights = v.dot(M).dot(Vout[:, i + 1:])
            Vout[:, i + 1:] -= v.reshape((-1, 1)) * weights
        Vout[:, -1] = Vout[:, -1]/np.sqrt(Vout[:, -1].T.dot(M).dot(Vout[:, -1]))
    return Vout
