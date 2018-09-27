# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from numpy import tensordot
from scipy.linalg import inv


__all__ = ['coordinate_transform']


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
