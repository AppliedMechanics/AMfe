# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains vector, matrix, time signal and system norms
"""


import numpy as np


__all__ = [
    'euclidean_norm_of_vector'
]


def euclidean_norm_of_vector(vector):
    '''
    Computes the Euclidean norm = 2-norm of a vector.

    Parameters
    ----------
    vector : ndarray
        One dimensional array

    Returns
    -------
    norm : float
        Euclidean norm = 2-norm of given vector.

    '''

    return np.sqrt(vector.T@(vector))

