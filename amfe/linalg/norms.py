# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains vector, matrix, signal and system norms
"""


import numpy as np


__all__ = [
    'signal_norm'
]


def signal_norm(x, t=None, dt=1.0, ord=None):
    if ord is 1:
        norm_x = np.trapz(np.abs(x), t, dt)
    elif (ord is None) or (ord is 2):
        norm_x = np.sqrt(np.trapz(x**2, t, dt))
    elif ord is 'inf':
        norm_x = np.max(np.abs(x))
    else:
        try:
            ord + 1
        except TypeError:
            raise ValueError('Error: Invalid norm order for signals.')
        norm_x = np.trapz(np.abs(x)**ord, t, dt)**(1/ord)

    return norm_x

