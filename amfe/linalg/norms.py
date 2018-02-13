# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module contains all types of norms
"""


import numpy as np
import scipy as sp


__all__ = [
    'signal_norm',
    'lti_system_norm'
]


def signal_norm(x, t=None, dt=1.0, ord=2, axis=-1):
    """
    Signal norm ||x(t)||_ord.
    Function is able to return one of an infinite number of signal norms (described below) depending on the value of
    the parameter ord. The function uses numpy's amax (maximum of array) for finding the supremum, amin (minimum of
    array) for finding infimum or trapz (composite trapezoidal rule) for time integration.

    Parameters
    ----------
    x : array_like
        Input array containing samples of the signal(s).
    t : array_like, optional
        Input array containing time samples corresponding to x. If t is None, samples x are assumed to be evenly spaced
        with dt. Default is None.
    dt : float, optional
        Spacing between samples x when t is None. Default is 1.
    ord : {non-zero int or float, inf, -inf, 2, 1}, optional
        Order of the norm (see table under Notes). inf means numpy's inf object. Default is 2.
    axis : int, optional
        Axis along which to integrate in norm. Default -1.

    Returns
    -------
    norm_x : float or ndarray
        Norm of the signal(s).

    Notes
    -----
    For values of ord < 0 the result is - strictly speaking - not a mathematical 'norm', but it may still be useful for
    various numerical purposes
    The following norms can be calculated:
    ====  ========================================
    ord   norm for signals
    ====  ========================================
    inf   L_inf-norm sup(abs(x))
    > 0   L_ord-norm (integral(abs(x)^ord)^(1/ord)
    2     L_2-norm integral(abs(x))
    1     L_1-norm sqrt(integral(abs(x)^2))
    < 0   (integral(abs(x)^ord)^(1/ord)
    -inf  inf(abs(x))
    ====  ========================================
    """

    if ord == 1:
        norm_x = np.trapz(y=np.abs(x), x=t, dx=dt, axis=axis)
    elif ord == 2:
        norm_x = np.sqrt(np.trapz(y=x**2, x=t, dx=dt, axis=axis))
    elif ord == np.inf:
        norm_x = np.amax(a=np.abs(x), axis=axis)
    elif ord == -np.inf:
        norm_x = np.amin(a=np.abs(x), axis=axis)
    else:
        try:
            ord + 1
        except TypeError:
            raise ValueError('Error: Invalid norm order for signals.')
        if ord == 0:
            raise ValueError('Error: Invalid norm order for signals.')
        norm_x = np.trapz(y=np.abs(x)**(1.0*ord), x=t, dx=dt, axis=axis)**(1/ord)
    return norm_x


def lti_system_norm(A, B, C, E=None, ord=2):
    if ord == 2:
        if E is not None:
            A = sp.sparse.linalg.spsolve(E, A)
            B = sp.sparse.linalg.spsolve(E, B)
        if len(B.shape) == 1:
            B = B.reshape((-1, 1))
        if len(C.shape) == 1:
            C = C.reshape((1, -1))
        G_c = sp.linalg.solve_continuous_lyapunov(A.todense(), -B@B.T)
        norm_sys_c = np.sqrt(np.trace(C@G_c@C.T))
        G_o = sp.linalg.solve_continuous_lyapunov(A.todense().T, -C.T@C)
        norm_sys_o = np.sqrt(np.trace(B.T@G_o@B))
        print(norm_sys_c)
        print(norm_sys_o)
        print((norm_sys_c - norm_sys_o)/norm_sys_c)
        if abs((norm_sys_o - norm_sys_c)/norm_sys_c) < 1.0e-6:
            norm_sys = norm_sys_c
        else:
            raise ValueError('Error: H_2-norm calculation failed.')
    elif ord == np.inf:
        raise ValueError('Error: Not implemented yet. You may do so.')
    else:
        raise ValueError('Error: Invalid norm order for systems.')
    return norm_sys

