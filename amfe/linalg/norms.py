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
    'm_normalize',
    'vector_norm',
    'matrix_norm',
    'signal_norm',
    'lti_system_norm'
]


# shortcut for vector norm to numpy.linalg's norm
vector_norm = np.linalg.norm

# shortcut for matrix norm to numpy.linalg's norm
matrix_norm = np.linalg.norm


def m_normalize(X, M):
    """
    Returns M-normalized vectors X
    
    Parameters
    ----------
    X : numpy.ndarray
        Matrix with column vectors to M-normalize
    M : numpy.ndarray
        Matrix for normalization x.T @ M @ x = 1 
    
    Returns
    -------
    Xn : numpy.ndarray
        Returns M-normalized vectors
    """
    if M is not None:
        if X.ndim == 1:
            n = np.sqrt(X.T@M@X)
        else:
            n = np.sqrt(np.diag(X.T@M@X))
    else:
        if X.ndim == 1:
            n = np.sqrt(X.T@X)
        else:
            n = np.sqrt(np.diag(X.T@X))
    return X/n


def signal_norm(x, t=None, dt=1.0, ord=2, axis=-1):
    """
    Returns signal norm ||x(t)||.
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
    ord : {non-zero int or float, inf, -inf, 2, 1, 'mean', 'rms', 'max', 'min'}, optional
        Order of the norm (see table under Notes). inf means numpy's inf object. Default is 2.
    axis : int, optional
        Axis along which to integrate in norm. Default -1.

    Returns
    -------
    norm_x : float or ndarray
        Norm of the signal(s).

    Notes
    -----
    For values of ord < 0 and ord in {'mean', 'rms', 'max', 'min'} the result is - strictly speaking - not a
    mathematical 'norm', but it may still be useful for various numerical purposes.
    The following norms can be calculated:
    ======  =========================================
    ord     norm for signals
    ======  =========================================
    inf     L_inf-norm: sup(abs(x)) in [t_0,t_end]
    > 0     L_ord-norm: (integral(abs(x)^ord)^(1/ord)
    2       L_2-norm: sqrt(integral(abs(x)^2))
    1       L_1-norm: integral(abs(x))
    < 0     (integral(abs(x)^ord)^(1/ord)
    -inf    inf(abs(x)) in [t_0,t_end]
    'mean'  mean value: 1/T*integral(x)
    'rms'   rms value: sqrt(1/T*integral(x^2))
    'max'   maximal value: max(x) in [t_0,t_end]
    'min'   minimal value: min(x) in [t_0,t_end]
    ======  =========================================
    """

    if ord == 2:
        norm_x = np.sqrt(np.trapz(y=x**2, x=t, dx=dt, axis=axis))
    elif ord == 1:
        norm_x = np.trapz(y=np.abs(x), x=t, dx=dt, axis=axis)
    elif ord == np.inf:
        norm_x = np.amax(a=np.abs(x), axis=axis)
    elif ord == -np.inf:
        norm_x = np.amin(a=np.abs(x), axis=axis)
    elif ord == 'mean':
        if t is not None:
            T = t[-1] - t[0]
        else:
            T = (np.ma.size(obj=x, axis=axis) - 1)*dt
        norm_x = np.trapz(y=x, x=t, dx=dt, axis=axis)/T
    elif ord == 'rms':
        if t is not None:
            T = t[-1] - t[0]
        else:
            T = (np.ma.size(obj=x, axis=axis) - 1)*dt
        norm_x = np.sqrt(np.trapz(y=x**2, x=t, dx=dt, axis=axis)/T)
    elif ord == 'max':
        norm_x = np.amax(a=x, axis=axis)
    elif ord == 'min':
        norm_x = np.amin(a=x, axis=axis)
    else:
        try:
            ord + 1
        except TypeError:
            raise ValueError('Invalid norm order for signals.')
        if ord == 0:
            raise ValueError('Invalid norm order for signals.')
        norm_x = np.trapz(y=np.abs(x)**(1.0*ord), x=t, dx=dt, axis=axis)**(1/ord)
    return norm_x

def lti_system_norm(A, B, C, E=None, ord=2, **kwargs):
    """
    Returns norm ||G(s)|| of LTI system (E,A,B,C,D=0) or (A,B,C,D=0).
    Function is able to return one of the LTI system norms (described below) depending on the value of the parameter
    ord.

    Parameters
    ----------
    E : 2darray
        Descriptor matrix of LTI system.
    A : 2darray
        Dynamic matrix of LTI system.
    B : 1darray or 2darray
        Input vector/matrix of LTI system. 1darray will be automatically broadcasted to appropriate 2darray.
    C : 1darray or 2darray
        Output vector/matrix of LTI system. 1darray will be automatically broadcasted to appropriate 2darray.
    ord : {inf, 2}, optional
        Order of the norm (see table under Notes). inf means numpy's inf object. Default is 2.
    **kwargs : additional arguments, optional
        Additional optional arguments:
        use_controllability_gramian : boolean
            Whether to use the controllability Gramian (True) or the observability Gramian (False) in the H_2-norm
            calculation. Default True, i.e. use controllability Gramian.

    Returns
    -------
    norm_sys : float
        Norm of the LTI system.

    Notes
    -----
    The following norms can be calculated:
    ===  ====================
    ord  norm for LTI systems
    ===  ====================
    inf  H_inf-norm
    2    H_2-norm
    ===  ====================
    """

    if ord == 2:
        # convert to and prepare system (A, B, C)
        if E is not None:
            A = sp.sparse.linalg.spsolve(E, A)
            B = sp.sparse.linalg.spsolve(E, B)
        if B.ndim == 1:
            B = B.reshape((-1, 1))
        if C.ndim == 1:
            C = C.reshape((1, -1))

        # read kwargs
        if 'use_controllability_gramian' in kwargs:
            use_controllability_gramian = kwargs['use_controllability_gramian']
        else:
            print('Attention: No instruction which Gramian to use was given, setting ' \
                  + 'use_controllability_gramian = True.')

        if use_controllability_gramian: # via controllability Gramian
            # TODO: Find/implement sparse solver for lyapunov equations.
            G_c = sp.linalg.solve_continuous_lyapunov(A.todense(), -B@B.T)
            norm_sys = np.sqrt(np.trace(C@G_c@C.T))

        if not use_controllability_gramian: # via observability Gramian
            # TODO: Find/implement sparse solver for lyapunov equations.
            G_o = sp.linalg.solve_continuous_lyapunov(A.T.todense(), -C.T@C)
            norm_sys = np.sqrt(np.trace(B.T@G_o@B))
    elif ord == np.inf:
        raise ValueError('Not implemented yet. You may do so.')
    else:
        raise ValueError('Invalid norm order for LTI systems.')
    return norm_sys

