#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module for updating bases
"""

import numpy as np
import scipy as sp
from scipy.linalg import eigh, svd

from amfe.linalg.orth import m_orthogonalize

__all__ = ['ifpks',
           'ifpks_modified',
           'update_modes',
           ]


def update_modes(K, M, V_old, Kinv_operator, r=2, tol=1e-6, verbose=False, modified_ifpks=True):
    """
    Shortcut to update modes with Inverse Free Preconditioned Krylov Subspace Method

    Parameters
    ----------
    K : array_like
        stiffness matrix
    M : array_like
        mass matrix
    V_old : numpy.ndarray
        Matrix with old eigenmodes as column vectors
    Kinv_operator : LinearOperator
        Linear Operator solving K_old x = b for a similar stiffness matrix K_old
        This should be a cheap operation e.g. from a direct solver, K_old already factorized
    r : int
        Number of Krylov search directions
    tol : float
        desired tolerance for the squared eigenfrequency omega**2
    verbose : bool
        Flag if verbose version shall be used
    modified_ifpks : bool (default: True)
        Flag if modified version of ifpks shall be used (recommended)

    Returns
    -------
    omega : list
        list with new omega values
    X : numpy.ndarray
        matrix with eigenmodes as column vectors
    """
    if modified_ifpks:
        rho, V = ifpks_modified(K, M, Kinv_operator, V_old, r, tol,
                                verbose=verbose)
    else:
        rho, V = ifpks(K, M, Kinv_operator, V_old, r, tol,
                       verbose=verbose)
    return np.sqrt(rho), V


def ifpks(K, M, P, X_0, r=2, tol=1e-6, verbose=False, m_orth=m_orthogonalize):
    """
    Inverse Free Preconditioned Krylov Subspace Method according to [Voormeeren2013]_

    This method can be helpful to update Modes if some parameters of the model have changed slightly

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix of the current problem
    M : sparse_matrix
        Mass matrix of the current problem
    P : LinearOperator
        Linear Operator for Preconditioning
        e.g. solving K_old x = b for a similar stiffness matrix K_old
        This should be a cheap operation e.g. from a direct solver, K_old already factorized
    X_0 : numpy.ndarray
        Matrix with Eigenmodes of the old/reference problem
    r : int
        number of Krylov search directions
    tol : float
        desired tolerance for the squared eigenfrequency omega**2
    verbose : bool
        Flag if verbose version shall be used
    m_orth : function
        Pointer to orthogonalization scheme

    Returns
    -------
    rho : list
        list with new omega**2 values
    X : numpy.ndarray
        matrix with eigenmodes as column vectors

    References
    ----------
    .. [Voormeeren2013] Sven Voormeeren, Daniel Rixen,
       "Updating component reduction bases of static and vibration modes using preconditioned iterative techniques",
       Computer Methods in Applied Mechanics and Engineering, Volume 253, 2013, Pages 39-59.

    """
    X = X_0
    no_of_modes = X.shape[1]
    X = m_orth(X, M)
    Zm = np.zeros((X.shape[0], X.shape[1] * (r + 1)))
    rho = list()
    k = 0
    rho.append(np.diag(X.T @ K @ X))
    if verbose:
        print('IFPKS Iteration No. {}, rho: {}'.format(k, rho[k]))
    while k == 0 or np.linalg.norm(rho[k] - rho[k - 1]) > tol * np.linalg.norm(rho[k]):
        # Generate Krylov Subspace
        rho.append(np.Inf)
        # Zm[:,0:no_of_modes] = m_orth(X, M)
        # stattdessen:
        Zm[:, 0:no_of_modes] = X
        # no_of_orth = no_of_modes
        for j in np.arange(r):
            Zm[:, no_of_modes * (j + 1):no_of_modes * (j + 2)] = P.dot(
                K @ Zm[:, no_of_modes * (j):no_of_modes * (j + 1)] - M @ Zm[:,
                       no_of_modes * (j):no_of_modes * (j + 1)] * rho[
                       k])
            Zm[:, :no_of_modes * (j + 2)] = m_orth(Zm[:, :no_of_modes * (j + 2)], M)
        Kr = Zm.T @ K @ Zm
        lam, V = eigh(Kr)
        V = m_orth(V[:, :no_of_modes])
        X = Zm @ V
        X = m_orth(X, M)
        rho[k + 1] = np.diag(X.T @ K @ X)
        k = k + 1
        print('IFPKS Iteration No. {}, rho: {}'.format(k, rho[k]))
    return rho[-1], X


def ifpks_modified(K, M, P, X_0, r=2, tol=1e-6, verbose=False, m_orth=m_orthogonalize):
    """
    Modified Inverse Free Preconditioned Krylov Subspace Method

    This method can be helpful to update Modes if some parameters of the model have changed slightly
    It is slightly modified compared to [Voormeeren2013]_.
    Instead M-orthogonalize all Krylov vectors, a Orthogonalization and truncation using an SVD is used instead.
    This is more stable as the Krylov vectors can be almost linear dependent.
    Additionally a M-orthogonalization would slow down the algorithm significantly.

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix of the current problem
    M : sparse_matrix
        Mass matrix of the current problem
    P : LinearOperator
        Linear Operator for Preconditioning
        e.g. solving K_old x = b for a similar stiffness matrix K_old
        This should be a cheap operation e.g. from a direct solver, K_old already factorized
    X_0 : numpy.ndarray
        Matrix with Eigenmodes of the old/reference problem
    r : int
        number of Krylov search directions
    tol : float
        desired tolerance for the squared eigenfrequency omega**2
    verbose : bool
        Flag if verbose version shall be used
    m_orth : function
        Pointer to orthogonalization scheme

    Returns
    -------
    rho : list
        list with new omega**2 values
    X : numpy.ndarray
        matrix with eigenmodes as column vectors

    References
    ----------
    .. [Voormeeren2013] Sven Voormeeren, Daniel Rixen,
       "Updating component reduction bases of static and vibration modes using preconditioned iterative techniques",
       Computer Methods in Applied Mechanics and Engineering, Volume 253, 2013, Pages 39-59.

    """
    X = X_0
    no_of_modes = X.shape[1]
    X = m_orth(X, M)
    Zm = np.zeros((X.shape[0], X.shape[1] * (r + 1)))
    rho = list()
    k = 0
    rho.append(np.diag(X.T @ K @ X))
    if verbose:
        print('IFPKS Iteration No. {}, rho: {}'.format(k, rho[k]))
    while k == 0 or np.linalg.norm(rho[k] - rho[k - 1]) > tol * np.linalg.norm(rho[k]):
        # Generate Krylov Subspace
        rho.append(np.Inf)
        # Zm[:,0:no_of_modes] = m_orth(X, M)
        # stattdessen:
        Zm[:, 0:no_of_modes] = X
        # no_of_orth = no_of_modes
        for j in np.arange(r):
            Zm[:, no_of_modes * (j + 1):no_of_modes * (j + 2)] = P.solve(
                K.dot(Zm[:, no_of_modes * (j):no_of_modes * (j + 1)]) - M.dot(Zm[:,
                                                                         no_of_modes * (j):no_of_modes * (j + 1)]) * rho[
                    k])
            Zm[:, :no_of_modes * (j + 2)] = m_orth(Zm[:, :no_of_modes * (j + 2)], M)
            # Facebook svd
            # Zm[:,:no_of_modes*(j+2)],s,_ = pca(Zm[:,:no_of_modes*(j+2)],no_of_modes*(j+2), raw=True)
            # Scipy svd
            Zm[:, :no_of_modes * (j + 2)], s, _ = svd(Zm[:, :no_of_modes * (j + 2)], full_matrices=False)
        Kr = Zm[:, s > 1e-8].T.dot(K).dot(Zm[:, s > 1e-8])
        Mr = Zm[:, s > 1e-8].T.dot(M).dot(Zm[:, s > 1e-8])
        lam, V = eigh(Kr, Mr)
        X = Zm[:, s > 1e-8].dot(V)
        X = m_orth(X[:, :no_of_modes], M)
        rho[k + 1] = np.diag(X.T.dot(K).dot(X))
        k = k + 1
        print('IFPKS Iteration No. {}, rho: {}'.format(k, rho[k]))
    return rho[-1], X


def update_static_derivatives(V, K_func, Kinv_operator, Theta_0, M=None, omega=0.0, h=1.0, verbose=False, symmetric=True,
                              finite_diff='central'):
    """
    Update the static correction derivatives for the given basis V.

    Optionally, a frequency shift can be performed.

    Parameters
    ----------
    V : ndarray
        array containing the linear basis
    K_func : function
        function returning the tangential stiffness matrix for a given
        displacement. Has to work like `K = K_func(u)`.
    Kinv_operator : LinearOperator
        Linear Operator solving K_old x = b for a similar stiffness matrix K_old
        This should be a cheap operation e.g. from a direct solver, K_old already factorized
    M : ndarray, optional
        mass matrix. Can be sparse or dense. If `None` is given, the mass of 0
        is assumed. Default value is `None`.
    omega : float, optional
        shift frequency. Default value is 0.
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True
    finite_diff : str {'central', 'forward', backward}
        Method for finite difference scheme. 'central' computes the finite
        difference based on a central difference scheme, 'forward' based on an
        forward scheme etc. Note that the upwind scheme can cause severe
        distortions of the static correction derivative.

    Returns
    -------
    Theta : ndarray
        three dimensional array of static corrections derivatives. Theta[:,i,j]
        contains the static derivative 1/2 * dx_i / dx_j. As the static
        derivatives are symmetric, Theta[:,i,j] == Theta[:,j,i].

    See Also
    --------
    static_derivatives

    """
    P = Kinv_operator
    no_of_dofs = V.shape[0]
    no_of_modes = V.shape[1]
    Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes))
    K = K_func(np.zeros(no_of_dofs))
    if (omega > 0) and (M is not None):
        K_dyn = K - omega ** 2 * M
    else:
        K_dyn = K
    for i in range(no_of_modes):
        if verbose:
            print('Computing finite difference K-matrix')
        if finite_diff == 'central':
            dK_dx_i = (K_func(h * V[:, i]) - K_func(-h * V[:, i])) / (2 * h)
        elif finite_diff == 'forward':
            dK_dx_i = (K_func(h * V[:, i]) - K) / h
        elif finite_diff == 'backward':
            dK_dx_i = (-K_func(-h * V[:, i]) + K) / h
        else:
            raise ValueError('Finite difference scheme is not valid.')
        b = - dK_dx_i @ V
        if verbose:
            print('Solving linear system #', i)
        for j in range(b.shape[1]):
            Theta[:, j, i] = sp.sparse.linalg.cg(K_dyn, b[:, j], x0=Theta_0[:, j, i], M=P)[0]
        if verbose:
            print('Done solving linear system #', i)
    if verbose:
        residual = np.linalg.norm(Theta - Theta.transpose(0, 2, 1)) / \
                   np.linalg.norm(Theta)
        print('The residual, i.e. the unsymmetric values, are', residual)
    if symmetric:
        # make Theta symmetric
        Theta = 1 / 2 * (Theta + Theta.transpose(0, 2, 1))
    return Theta
