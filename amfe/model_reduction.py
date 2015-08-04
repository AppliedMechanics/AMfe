# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:06:59 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
from scipy import linalg


sq_eps = np.sqrt(np.finfo(float).eps)

def modal_derivative(x_i, x_j, K_func, M, omega_i, h=500*sq_eps, verbose=True):
    '''
    Compute the real modal derivative of the given system using Nelson's formulation.

    The modal derivative computed is dx_i / dx_j, i.e. the change of the
    mode x_i when the system is perturbed along x_j.

    Parameters
    ----------
    x_i : ndarray
        modeshape-vector
    x_j : ndarray
        modeshape-vector
    K_func : function
        function for the tangential stiffness matrix; It is evoked by K_func(u)
        with the displacement vector u
    M : ndarray
        mass matrix
    omega_i : float
        eigenfrequency corresponding to the modeshape x_i
    h : float, optional
        step size for the computation of the finite difference scheme. Default value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.

    Returns
    -------
    dx_i / dx_j : ndarray
        The modal derivative dx_i / dx_j with mass consideration

    Note
    ----
    The the vectors x_i and x_j are internally mass normalized;

    See Also
    --------
    static_correction_derivative

    Examples
    --------
    todo

    References
    ---------
    S. R. Idelsohn and A. Cardona. A reduction method for nonlinear structural
    dynamic analysis. Computer Methods in Applied Mechanics and Engineering,
    49(3):253–279, 1985.

    S. R. Idelsohn and A. Cardona. A load-dependent basis for reduced nonlinear
    structural dynamics. Computers & Structures, 20(1):203–210, 1985.


    '''
    x_i /= np.sqrt(x_i.dot(M.dot(x_i)))
    x_j /= np.sqrt(x_j.dot(M.dot(x_j)))
#    h = np.sqrt(np.finfo(float).eps)*100 # step size length
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_x_j = (K_func(x_j*h) - K)/h
    d_omega_2_d_x_i = x_i.dot(dK_x_j.dot(x_i))
    F_i = (d_omega_2_d_x_i*M - dK_x_j).dot(x_i)
    K_dyn_i = K - omega_i**2 * M
    row_index = np.argmax(abs(x_i))
    K_dyn_i[:,row_index], K_dyn_i[row_index,:], K_dyn_i[row_index,row_index] = 0, 0, 1
    F_i[row_index] = 0
    v_i = linalg.solve(K_dyn_i, F_i)
    c_i = -v_i.dot(M.dot(x_i))
    dx_i_dx_j = v_i + c_i*x_i
    if verbose:
        print('\nComputation of modal derivatives. ')
        print('Influence of the change of the eigenfrequency:', d_omega_2_d_x_i)
        print('The condition number of the problem is', np.linalg.cond(K_dyn_i))
        res = (K - omega_i**2 * M).dot(dx_i_dx_j) - (d_omega_2_d_x_i*M - dK_x_j).dot(x_i)
        print('The residual is', np.sqrt(res.dot(res)),
              ', the relative residual is', np.sqrt(res.dot(res))/np.sqrt(F_i.dot(F_i)))
    return dx_i_dx_j



def static_correction_derivative(x_i, x_j, K_func, h=500*sq_eps, verbose=True):
    '''
    Computes the static correction vectors.

    Parameters
    ----------
    x_i : ndarray
        displacement vector i
    x_j : ndarray
        displacement vector j
    K_func : function
        function for the tangential stiffness matrix to be called in the form K_tangential = K_func(x_i)
    h : float, optional
        step size for the computation of the finite difference scheme. Default value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.

    Returns
    -------
    dx_i_dx_j : ndarray
        static correction derivative (if x_i and x_j is a modal vector it's the modal derivative neglecting mass terms) of displacement x_i with respect to displacement x_j

    Notes
    -----
    The static correction is done purely on the arrays x_i and x_j, so there is no mass normalization. This is a difference in contrast to the technique used in the related function modal_derivative.

    See Also
    --------
    modal_derivative

    '''
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_x_j = (K_func(x_j*h) - K)/h
    b = - dK_x_j.dot(x_i) # rigth hand side of equation
    dx_i_dx_j = linalg.solve(K, b)
    if verbose:
        res = K.dot(dx_i_dx_j) + dK_x_j.dot(x_i)
        print('\nComputation of static correction derivative. ')
        print('The condition number of the solution procedure is', np.linalg.cond(K))
        print('The residual is', linalg.norm(res),
              ', the relative residual is', linalg.norm(res)/linalg.norm(b))
    return dx_i_dx_j


def principal_angles_and_vectors(V1, V2, cosine=True):
    '''
    Return the cosine of the principal angles of the two bases V1 and V2.

    Parameters
    ----------
    V1 : ndarray
        array denoting n-dimensional subspace spanned by V1 (Mxn)
    V2 : ndarray
        array denoting subspace 2. Dimension is (MxO)
    cosine : bool, optional
        flag stating, if the cosine of the angles is to be used

    Returns
    -------
    sigma : ndarray
        cosine of subspace angles
    F1 : ndarray
        array of principal vectors of subspace spanned by V1. The columns give
        the principal vectors, i.e. F1[:,0] is the first principal vector
        associated with theta[0] and so on.
    F2 : ndarray
        array of principal vectors of subspace spanned by V2.

    Note
    ----
    Both matrices V1 and V2 have live in the same vector space, i.e. they have
    to have the same number of rows

    Examples
    --------
    TODO

    See Also
    --------
    principal_angles

    References
    ----------
    G. H. Golub and C. F. Van Loan. Matrix computations, volume 3. JHU Press, 2012.

    '''
    Q1, R1 = linalg.qr(V1, mode='economic')
    Q2, R2 = linalg.qr(V2, mode='economic')
    U, sigma, V = linalg.svd(Q1.T.dot(Q2))
    F1 = Q1.dot(U)
    F2 = Q2.dot(V)
    if not cosine:
        sigma = np.arccos(sigma)
    return sigma, F1, F2


def principal_angles(V1, V2, cosine=True):
    '''
    Return the cosine of the principal angles of V1 and V2 in the vectornorm M.

    Parameters
    ----------
    V1 : ndarray
        array denoting n-dimensional subspace spanned by V1 (Mxn)
    V2 : ndarray
        array denoting subspace 2. Dimension is (MxO)
    cosine : bool, optional
        flag stating, if the cosine of the angles is to be used

    Returns
    -------
    sigma : ndarray
        cosine of subspace angles

    Examples
    --------
    TODO

    Note
    ----
    Both matrices V1 and V2 have live in the same vector space, i.e. they have
    to have the same number of rows

    See Also
    --------
    principal_angles_and_vectors

    References
    ----------
    G. H. Golub and C. F. Van Loan. Matrix computations, volume 3. JHU Press, 2012.

    '''
    Q1, R1 = linalg.qr(V1, mode='economic')
    Q2, R2 = linalg.qr(V2, mode='economic')
    sigma = linalg.svdvals(Q1.T.dot(Q2))
    if not cosine:
        sigma = np.arccos(sigma)

    return sigma


def krylov_subspace(M, K, b, omega = 0, no_of_moments=3):
    '''
    Computes the Krylov Subspace associated with the input matrix b at the frequency omega.

    Parameters
    ----------
    M : ndarray
        Mass matrix of the system.
    K : ndarray
        Stiffness matrix of the system.
    b : ndarray
        input vector of external forcing.
    omega : float, optional
        frequency for the frequency shift of the stiffness. Default value 0.
    no_of_moments : int, optional
        number of moments matched. Default value 3.

    Returns
    -------
    V : ndarray
        Krylov basis where vectors V[:,i] give the basis vectors.

    Examples
    --------
    TODO

    References
    ----------

    '''
    ndim = M.shape[0]
    no_of_inputs = b.size//ndim
    V = np.zeros((ndim, no_of_moments*no_of_inputs))
    lu = linalg.lu_factor(K - omega**2 * M)
    b_new = linalg.lu_solve(lu, b)
    b_new /= linalg.norm(b_new)
    V[:,0:no_of_inputs] = b_new.reshape((-1, no_of_inputs))
    for i in np.arange(1, no_of_moments):
        f = M.dot(b_new)
        b_new = linalg.lu_solve(lu, f)
        b_new /= linalg.norm(b_new)
        V[:,i*no_of_inputs:(i+1)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        V[:,:(i+1)*no_of_inputs], r = linalg.qr(V[:,:(i+1)*no_of_inputs], mode='economic')
        b_new = V[:,i*no_of_inputs:(i+1)*no_of_inputs]
    sigmas = linalg.svdvals(V)
    print('Krylov Basis constructed. The singular values of the basis are', sigmas)
    return V


def craig_bampton(M, K, b, no_of_modes=5, one_basis=True):
    '''
    Computes the Craig-Bampton basis for the System M and K with the input Matrix b.

    Parameters
    ----------
    M : ndarray
        Mass matrix of the system.
    K : ndarray
        Stiffness matrix of the system.
    b : ndarray
        Input vector of the system
    no_of_modes : int, optional
        Number of internal vibration modes for the reduction of the system.
        Default is 5.
    one_basis : bool, optional
        Flag for setting, if one Craig-Bampton basis should be returned or if
        the static and the dynamic basis is chosen separately
    Returns
    -------
    V : array
        Basis constisting of static displacement modes and internal vibration modes

    if one_basis=True is chosen:

    V_static : ndarray
        Static displacement modes corresponding to the input vectors b with
        V_static[:,i] being the corresponding static displacement vector to b[:,i].
    V_dynamic : ndarray
        Internal vibration modes with the boundaries fixed.
    omega : ndarray
        eigenfrequencies of the internal vibration modes.

    Examples
    --------
    TODO

    Note
    ----
    There is a filter-out command to remove the interface eigenvalues of the system.

    References
    ----------
    TODO
    '''
    # boundaries
    ndof = M.shape[0]
    b_internal = b.reshape((ndof, -1))
    indices = sp.nonzero(b)
    boundary_indices = list(set(indices[0])) # indices
    no_of_inputs = b_internal.shape[-1]
    V_static_tmp = np.zeros((ndof, len(boundary_indices)))
    K_tmp = K.copy()
    K_tmp[:, boundary_indices] *= 0
    K_tmp[boundary_indices, :] *= 0
    K_tmp[boundary_indices, boundary_indices] = 1
    for i, index  in enumerate(boundary_indices):
        f = - K[:,index]
        f[boundary_indices] = 0
        f[index] = 1
        V_static_tmp[:,i] = linalg.solve(K_tmp, f)
    # Static Modes:
    V_static = np.zeros((ndof, no_of_inputs))
    for i in range(no_of_inputs):
        V_static[:,i] = V_static_tmp.dot(b_internal[boundary_indices, [i,]])

    # inner modes
    M_tmp = M.copy()
    # Attention: introducing eigenvalues of magnitude 1 into the system
    M_tmp[:, boundary_indices] *= 0
    M_tmp[boundary_indices, :] *= 0
    M_tmp[boundary_indices, boundary_indices] = 1E0
    K_tmp[boundary_indices, boundary_indices] = 1E0
    omega, V_dynamic = linalg.eigh(K_tmp, M_tmp)
    indexlist = np.nonzero(np.round(omega - 1, 3))[0]
    omega = np.sqrt(omega[indexlist])
    V_dynamic = V_dynamic[:, indexlist]
    if one_basis:
        return sp.hstack((V_static, V_dynamic[:, :no_of_modes]))
    else:
        return V_static, V_dynamic[:, :no_of_modes], omega[:no_of_modes]


def pod_basis(u_series):
    '''
    Compute the pod basis for the series of displacements u.

    Attention! this method is not implemented yet!
    '''
    pass






