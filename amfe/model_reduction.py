# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Module of AMfe which handles the reduced order models.
"""

__all__ = ['reduce_mechanical_system',
           'qm_reduce_mechanical_system',
           'modal_derivative',
           'modal_derivative_theta',
           'static_correction_derivative',
           'static_correction_theta',
           'principal_angles',
           'krylov_subspace',
           'mass_orth',
           'craig_bampton',
           'vibration_modes',
           'pod',
           'theta_orth_v',
           'linear_qm_basis',
           'krylov_force_subspace',
           'force_norm',
           'compute_nskts',
           'modal_analysis',
           'modal_analysis_pardiso',
           'modal_assurance',
           'ranking_of_weighting_matrix',
           'linear_qm_basis_ranked',
           ]

import copy
import numpy as np
import scipy as sp
from scipy import linalg
import time
import multiprocessing as mp

from .mechanical_system import ReducedSystem, QMSystem
from .solver import solve_sparse, SpSolve, solve_nonlinear_displacement
from .num_exp_toolbox import apply_async

def reduce_mechanical_system(mechanical_system, V, overwrite=False,
                             assembly='indirect'):
    '''
    Reduce the given mechanical system with the linear basis V.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray
        Reduction Basis for the reduced system
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory
        intensive for large systems) or not.
    assembly : str {'direct', 'indirect'}
            flag setting, if direct or indirect assembly is done. For larger
            reduction bases, the indirect method is much faster.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and
        reduction basis V

    Example
    -------

    '''

    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
    reduced_sys.__class__ = ReducedSystem
    reduced_sys.V = V.copy()
    reduced_sys.V_unconstr = reduced_sys.dirichlet_class.unconstrain_vec(V)
    reduced_sys.u_red_output = []
    reduced_sys.M_constr = None
    # reduce Rayleigh damping matrix
    if reduced_sys.D_constr is not None:
        reduced_sys.D_constr = V.T @ reduced_sys.D_constr @ V
    reduced_sys.assembly_type = assembly
    return reduced_sys


def qm_reduce_mechanical_system(mechanical_system, V, Theta, overwrite=False):
    '''
    Reduce the given mechanical system to a QM system with the basis V and the
    quadratic part Theta.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray
        Reduction Basis for the reduced system
    Theta : ndarray
        Quadratic tensor for the Quadratic manifold. Has to be symmetric with
        respect to the last two indices and is of shape (n_full, n_red, n_red).
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory
        intensive for large systems) or not.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Quadratic Manifold reduced system with same properties of the
        mechanical system and reduction basis V and Theta

    Example
    -------

    '''
    # consistency check
    assert V.shape[-1] == Theta.shape[-1]
    assert Theta.shape[1] == Theta.shape[2]
    assert Theta.shape[0] == V.shape[0]

    no_of_red_dofs = V.shape[-1]
    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)

    reduced_sys.__class__ = QMSystem
    reduced_sys.V = V.copy()
    reduced_sys.Theta = Theta.copy()

    # reduce Rayleigh damping matrix
    if reduced_sys.D_constr is not None:
        reduced_sys.D_constr = V.T @ reduced_sys.D_constr @ V

    # define internal variables
    reduced_sys.u_red_output = []
    reduced_sys.no_of_red_dofs = no_of_red_dofs
    return reduced_sys


SQ_EPS = np.sqrt(np.finfo(float).eps)

def modal_derivative(x_i, x_j, K_func, M, omega_i, h=1.0, verbose=True,
                     finite_diff='central'):
    '''
    Compute the real modal derivative of the given system using Nelson's formulation.

    The modal derivative computed is :math:`\\frac{dx_i}{dx_j}`, i.e. the change of the
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
        step size for the computation of the finite difference scheme. Default
        value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.
    finite_diff : str {'central', 'upwind'}
        Method for finite difference scheme. 'central' computes the finite difference
        based on a central difference scheme, 'upwind' based on an upwind scheme. Note
        that the upwind scheme can cause severe distortions of the modal derivative.

    Returns
    -------
    dx_i / dx_j : ndarray
        The modal derivative dx_i / dx_j with mass consideration

    Notes
    -----
    The the vectors x_i and x_j are internally mass normalized;

    See Also
    --------
    static_correction_derivative
    modal_derivative_theta

    Examples
    --------
    todo

    References
    ----------
    .. [1]  S. R. Idelsohn and A. Cardona. A reduction method for nonlinear
            structural dynamic analysis. Computer Methods in Applied Mechanics
            and Engineering, 49(3):253–279, 1985.
    .. [2]  S. R. Idelsohn and A. Cardona. A load-dependent basis for reduced
            nonlinear structural dynamics. Computers & Structures,
            20(1):203–210, 1985.


    '''
    # mass normalization
    x_i /= np.sqrt(x_i.dot(M.dot(x_i)))
    x_j /= np.sqrt(x_j.dot(M.dot(x_j)))

    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    # finite difference scheme
    if finite_diff == 'central':
        dK_dx_j = (K_func(h*x_j) - K_func(-h*x_j))/(2*h)
    elif finite_diff == 'upwind':
        dK_dx_j = (K_func(x_j*h) - K)/h
    else:
        raise ValueError('Finite difference scheme is not valid.')
    dK_dx_j = (K_func(x_j*h) - K)/h
    d_omega_2_d_x_i = x_i @ dK_dx_j @ x_i
    F_i = (d_omega_2_d_x_i*M - dK_dx_j) @ x_i
    K_dyn_i = K - omega_i**2 * M
    # fix the point with the maximum displacement of the vibration mode
    row_index = np.argmax(abs(x_i))
    K_dyn_i[:,row_index], K_dyn_i[row_index,:], K_dyn_i[row_index,row_index] = 0, 0, 1
    F_i[row_index] = 0
    v_i = solve_sparse(K_dyn_i, F_i, matrix_type='symm')
    c_i = - v_i @ M @ x_i
    dx_i_dx_j = v_i + c_i*x_i
    if verbose:
        print('\nComputation of modal derivatives. ')
        print('Influence of the change of the eigenfrequency:', d_omega_2_d_x_i)
        print('The condition number of the problem is', np.linalg.cond(K_dyn_i))
        res = (K - omega_i**2 * M).dot(dx_i_dx_j) - (d_omega_2_d_x_i*M - dK_dx_j).dot(x_i)
        print('The residual is', np.sqrt(res.dot(res)),
              ', the relative residual is', np.sqrt(res.dot(res))/np.sqrt(F_i.dot(F_i)))
    return dx_i_dx_j


def modal_derivative_theta(V, omega, K_func, M, h=1.0, verbose=True,
                           symmetric=True, finite_diff='central'):
    r'''
    Compute the basis theta based on real modal derivatives.

    Parameters
    ----------
    V : ndarray
        array containing the linear basis
    omega : ndarray
        eigenfrequencies of the system in rad/s.
    K_func : function
        function returning the tangential stiffness matrix for a given
        displacement. Has to work like K = K_func(u).
    M : ndarray or sparse matrix
        Mass matrix of the system.
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True
    symmetric : bool, optional
        flag for making the modal derivative matrix theta symmetric. Default is
        `True`.
    finite_diff : str {'central', 'upwind'}
        Method for finite difference scheme. 'central' computes the finite difference
        based on a central difference scheme, 'upwind' based on an upwind scheme. Note
        that the upwind scheme can cause severe distortions of the modal derivative.

    Returns
    -------
    Theta : ndarray
        three dimensional array of modal derivatives. Theta[:,i,j] contains
        the modal derivative 1/2 * dx_i / dx_j. The basis Theta is made symmetric, so
        that `Theta[:,i,j] == Theta[:,j,i]` if `symmetic=True`.

    See Also
    --------
    static_correction_theta : modal derivative with mass neglection.
    modal_derivative : modal derivative for only two vectors.

    '''
    no_of_dofs = V.shape[0]
    no_of_modes = V.shape[1]
    Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes))

    # Check, if V is mass normalized:
    if not np.allclose(np.eye(no_of_modes), V.T @ M @ V, rtol=1E-5, atol=1E-8):
        Exception('The given modes are not mass normalized!')

    K = K_func(np.zeros(no_of_dofs))

    for i in range(no_of_modes): # looping over the columns
        x_i = V[:,i]
        K_dyn_i = K - omega[i]**2 * M

        # fix the point with the maximum displacement of the vibration mode
        fix_idx = np.argmax(abs(x_i))
        K_dyn_i[:,fix_idx], K_dyn_i[fix_idx,:], K_dyn_i[fix_idx, fix_idx] = 0, 0, 1

        # factorization of the dynamic stiffness matrix
        if verbose:
            print('Factorizing the dynamic stiffness matrix for eigenfrequency',
                  '{0:d} with {1:4.2f} rad/s.'.format(i, omega[i]) )
        LU_object = SpSolve(K_dyn_i)

        for j in range(no_of_modes): # looping over the rows
            x_j = V[:,j]
            # finite difference scheme
            if finite_diff == 'central':
                dK_dx_j = (K_func(h*x_j) - K_func(-h*x_j))/(2*h)
            elif finite_diff == 'upwind':
                dK_dx_j = (K_func(h*x_j) - K)/h
            else:
                raise ValueError('Finite difference scheme is not valid.')

            d_omega_2_d_x_i = x_i @ dK_dx_j @ x_i
            F_i = (d_omega_2_d_x_i*M - dK_dx_j) @ x_i
            F_i[fix_idx] = 0
            v_i = LU_object.solve(F_i)
            c_i = - v_i @ M @ x_i
            Theta[:,i,j] = v_i + c_i*x_i

    LU_object.clear()
    if symmetric:
        Theta = 1/2*(Theta + Theta.transpose((0,2,1)))
    return Theta


def static_correction_derivative(x_i, x_j, K_func, h=1.0, verbose=True,
                                 finite_diff='central'):
    r'''
    Computes the static correction vectors
    :math:`\frac{\partial x_i}{\partial x_j}` of the system with a nonlinear
    force.

    Parameters
    ----------
    x_i : ndarray
        array containing displacement vectors i in the rows. x_i[:,i] is the
        i-th vector
    x_j : ndarray
        displacement vector j
    K_func : function
        function for the tangential stiffness matrix to be called in the form
        K_tangential = K_func(x_j)
    h : float, optional
        step size for the computation of the finite difference scheme. Default
        value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.
    finite_diff : str {'central', 'upwind'}
        Method for finite difference scheme. 'central' computes the finite difference
        based on a central difference scheme, 'upwind' based on an upwind scheme. Note
        that the upwind scheme can cause severe distortions of the static correction
        derivative.

    Returns
    -------
    dx_i_dx_j : ndarray
        static correction derivative (if x_i and x_j is a modal vector it's
        the modal derivative neglecting mass terms) of displacement x_i with
        respect to displacement x_j

    Notes
    -----
    The static correction is done purely on the arrays x_i and x_j, so there is
    no mass normalization. This is a difference in contrast to the technique
    used in the related function modal_derivative.



    See Also
    --------
    modal_derivative

    '''
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    if finite_diff == 'central':
        dK_dx_j = (K_func(h*x_j) - K_func(-h*x_j))/(2*h)
    elif finite_diff == 'upwind':
        dK_dx_j = (K_func(x_j*h) - K)/h
    else:
        raise ValueError('Finite difference scheme is not valid.')
    b = - dK_dx_j.dot(x_i) # rigth hand side of equation
    dx_i_dx_j = solve_sparse(K, b, matrix_type='symm')
    if verbose:
        res = K.dot(dx_i_dx_j) + dK_dx_j.dot(x_i)
        print('\nComputation of static correction derivative. ')
        print('The condition number of the solution procedure is', np.linalg.cond(K))
        print('The residual is', linalg.norm(res),
              ', the relative residual is', linalg.norm(res)/linalg.norm(b))
    return dx_i_dx_j


def static_correction_theta(V, K_func, M=None, omega=0, h=1.0,
                            verbose=True, symmetric=True,
                            finite_diff='central'):
    '''
    Compute the static correction derivatives for the given basis V.

    Optionally, a frequency shift can be performed.

    Parameters
    ----------
    V : ndarray
        array containing the linear basis
    K_func : function
        function returning the tangential stiffness matrix for a given
        displacement. Has to work like `K = K_func(u)`.
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
    modal_derivative_theta
    static_correction_derivative

    '''
    no_of_dofs = V.shape[0]
    no_of_modes = V.shape[1]
    Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes))
    K = K_func(np.zeros(no_of_dofs))
    if (omega > 0) and (M != None):
        K_dyn = K - omega**2 * M
    else:
        K_dyn = K
    LU_object = SpSolve(K_dyn)
    for i in range(no_of_modes):
        if verbose:
            print('Computing finite difference K-matrix')
        if finite_diff == 'central':
            dK_dx_i = (K_func(h*V[:,i]) - K_func(-h*V[:,i]))/(2*h)
        elif finite_diff == 'forward':
            dK_dx_i = (K_func(h*V[:,i]) - K)/h
        elif finite_diff == 'backward':
            dK_dx_i = (-K_func(-h*V[:,i]) + K)/h
        else:
            raise ValueError('Finite difference scheme is not valid.')
        b = - dK_dx_i @ V
        if verbose:
            print('Solving linear system #', i)
        Theta[:,:,i] = LU_object.solve(b)
        if verbose:
            print('Done solving linear system #', i)
    if verbose:
        residual = np.linalg.norm(Theta - Theta.transpose(0,2,1)) / \
                   np.linalg.norm(Theta)
        print('The residual, i.e. the unsymmetric values, are', residual)
    LU_object.clear()
    if symmetric:
        # make Theta symmetric
        Theta = 1/2*(Theta + Theta.transpose(0,2,1))
    return Theta

def theta_orth_v(Theta, V, M, overwrite=False):
    '''
    Make third order tensor Theta fully mass orthogonal with respect to the
    basis V via a Gram-Schmid-process.

    Parameters
    ----------
    Theta : ndarray
        Third order Tensor describing the quadratic part of the basis
    V : ndarray
        Linear Basis
    M : ndarray or scipy.sparse matrix
        Mass Matrix
    overwrite : bool
        Flag for setting, if Theta should be overwritten in-place

    Returns
    -------
    Theta_orth : ndarray
        Third order tensor Theta mass orthogonalized, such that
        Theta_orth[:,i,j] is mass orthogonal to V[:,k]:
        :math:`\\theta_{ij}^T M V = 0`

    '''
    # Make sure, that V is M-orthogonal
    __, no_of_modes = V.shape
    V_M_space = M @ V
    np.testing.assert_allclose(V.T @ V_M_space, np.eye(no_of_modes), atol=1E-14)
    if overwrite:
        Theta_ret = Theta
    else:
        Theta_ret = Theta.copy()
    # inner product of Theta[:,j,k] with V[:,l] in the M-norm
    inner_prod = np.einsum('ijk, il -> jkl', Theta, V_M_space)
    for j in range(no_of_modes):
        for k in range(no_of_modes):
            Theta_ret[:,j,k] -= V @ inner_prod[j,k,:]
    return Theta_ret


def linear_qm_basis(V, theta, M=None, tol=1E-8, symm=True):
    '''
    Make a linear basis containing the subspace spanned by V and theta by
    deflation.

    Parameters
    ----------
    V : ndarray
        linear basis
    theta : ndarray
        third order tensor filled with the modal derivatices associated with V
    tol : float, optional
        Tolerance for the deflation via SVD. The omitted singular values are
        at least smaller than the largest one multiplied with tol.
        Default value: 1E-8.

    Returns
    -------
    V_ret : ndarray
        linear basis containing the subspace spanned by V and theta.

    '''
    ndof, n = V.shape
    if symm:
        V_raw = np.zeros((ndof, n*(n+3)//2))
        V_raw[:,:n] = V[:,:]
        for i in range(n):
            for j in range(i+1):
                idx = n + i*(i+1)//2 + j
                if M is None:
                    theta_norm = np.sqrt(theta[:,i,j].T @ theta[:,i,j])
                else:
                    theta_norm = np.sqrt(theta[:,i,j].T @ M @ theta[:,i,j])
                V_raw[:,idx] = theta[:,i,j] / theta_norm
    else:
        V_raw = np.zeros((ndof, n*(n+1)))
        V_raw[:,:n] = V[:,:]
        for i in range(n):
            for j in range(n):
                idx = n*(i+1) + j
                if M is None:
                    theta_norm = np.sqrt(theta[:,i,j].T @ theta[:,i,j])
                else:
                    theta_norm = np.sqrt(theta[:,i,j].T @ M @ theta[:,i,j])
                V_raw[:,idx] = theta[:,i,j] / theta_norm

    # Deflation algorithm
    U, s, V_svd = sp.linalg.svd(V_raw, full_matrices=False)
    idx_defl = s > s[0]*tol
    V_ret = U[:,idx_defl]
    return V_ret


def principal_angles(V1, V2, cosine=True, principal_vectors=False):
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
    principal_vectors : bool, optional
        Option flag for returning principal vectors. Default is False.

    Returns
    -------
    sigma : ndarray
        cosine of subspace angles
    F1 : ndarray
        array of principal vectors of subspace spanned by V1. The columns give
        the principal vectors, i.e. F1[:,0] is the first principal vector
        associated with theta[0] and so on. Only returned, if
        ``principal_vectors=True``.
    F2 : ndarray
        array of principal vectors of subspace spanned by V2. Only returned if
        ``principal_vectors=True``.

    Notes
    -----
    Both matrices V1 and V2 have live in the same vector space, i.e. they have
    to have the same number of rows.

    Examples
    --------
    TODO

    References
    ----------
    ..  [1] G. H. Golub and C. F. Van Loan. Matrix computations, volume 3. JHU
        Press, 2012.

    '''
    Q1, __ = linalg.qr(V1, mode='economic')
    Q2, __ = linalg.qr(V2, mode='economic')
    U, sigma, V = linalg.svd(Q1.T @ Q2)

    if not cosine:
        sigma = np.arccos(sigma)

    if principal_vectors is True:
        F1 = Q1.dot(U)
        F2 = Q2.dot(V.T)
        return sigma, F1, F2
    else:
        return sigma


def krylov_subspace(M, K, b, omega=0, no_of_moments=3, mass_orth=True):
    '''
    Computes the Krylov Subspace associated with the input matrix b at the
    frequency omega.

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
    mass_orth : bool, optional
        flag for setting orthogonality of returnd Krylov basis vectors. If
        True, basis vectors are mass-orthogonal (V.T @ M @ V = eye). If False,
        basis vectors are orthogonal (V.T @ V = eye)

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
    f = b.copy()
    V = np.zeros((ndim, no_of_moments*no_of_inputs))
    LU_object = SpSolve(K - omega**2 * M)

    for i in np.arange(no_of_moments):
        b_new = LU_object.solve(f)
        # b_new /= linalg.norm(b_new)
        V[:,i*no_of_inputs:(i+1)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        V[:,:(i+1)*no_of_inputs], R = linalg.qr(V[:,:(i+1)*no_of_inputs],
                                                mode='economic')
        b_new = V[:,i*no_of_inputs:(i+1)*no_of_inputs]
        f = M.dot(b_new)
    LU_object.clear()
    sigmas = linalg.svdvals(V)

    # mass-orthogonalization of V:
    if mass_orth:
        # Gram-Schmid-process
        for i in range(no_of_moments*no_of_inputs):
            v = V[:,i]
            v /= np.sqrt(v @ M @ v)
            V[:,i] = v
            weights = v @ M @ V[:,i+1:]
            V[:,i+1:] -= v.reshape((-1,1)) * weights

    print('Krylov Basis constructed. The singular values of the basis are', sigmas)
    return V


def krylov_force_subspace(M, K, b, omega=0, no_of_moments=3,
                          orth='euclidean'):
    '''
    Compute a krylov force subspace for the computation of snapshots needed in
    hyper reduction.

    The Krylov force basis is given as

    ..code::

        [b, M @ inv(K - omega**2) @ b, ...,
         (M @ inv(K - omega**2))**(no_of_moments-1) @ b]

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
    orth : str, {'euclidean', 'impedance', 'kinetic'} optional
        flag for setting orthogonality of returnd Krylov basis vectors.

        * 'euclidean' : ``V.T @ V = eye``
        * 'impedance' : ``V.T @ inv(K) @ V = eye``
        * 'kinetic' : ``V.T @ inv(K).T @ M @ inv(K) @ V = eye``

    Returns
    -------
    V : ndarray
        Krylov force basis where vectors V[:,i] give the basis vectors.

    '''
    ndim = M.shape[0]
    no_of_inputs = b.size//ndim
    f = b.copy()
    V = np.zeros((ndim, no_of_moments*no_of_inputs))
    LU_object = SpSolve(K - omega**2 * M)
    b_new = f
    for i in np.arange(no_of_moments):
        V[:,i*no_of_inputs:(i+1)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        V[:,:(i+1)*no_of_inputs], R = sp.linalg.qr(V[:,:(i+1)*no_of_inputs],
                                                   mode='economic')
        f = V[:,i*no_of_inputs:(i+1)*no_of_inputs]
        u = LU_object.solve(f)
        b_new = M.dot(u)

    sigmas = sp.linalg.svdvals(V)

    # mass-orthogonalization of V:
    if orth == 'impedance':
        # Gram-Schmid-process
        for i in range(no_of_moments*no_of_inputs):
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u @ v)
            V[:,i] = v
            weights = u.T @ V[:,i+1:]
            V[:,i+1:] -= v.reshape((-1,1)) * weights
    if orth == 'kinetic':
        for i in range(no_of_moments*no_of_inputs):
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:,i] = v
            if i+1 < no_of_moments*no_of_inputs:
                weights = u.T @ M @ LU_object.solve(V[:,i+1:])
                V[:,i+1:] -= v.reshape((-1,1)) * weights


    LU_object.clear()
    print('Krylov force basis constructed.',
          'The singular values of the basis are', sigmas)
    return V


def modal_force_subspace(M, K, no_of_modes=3, orth='euclidean'):
    '''
    Force subspace spanned by the forces producing the vibration modes.
    '''
    lambda_, Phi = sp.sparse.linalg.eigsh(K, M=M, k=no_of_modes, sigma=0,
                                          which='LM',
                                          maxiter=100)
    V = K @ Phi

    LU_object = SpSolve(K)

    if orth == 'euclidean':
        V, _ = sp.linalg.qr(V, mode='economic')

    elif orth == 'impedance':
        omega = np.sqrt(lambda_)
        V /= omega
        # Gram-Schmid-process
#        for i in range(no_of_modes):
#            v = V[:,i]
#            u = LU_object.solve(v)
#            v /= np.sqrt(u @ v)
#            V[:,i] = v
#            weights = u.T @ V[:,i+1:]
#            V[:,i+1:] -= v.reshape((-1,1)) * weights

    elif orth == 'kinetic':
        for i in range(no_of_modes):
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:,i] = v
            if i+1 < no_of_modes:
                weights = u.T @ M @ LU_object.solve(V[:,i+1:])
                V[:,i+1:] -= v.reshape((-1,1)) * weights

    LU_object.clear()
    print('Modal force basis constructed with orth type {}.'.format(orth))
    return V

def force_norm(F, K, M, norm='euclidean'):
    '''
    Compute the norm of the given force vector or array

    Parameters
    ----------
    F : ndarray, shape (n,) or shape (n, m)
        Array representing the external force in constrained coordinates
    K : sparse array, shape (n, n)
        Stiffness matrix
    M : sparse array, shape (n, n)
        Mass matrix
    norm : str, {'euclidean', 'impedance', 'kinetic'}
        norm flag indicating the norm type:

        * 'euclidean' : ``sqrt(F.T @ F)``
        * 'impedance' : ``sqrt(F.T @ inv(K) @ F)``
        * 'kinetic' : ``sqrt(F.T @ inv(K).T @ M @ inv(K) @ F)``

    Returns
    -------
    norm : float or array of shape(m)
        norm of the given force vector or array. When F is an array with m
        columns, norm is a vector with the norm given for every column
    '''
    # define diag operator which also works for floats
    if len(F.shape) == 1:
        diag = lambda x : x
    elif len(F.shape) == 2:
        diag = np.diag
    else:
        raise ValueError('Dimension mismatch')

    if norm == 'euclidean':
        output =  np.sqrt(diag(F.T @ F))
    elif norm == 'impedance':
        u = solve_sparse(K, F)
        output =  np.sqrt(diag(F.T @ u))
    elif norm == 'kinetic':
        u = solve_sparse(K, F)
        output = np.sqrt(diag(u.T @ M @ u))

    return output


def mass_orth(V, M, overwrite=False, niter=2):
    '''
    Mass-orthogonalize the matrix V with respect to the mass matrix M with a
    Gram-Schmid-procedure.

    Parameters
    ----------
    V : ndarray
        Matrix (e.g. projection basis) containing displacement vectors in the
        column. Shape is (ndim, no_of_basis_vectors)
    M : ndarray / sparse matrix.
        Mass matrix. Shape is (ndim, ndim).
    overwrite : bool
        Flag setting, if matrix V should be overwritten.
    niter : int
        Number of Gram-Schmid runs for the orthogonalization. As the
        Gram-Schmid-procedure is not stable, more then one iteration are
        recommended.

    Returns
    -------
    V_orth : ndarray
        Mass-orthogonalized basis V

    '''
    if overwrite:
        V_orth = V
    else:
        V_orth = V.copy()

    __, no_of_basis_vecs = V.shape
    for run_no in range(niter):
        for i in range(no_of_basis_vecs):
            v = V_orth[:,i]
            v /= np.sqrt(v @ M @ v)
            V_orth[:,i] = v
            weights = v @ M @ V_orth[:,i+1:]
            V_orth[:,i+1:] -= v.reshape((-1,1)) * weights
    return V_orth


def craig_bampton(M, K, b, no_of_modes=5, one_basis=True):
    '''
    Computes the Craig-Bampton basis for the System M and K with the input
    Matrix b.

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
    if `one_basis=True` is chosen:

    V : array
        Basis constisting of static displacement modes and internal vibration
        modes

    if `one_basis=False` is chosen:

    V_static : ndarray
        Static displacement modes corresponding to the input vectors b with
        V_static[:,i] being the corresponding static displacement vector to
        b[:,i].
    V_dynamic : ndarray
        Internal vibration modes with the boundaries fixed.
    omega : ndarray
        eigenfrequencies of the internal vibration modes.

    Examples
    --------
    TODO

    Notes
    -----
    There is a filter-out command to remove the interface eigenvalues of the
    system.

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
        V_static_tmp[:,i] = solve_sparse(K_tmp, f, matrix_type='symm')
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


def vibration_modes(mechanical_system, n=10, save=False):
    '''
    Compute the n first vibration modes of the given mechanical system using
    a power iteration method.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system to be analyzed.
    n : int
        number of modes to be computed.
    save : bool
        Flag for saving the modes in mechanical_system for ParaView export.
        Default: False.

    Returns
    -------
    omega : ndarray
        vector containing the eigenfrequencies of the mechanical system in
        rad / s.
    Phi : ndarray
        Array containing the vibration modes. Phi[:,0] is the first vibration
        mode corresponding to eigenfrequency omega[0]

    Examples
    --------

    Notes
    -----
    The core command using the ARPACK library is a little bit tricky. One has
    to use the shift inverted mode for the solution of the mechanical
    eigenvalue problem with the largest eigenvalues. Generally no convergence
    is gained when the smallest eigenvalue is to be found.

    If the squared eigenvalue omega**2 is negative, as it might happen due to
    round-off errors with rigid body modes, the negative sign is traveled to
    the eigenfrequency omega, though this makes physically no sense...
    '''
    K = mechanical_system.K()
    M = mechanical_system.M()

    lambda_, V = sp.sparse.linalg.eigsh(K, M=M, k=n, sigma=0, which='LM',
                                        maxiter=100)
    omega = np.sqrt(abs(lambda_))
    # Little bit of sick hack: The negative sign is transferred to the
    # eigenfrequencies
    omega[lambda_ < 0] *= -1

    if save:
        mechanical_system.clear_timesteps()
        for i, om in enumerate(omega):
            mechanical_system.write_timestep(om, V[:, i])

    return omega, V

modal_analysis = vibration_modes


def modal_analysis_pardiso(mechanical_system, n=10, u_eq=None,
                           save=False, niter_max=40,
                           rtol=1E-14):
    '''
    Make a modal analysis using a naive Lanczos iteration.

    Parameters
    ----------
    mechanical_system : instance of amfe.MechanicalSystem
        Mechanical system
    n : int
        number of modes to be computed
    u_eq : ndarray, optional
        equilibrium position, around which the vibration modes should be
        computed
    save : bool, optional
        flat setting, if the modes should be saved in mechanical system.
        Default value: False
    niter_max : int, optional
        Maximum number of Lanzcos iterations

    Returns
    -------
    om : ndarray, shape(n)
        eigenfrequencies of the system
    Phi : ndarray, shape(ndim, n)
        Vibration modes of the system

    Note
    ----
    In comparison to the modal_analysis method, this method uses the Pardiso
    solver if available and a naive Lanczos iteration. The modal_analysis method
    uses the arpack solver which is more accurate but takes also much longer for
    large systems, since the factorization is very inefficient by using superLU.
    '''
    K = mechanical_system.K(u=u_eq)
    M = mechanical_system.M(u=u_eq)
    k_diag = K.diagonal().sum()

    # factorizing
    K_mat = SpSolve(K)

    # build up Krylov sequence
    n_rand = n
    n_dim = K.shape[0]

    residual = np.zeros(n)
    b = np.random.rand(n_dim, n_rand)
    b, _ = sp.linalg.qr(b, mode='economic')
    krylov_subspace = b
    for n_iter in range(niter_max):
        print('Lanczos iteration # {}. '.format(n_iter), end='')
        new_directions = K_mat.solve(M @ b)
        krylov_subspace = np.concatenate((krylov_subspace, new_directions), axis=1)
        krylov_subspace, _ = sp.linalg.qr(krylov_subspace, mode='economic')
        b = krylov_subspace[:,-n_rand:]

        # check the modes
        K_red = krylov_subspace.T @ K @ krylov_subspace
        M_red = krylov_subspace.T @ M @ krylov_subspace
        lambda_r, Phi_r = sp.linalg.eigh(K_red, M_red, overwrite_a=True,
                                         overwrite_b=True)
        Phi = krylov_subspace @ Phi_r[:,:n]

        # check the tolerance to be below machine epsilon with some buffer...
        for i in range(n):
            residual[i] = np.sum(abs((-lambda_r[i] * M + K) @ Phi[:,i])) / k_diag

        print('Res max: {:.2e}'.format(np.max(residual)))
        if np.max(residual) < rtol:
            break

    if n_iter-1 == niter_max:
        print('No convergence gained in the given iteration steps.')

    print('The Lanczos solver took ' +
          '{} iterations to solve for {} eigenvectors.'.format(n_iter+1, n))
    omega = np.sqrt(abs(lambda_r[:n]))
    V = Phi[:,:n]
    K_mat.clear()
    # Little bit of sick hack: The negative sign is transferred to the
    # eigenfrequencies
    omega[lambda_r[:n] < 0] *= -1

    if save:
        mechanical_system.clear_timesteps()
        if u_eq is None:
            u_eq = np.zeros_like(V[:,0])
        for i, om in enumerate(omega):
            mechanical_system.write_timestep(om, V[:, i] + u_eq)

    return omega, V

def pod(mechanical_system, n=None):
    '''
    Compute the POD basis of a mechanical system.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        MechanicalSystem which has run a time simulation and thus displacement
        fields stored internally.
    n : int, optional
        Number of POD basis vectors which should be returned. Default is `None`
        returning all POD vectors.

    Returns
    -------
    sigma : ndarray
        Array of the singular values.
    V : ndarray
        Array containing the POD vectors. V[:,0] contains the POD-vector
        associated with sigma[0] etc.

    Examples
    --------
    TODO

    '''
    S = np.array(mechanical_system.u_output).T
    U, sigma, __ = sp.linalg.svd(S, full_matrices=False)
    U_return = mechanical_system.constrain_vec(U[:,:n])
    return sigma[:n], U_return


def modal_assurance(U, V):
    r'''
    Compute the Modal Assurance Criterion (MAC) of the vectors stacked
    in U and V:

    .. math::
        U = [u_1, \dots, u_n], V = [v_1, \dots, v_n] \\
        MAC_{i,j} = \frac{(u_i^Tv_j)^2}{u_i^T u_i \cdot v_j^T v_j}


    Parameters
    ----------
    U : ndarray, shape(N, no_of_modes)
        Array with one set of vectors
    V : ndarray, shape(N, no_of_modes)
        Array with another set of vectors

    Returns
    -------
    mac : ndarray, shape(no_of_modes, no_of_modes)
        mac criterion array showing, how the modes coincide. The rows are
        associated with the vectors in U, the columns with the vectors in V.
        mac[i,j] gives the squared correlation coefficient of the vecotor
        U[:,i] and V[:,j].

    References
    ----------
    .. [1]  Géradin, Michel and Rixen, Daniel: Mechanical Vibrations.
            John Wiley & Sons, 2014. p.499.

    '''
    nominator =  (U.T @ V)**2
    diag_u_squared = np.einsum('ij, ij->j', U, U)
    diag_v_squared = np.einsum('ij, ij->j', V, V)
    denominator = np.outer(diag_u_squared, diag_v_squared)
    return nominator / denominator


def compute_nskts(mechanical_system,
                  F_ext_max=None,
                  no_of_moments=4,
                  no_of_static_cases=8,
                  load_factor=2,
                  no_of_force_increments=20,
                  no_of_procs=None,
                  norm='impedance',
                  verbose=True,
                  force_basis='krylov'):
    '''
    Compute the Nonlinear Stochastic Krylov Training Sets (NSKTS).

    NSKTS can be used as training sets for Hyper Reduction of nonlinear systems.


    Parameters
    ----------
    mechanical_system : instance of amfe.MechanicalSystem
        Mechanical System to which the NSKTS should be computed
    F_ext_max : ndarray, optional
        Maximum external force. If None is given, 10 random samples of the
        external force in the time range t = [0,1] are computed and the maximum
        value is taken. Default value is None.
    no_of_moments : int, optional
        Number of moments consiedered in the Krylov force subspace. Default
        value is 4.
    no_of_static_cases : int, optional
        Number of stochastic static cases which are solved. Default value is 8.
    load_factor : int, optional
        Load amplification factor with which the maximum external force is
        multiplied. Default value is 2.
    no_of_force_increments : int, optional
        Number of force increments for nonlinear solver. Default value is 20.
    no_of_procs : {int, None}, optional
        Number of processes which are started parallel. For None
        no_of_static_cases processes are run.
    norm : str {'impedance', 'eucledian', 'kinetic'}, optional
        Norm which will be used to scale the higher order moments for the Krylov
        force subspace. Default value is 'impedance'.
    verbose : bool, optional
        Flag for setting verbose output. Default value is True.
    force_basis : str {'krylov', 'modal'}, optional
        Type of force basis used. Either krylov meaning the classical NSKTS or
        modal meaning the forces producing vibration modes.

    Returns
    -------
    nskts_arr : ndarray
        Nonlinear Stochastic Krylov Training Sets. Every column in nskts_arr
        represents one NSKTS displacement field.

    Reference
    ---------
    Todo

    '''
    def compute_stochastic_displacements(mechanical_system, F_rand):
        '''
        Solve a static problem for the given Force F_rand

        '''
        def f_ext_monkeypatched(u, du, t):
            return F_rand * t
        f_ext_tmp = mechanical_system.f_ext
        mechanical_system.f_ext = f_ext_monkeypatched

        u_arr = solve_nonlinear_displacement(mechanical_system,
                                             no_of_load_steps=no_of_force_increments,
                                             n_max_iter=no_of_force_increments,
                                             verbose=verbose,
                                             conv_abort=True,
                                             save=False)

        mechanical_system.f_ext = f_ext_tmp
        return u_arr

    print('*'*80)
    print('Start computing nonlinear stochastic ' +
          '{} training sets.'.format(force_basis))
    print('*'*80)
    time_1 = time.time()
    K = mechanical_system.K()
    M = mechanical_system.M()
    ndim = K.shape[0]
    u = du = np.zeros(ndim)
    if F_ext_max is None:
        F_ext_max = 0
        # compute the maximum external force
        for i in range(10):
            F_tmp = mechanical_system.f_ext(u, du, np.random.rand())
            if np.linalg.norm(F_tmp) > np.linalg.norm(F_ext_max):
                F_ext_max = F_tmp

    if force_basis == 'krylov':
        F_basis = krylov_force_subspace(M, K, F_ext_max,
                                        no_of_moments=no_of_moments,
                                        orth=norm)
    elif force_basis == 'modal':
        F_basis = modal_force_subspace(M, K, no_of_modes=no_of_moments,
                                       orth=norm)
    else:
        raise ValueError('Force basis type ' + force_basis + 'not valid.')

    norm_of_forces = force_norm(F_ext_max, K, M, norm=norm)
    standard_deviation = np.ravel(np.array(
            [norm_of_forces for i in range(no_of_moments)]))
    standard_deviation *= load_factor

    # Do the parallel run
    with mp.Pool(processes=no_of_procs) as pool:
        results = []
        for i in range(no_of_static_cases):
            F_rand = F_basis @ np.random.normal(0, standard_deviation)
            vals = [copy.deepcopy(mechanical_system), F_rand.copy()]
            res = apply_async(pool, compute_stochastic_displacements, vals)
            results.append(res)
        u_list = []
        for res in results:
            u = res.get()
            u_list.append(u)

    snapshot_arr = np.concatenate(u_list, axis=1)
    time_2 = time.time()
    print('Finished computing nonlinear stochastic krylov training sets.')
    print('It took {0:2.2f} seconds to build the nskts.'.format(time_2 - time_1))
    return snapshot_arr


def ranking_of_weighting_matrix(W, symm=True):
    '''
    Return the indices of a weighting matrix W in decreasing order

    Parameters
    ----------
    W : ndarray, shape: (n, m)
        weighting matrix
    symm : bool, optional
        flag indicating symmetry

    Returns
    -------
    idxs : ndarray, shape (n*m, 2)
        array of indices, where i, j = idxs[k] are the indices of the k-largest
        value in W

    '''
    n, m = W.shape
    ranking_list = np.argsort(W, axis=None)[::-1]
    if symm:
        assert n == m
        ranking = np.zeros((n*(n+1)//2, 2), dtype=int)
        i = 0
        for val in ranking_list:
            j, k = val // m, val % m
            if k >= j: # check, if indiced are in upper half
                ranking[i,:] = j, k
                i += 1

    else: # not symmetric
        ranking = np.zeros((n*m, 2), dtype=int)
        for i, val in enumerate(ranking_list):
            ranking[i,:] = val // m, val % m
    return ranking

def linear_qm_basis_ranked(V, Theta, W, n, tol=1E-8, symm=True):
    '''
    Build a linear basis from a linear basis V and a QM Tensor Theta with
    a given weighting matrix W.

    Parameters
    ----------
    V : ndarray, shape: (ndim, m)
        Linear basis
    Theta : narray, shape: (ndim, m, m)
        Third order tensor containing the derivatives
    W : ndarray, shape (m, m)
        Weighting matrix containing the weights
    n : int
        number of derivative vectors
    tol : float, optional
        tolerance for the SVD selection. Default value: 1E-8
    symm : bool, optional
        flag if weighting matrix and Theta is symmetric. If true, only the
        upper half of Theta and W is evaluated

    Returns
    -------
    V_red : ndarray, shape: (ndim, n + m)

    '''
    ranking = ranking_of_weighting_matrix(W, symm=symm)
    Theta_ranked = Theta[:,ranking[:,0], ranking[:,1]]
    V_raw = np.concatenate((V, Theta_ranked[:,:n]), axis=1)
    U, s, V_svd = sp.linalg.svd(V_raw, full_matrices=False)
    idx_defl = s > s[0]*tol
    V_red = U[:,idx_defl]
    return V_red
