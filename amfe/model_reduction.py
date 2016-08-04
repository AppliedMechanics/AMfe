"""
Module of AMfe which handles the reduced order models.
"""

__all__ = ['reduce_mechanical_system', 'qm_reduce_mechanical_system',
           'modal_derivative', 'modal_derivative_theta',
           'static_correction_derivative', 'static_correction_theta',
           'principal_angles', 'krylov_subspace', 'mass_orth', 'craig_bampton',
           'vibration_modes', 'pod', 'theta_orth_v', 'linear_qm_basis']

import copy
import numpy as np
import scipy as sp
from scipy import linalg

from .mechanical_system import ReducedSystem, QMSystem
from .solver import solve_sparse, SpSolve


def linsolve(A, b, matrix_type='symm', verbose=False):
    '''
    Solve the linear system A @ x = b where A might be sparse. 
    
    Parameters
    ----------
    A : ndarray or sparse matrix
    b : ndarray
    matrix_type : str 
    verbose : bool
    
    
    Returns
    -------
    x : ndarray
        Solution of the linear system of equation. 
    '''
    if sp.sparse.issparse(A):
        return solve_sparse(A, b, matrix_type=matrix_type, verbose=verbose)
    else:
        return sp.linalg.solve(A, b)
    

def reduce_mechanical_system(mechanical_system, V, overwrite=False):
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
    reduced_sys.u_red_output = []
    reduced_sys.M_constr = None
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

    # define internal variables
    reduced_sys.u_red_output = []
    reduced_sys.no_of_red_dofs = no_of_red_dofs
    return reduced_sys


SQ_EPS = np.sqrt(np.finfo(float).eps)

def modal_derivative(x_i, x_j, K_func, M, omega_i, h=500*SQ_EPS, verbose=True,
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
    ---------
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
    v_i = linsolve(K_dyn_i, F_i)
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

def modal_derivative_theta(V, omega, K_func, M, h=500*SQ_EPS, verbose=True,
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


def static_correction_derivative(x_i, x_j, K_func, h=500*SQ_EPS, verbose=True,
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
    dx_i_dx_j = linsolve(K, b)
    if verbose:
        res = K.dot(dx_i_dx_j) + dK_dx_j.dot(x_i)
        print('\nComputation of static correction derivative. ')
        print('The condition number of the solution procedure is', np.linalg.cond(K))
        print('The residual is', linalg.norm(res),
              ', the relative residual is', linalg.norm(res)/linalg.norm(b))
    return dx_i_dx_j


def static_correction_theta(V, K_func, M=None, omega=0, h=500*SQ_EPS,
                            verbose=True, finite_diff='central'):
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
    finite_diff : str {'central', 'upwind'}
        Method for finite difference scheme. 'central' computes the finite difference
        based on a central difference scheme, 'upwind' based on an upwind scheme. Note
        that the upwind scheme can cause severe distortions of the static correction
        derivative.

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
        elif finite_diff == 'upwind':
            dK_dx_i = (K_func(h*V[:,i]) - K)/h
        else:
            raise ValueError('Finite difference scheme is not valid.')
        b = - dK_dx_i @ V
        if verbose:
            print('Solving linear system #', i)
        Theta[:,:,i] = LU_object.solve(b)
        if verbose:
            print('Done solving linear system #', i)
    if verbose:
        residual = np.sum(Theta - Theta.transpose(0,2,1))
        print('The residual, i.e. the unsymmetric values, are', residual)
    LU_object.clear()
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


def linear_qm_basis(V, theta, tol=1E-6):
    '''
    Make a linear basis containing the subspace spanned by V and theta by deflation.

    Parameters
    ----------
    V : ndarray
        linear basis
    theta : ndarray
        third order tensor filled with the modal derivatices associated with V
    tol : float, optional
        Tolerance for the deflation via SVD. The omitted singular values are
        at least smaller than the largest one multiplied with tol.
        Default value: 1E-6.

    Returns
    -------
    V_ret : ndarray
        linear basis containing the subspace spanned by V and theta.

    '''
    ndof, n = V.shape
    V_raw = np.zeros((ndof, n*(n+3)//2))
    V_raw[:,:n] = V[:,:]
    for i in range(n):
        for j in range(i+1):
            idx = n + i*(i+1)//2 + j
            V_raw[:,idx] = theta[:,i,j] / np.sqrt(theta[:,i,j] @ theta[:,i,j])

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
        F2 = Q2.dot(V)
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
        V_static_tmp[:,i] = linsolve(K_tmp, f)
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
    '''
    K = mechanical_system.K()
    M = mechanical_system.M()

    lambda_, V = sp.sparse.linalg.eigsh(K, M=M, k=n, sigma=0, which='LM',
                                        maxiter=100)
    omega = np.sqrt(lambda_)

    if save:
        for i, om in enumerate(omega):
            mechanical_system.write_timestep(om, V[:, i])

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
    return U[:,:n], sigma[:n]
