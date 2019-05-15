#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Methods to generate reduction bases
"""

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator, eigsh, splu

from amfe.linalg.linearsolvers import ScipySparseLinearSolver
from amfe.linalg.orth import m_orthogonalize
from amfe.linalg.tools import arnoldi

__all__ = ['krylov_basis',
           'craig_bampton',
           'pod',
           'modal_derivatives',
           'static_derivatives',
           'shifted_static_derivatives',
           'modal_derivatives_cruz',
           'augment_with_derivatives',
           'augment_with_ranked_derivatives',
           'ranking_of_weighting_matrix',
           ]


def krylov_basis(M, K, b, n=3, omega=0.0, mass_orth=True,
                 n_iter_orth=1):
    r"""
    Computes the Krylov Subspace associated with the input matrix b at the
    frequency omega.

    .. math::
        \mathcal{K}(K^{-1} M, K^{-1} b)

    or with omega shift

    .. math::
        \mathcal{K}( (K - \omega^2 M)^{-1} M, (K - \omega^2 M)^{-1} b)

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
    n : int, optional
        number of moments matched. Default value 3.
    mass_orth : bool, optional
        flag for setting orthogonality of returned Krylov basis vectors. If
        True, basis vectors are mass-orthogonal (V.T @ M @ V = eye). If False,
        basis vectors are orthogonal (V.T @ V = eye)
    n_iter_orth : int
        Number of iterations for mass orthogonalization

    Returns
    -------
    V : ndarray
        Krylov basis where vectors V[:,i] give the basis vectors.

    References
    ----------

    """
    ndim = M.shape[0]
    no_of_inputs = b.size//ndim
    r = b.reshape(-1, no_of_inputs).copy()
    V = np.zeros((ndim, n*no_of_inputs))
    A = K - omega**2 * M

    solver = ScipySparseLinearSolver()

    # compute first krylov vector (K-om**2*M)^(-1) * b
    rows, cols = r.shape
    first_krylov_vectors = np.zeros((rows, cols), dtype=float)
    for i, rhs in enumerate(r.T):
        first_krylov_vectors[:, i] = solver.solve(A, rhs)

    def matvec(v):
        result = M @ solver.solve(A, v)
        return result

    linear_operator = LinearOperator(shape=A.shape, matvec=matvec)

    V = arnoldi(linear_operator, first_krylov_vectors, n, Vout=V)

    # mass-orthogonalization of V:
    if mass_orth:
        m_orthogonalize(V, M, niter=n_iter_orth)
    return V


def craig_bampton(K, M, interface_dofs, no_of_modes=5,
                  linsolvefunc=None, linsolvekwargs=None, omega_spurious=1.0, shift=0.0):
    """
    Computes the Craig-Bampton basis for the System K and M with the interface dofs given at
    the passed dof indices

    Parameters
    ----------
    K : ndarray
        Stiffness matrix of the system.
    M : ndarray
        Mass matrix of the system
    interface_dofs : list or ndarray
        list containing the dof numbers (zero indexed) that are the interface dofs
    no_of_modes : int, optional
        Number of fixed interface modes
        Default is 5.
    linsolvefunc : function
        linear solver function with signature x = func(A, b) for solving Ax=b
        This solve is called for the static part of the basis
    linsolvekwargs : dict
        keyword arguments for the linear solver
    omega_spurious : float
        During the solution of the fixed interface modes, spurious modes are introduced which are removed afterwards
        These spurious modes have a frequency of omega_spurious. They are removed by identifying these modes
        by their eigenfrequency. This could cause problems if a real eigenfrequency is nearly the eigenfrequency of the
        spurious modes. If this is the case one can shift the eigenfrequency of th spurious modes by setting this
        parameter.
    shift : float, optional
        Pass the shift for the fixed interface modes. This makes the eigensolver find frequencies near this shift.
        Default is 0.0 which means the lowest eigenfrequencies are searched for.

    Returns
    -------
    V : ndarray
        Basis consisting of static displacement modes and fixed interface modes

    omega : ndarray
        eigenfrequencies of the fixed interface modes.

    Examples
    --------
        >>> V, omega = craig_bampton(K, M, [0, 1, 105, 106], 10)

    References
    ----------
    [1] Géradin, M., & Rixen, D. J. (2014). Mechanical vibrations: theory and application to structural dynamics.
        John Wiley & Sons.

    """
    # Check if linear solve function is passed else create default solver
    if linsolvefunc is None:
        linearsolver = ScipySparseLinearSolver()
        linsolvefunc = linearsolver.solve
    if linsolvekwargs is None:
        linsolvekwargs = dict()

    # Number of dofs of the full system
    ndof = K.shape[0]

    # ---------------------------------------------------------------------------
    # Static modes (Guyan part)
    #
    # Preallocate Static basis (Guyan Basis)
    V_static = np.zeros((ndof, len(interface_dofs)))
    # Copy the K array to not overwrite the passed by referenced K
    K_tmp = K.copy()
    # Set the rows and columns of interface dofs to zero
    K_tmp[:, interface_dofs] *= 0
    K_tmp[interface_dofs, :] *= 0
    # Set the diagonal of interface dofs to 1
    K_tmp[interface_dofs, interface_dofs] = 1

    # For each interface dof solve the static problem
    # For better understanding: The following problem is solved:
    #
    #  -           - -      -     -      -
    #  | 1    0    | | v_bk |  =  | 1_k  |
    #  | 0    K_ii | | v_ik |     | -K_ik|
    #  -           - –      -     -      -
    #
    # This leads to v_bk = 1_k (first line of the linear equation
    # and second line K_ii v_ik = - K_ik     <=>    v_ik = K_ii^(-1) K_ik
    #
    for i, index in enumerate(interface_dofs):
        f = - K[:, index]
        f[interface_dofs] = 0
        f[index] = 1
        V_static[:, i] = linsolvefunc(K_tmp, f, **linsolvekwargs)

    # --------------------------------------------------------------------------
    # fixed interface modes
    #
    M_tmp = M.copy()
    # Attention: introducing eigenvalues of magnitude omega_spurious into the system
    M_tmp[:, interface_dofs] *= 0
    M_tmp[interface_dofs, :] *= 0
    M_tmp[interface_dofs, interface_dofs] = 1E0
    K_tmp[interface_dofs, interface_dofs] = omega_spurious**2
    # Solve eigenvalue problem
    omega, V_dynamic = eigsh(K_tmp, no_of_modes+len(interface_dofs), M_tmp, sigma=shift)
    # Find spurious modes if any
    indexlist = np.nonzero(np.round(omega - omega_spurious**2, 3))[0]
    # Remove spuious modes
    omega = np.sqrt(omega[indexlist])
    V_dynamic = V_dynamic[:, indexlist]

    return np.concatenate((V_static, V_dynamic[:, :no_of_modes]), axis=1), omega[:no_of_modes]


def pod(S, n=None, tol=1e-16):
    """
    Compute the POD basis of a training set S

    Parameters
    ----------
    S : array_like
        training set (rows=coordinates, columns=different training vectors)
    n : int
        Number of POD basis vectors which should be returned.
    tol : float
        if tol is given all left singular vectors will be returned whose singular value is greater than tol

    Returns
    -------
    sigma : ndarray
        Array of the singular values.
    V : ndarray
        Array containing the POD vectors. V[:,0] contains the POD-vector
        associated with sigma[0] etc.

    """
    U, sigma, __ = sp.linalg.svd(S, full_matrices=False)
    U_return = U[:, sigma > tol]
    if n is not None:
        if U_return.shape[1] < n:
            print('Warning the tolerance would lead to less pod vectors than desired')
        return sigma[:n], U[:, :n]
    else:
        return sigma[sigma > tol], U_return[:, :n]


def nelson_method(A_func, X0, lambda0, p_directions, p0=None, M=None, dA_dp=None,
                  finite_diff='central', h=1.0, verbose=True, out=None):
    r"""
    Computes the derivatives of eigenvectors w.r.t to parameters p of an Eigenvalue Problem

    .. math::
        (A(p) - \lambda_i M) x_i = 0

    Parameters
    ----------
    A_func: callable
        Returns A for a given p
    X0: array_like
        Eigenvectors at p_0
    lambda0: Iterable
        Eigenvalues at p_0
    p_directions: array_like, ndmin=2
        Parameter directions w.r.t which the derivatives shall be computed
    p0: array_like
        p_0 around which the derivatives are calculated
    M: array_like
        matrix M (not callable or dependent on p)
    dA_dp: callable
        If available a function that returns the derivative can be passed
        Otherwise a finite difference scheme can be used
    finite_diff: str, {'central', 'forward', 'backward'}
        If a finite difference scheme for computing the jacobian of A is used,
        the central, forward or backward scheme can be chosen
    h: float
        Stepsize for finite difference scheme if used
    verbose: bool, default: True
        Flag for verbose mode

    Returns
    -------
    Theta: array_like
        3 Dimensional Array containing the derivatives of the eigenvectors
        First dimension: Coordinates of the eigenvectors
        Second dimension: number of eigenvector
        Third dimension: number of parameter coordinate
    """
    no_of_coordinates, no_of_eigenvectors = X0.shape
    no_of_parameter_coords, no_of_directions = p_directions.shape
    if p0 is None:
        p0 = np.zeros(no_of_parameter_coords)

    if dA_dp is not None:
        raise NotImplementedError('Nelsons method is not implemented for given dA_dp')

    if M is None:
        M = np.eye(no_of_coordinates)

    if out is None:
        Theta = np.zeros((no_of_coordinates, no_of_eigenvectors, no_of_directions))
    else:
        Theta = out

    for i in range(no_of_eigenvectors):  # looping over the columns
        x_i = X0[:, i]
        A_dyn_i = A_func(p0) - lambda0[i] * M

        # fix the point with the maximum displacement of the vibration mode
        fix_idx = np.argmax(abs(x_i))
        A_dyn_i[:, fix_idx], A_dyn_i[fix_idx, :], A_dyn_i[fix_idx, fix_idx] = 0, 0, 1

        # factorization of the dynamic stiffness matrix
        if verbose:
            print('Factorizing the dynamic A matrix for eigenvalue',
                  '{0:d} with {1:4.2f}.'.format(i, lambda0[i]))
        solve_k_dyn = splu(A_dyn_i)

        for j in range(no_of_directions):  # looping over the directions
            p_j = p_directions[:, j]

            dA_dp_j = jacobian_finite_difference(A_func, p_j, p0, h, finite_diff)

            d_lambda_d_x_i = x_i.dot(dA_dp_j).dot(x_i)
            F_i = (d_lambda_d_x_i * M - dA_dp_j).dot(x_i)
            F_i[fix_idx] = 0
            v_i = solve_k_dyn.solve(F_i)
            c_i = - v_i.dot(M).dot(x_i)
            Theta[:, i, j] = v_i + c_i * x_i
    return Theta


def jacobian_finite_difference(A_func, direction, x0=None, h=1.0, method='central'):
    r"""
    Computes the jacobian of matrix A(x) in a certain direction

    .. math::
        \left. \frac{\partial A}{\partial x}\right|x0 \cdot v

    where v is the direction of the derivative

    Parameters
    ----------
    A_func: callable
        Function that returns A for a given x
    direction: array_like
        direction of the derivative
    x0: array_like
        point at which the derivative is evaluated
    h: float
        stepsize for finite difference scheme
    method: str, {'central', 'forward', 'backward'}
        finite difference scheme

    Returns
    -------
    jac: array_like
        Jacobian

    """
    if x0 is None:
        x0 = np.zeros_like(direction)
    # finite difference scheme
    if method == 'central':
        jac = (A_func(x0 + h * direction) - A_func(x0 - h * direction)) / (2 * h)
    elif method == 'upwind' or method == 'forward':
        jac = (A_func(x0 + h * direction) - A_func(x0)) / h
    elif method == 'backward':
        jac = (A_func(x0) - A_func(-h * direction)) / h
    else:
        raise ValueError('Finite difference scheme is not valid.')
    return jac


def modal_derivatives(V, omega, K_func, M, x0=None, h=1.0, verbose=True,
                      symmetric=True, finite_diff='central', out=None):
    r"""
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
    x0 : ndarray
        u vector around which the static derivatives shall be computed
    h : float, optional
        step width for finite difference scheme. Default value is 1.0
    verbose : bool, optional
        flag for verbosity. Default value: True
    symmetric : bool, optional
        flag for making the modal derivative matrix theta symmetric. Default is
        `True`.
    finite_diff : str {'central', 'upwind'}
        Method for finite difference scheme. 'central' computes the finite difference
        based on a central difference scheme, 'upwind' based on an upwind scheme. Note
        that the upwind scheme can cause severe distortions of the modal derivative.
    out : ndarray, optional
        ndarray where static derivatives shall be written to

    Returns
    -------
    Theta : ndarray
        three dimensional array of modal derivatives. Theta[:,i,j] contains
        the modal derivative 1/2 * dx_i / dx_j. The basis Theta is made symmetric, so
        that `Theta[:,i,j] == Theta[:,j,i]` if `symmetric=True`.

    See Also
    --------
    static_derivatives : modal derivative with mass neglection but much faster in computation.
    """

    no_of_dofs, no_of_modes = V.shape

    # Check, if V is mass normalized:
    if not np.allclose(np.eye(no_of_modes), V.T @ M @ V, rtol=1E-5, atol=1E-8):
        m_orthogonalize(V, M)

    if x0 is None:
        x0 = np.zeros(no_of_dofs)

    lambda0 = [om**2 for om in omega]

    if out is None:
        Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes), dtype=float)
    else:
        Theta = out

    Theta = nelson_method(K_func, V, lambda0, V, p0=x0, M=M, finite_diff=finite_diff, h=h, verbose=verbose,
                          out=Theta)

    if symmetric:
        Theta = 1/2*(Theta + Theta.transpose((0, 2, 1)))
    return Theta


def static_derivatives(V, K_func, M=None, shift=None, x0=None, h=1.0,
                       verbose=True, symmetric=True,
                       finite_diff='central', out=None):
    """
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
    shift : float, optional
        shift frequency in rad/s. Default value is 0.
    x0 : ndarray
        u vector around which the static derivatives shall be computed
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True
    symmetric : bool, optional
        flag for making the modal derivative matrix theta symmetric. Default is
        `True`.
    finite_diff : str {'central', 'forward', backward}
        Method for finite difference scheme. 'central' computes the finite
        difference based on a central difference scheme, 'forward' based on an
        forward scheme etc. Note that the upwind scheme can cause severe
        distortions of the static correction derivative.
    out : ndarray, optional
        ndarray where static derivatives shall be written to

    Returns
    -------
    Theta : ndarray
        three dimensional array of static modal derivatives. Theta[:,i,j]
        contains the static derivative 1/2 * dx_i / dx_j. As the static
        derivatives are symmetric, Theta[:,i,j] == Theta[:,j,i].

    See Also
    --------
    modal_derivative
    """

    no_of_dofs, no_of_modes = V.shape

    if x0 is None:
        x0 = np.zeros(no_of_dofs)

    if out is None:
        Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes), dtype=float)
    else:
        Theta = out

    if shift is not None:
        if (shift > 0.0) and (M is not None):
            K_dyn0 = K_func(x0) - shift**2 * M
        else:
            raise ValueError('shift must be positive and M must be given')
    else:
        K_dyn0 = K_func(x0)

    print('Factorizing K...', end='')
    solve_k_dyn = splu(K_dyn0)
    print('finished')

    for j, v_j in enumerate(V.T):
        if verbose:
            print('Computing finite difference K-matrix')
        dK_dv_j = jacobian_finite_difference(K_func, v_j, x0, h, finite_diff)
        b = - dK_dv_j @ V
        if verbose:
            print('Solving linear system #', j)
        Theta[:, :, j] = solve_k_dyn.solve(b)
        if verbose:
            print('Done solving linear system #', j)
    if verbose:
        residual = np.linalg.norm(Theta - Theta.transpose(0, 2, 1)) / \
                   np.linalg.norm(Theta)
        print('The residual, i.e. the unsymmetric values, are', residual)
    if symmetric:
        # make Theta symmetric
        Theta = 1/2*(Theta + Theta.transpose(0, 2, 1))
    return Theta


def shifted_static_derivatives(V, K_func, M, Shifts, x0=None, h=1.0, verbose=True,
                               symmetric=False, finite_diff='central', out=None):
    r"""
    Computes the shifted Modal Derivatives with different shifts for each derivative

    .. math::
        (K(x_0) - s_{ij}^2 M ) \cdot \Theta_{ij} = - \nabla_{V_j} K \cdot V_i


    V : ndarray
        array containing the linear basis
    K_func : function
        function returning the tangential stiffness matrix for a given
        displacement. Has to work like `K = K_func(u)`.
    M : ndarray
        mass matrix. Can be sparse or dense. If `None` is given, the mass of 0
        is assumed. Default value is `None`.
    Shifts : ndarray
        matrix that contains shifts for each derivative
        The derivative is then calculated by solving
        (K - Shift[i, j]**2 * M) Theta[i, j] = - dK/dV[:,j] * V[:, i]
    x0 : ndarray
        u vector around which the static derivatives shall be computed
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True
    symmetric : bool, optional
        flag for making the modal derivative matrix theta symmetric. Default is
        `True`.
    finite_diff : str {'central', 'forward', backward}
        Method for finite difference scheme. 'central' computes the finite
        difference based on a central difference scheme, 'forward' based on an
        forward scheme etc. Note that the upwind scheme can cause severe
        distortions of the static correction derivative.
    out : ndarray, optional
        ndarray where static derivatives shall be written to

    Returns
    -------
    Theta : ndarray
        three dimensional array of static modal derivatives. Theta[:,i,j]
        contains the static derivative 1/2 * dx_i / dx_j. As the static
        derivatives are symmetric, Theta[:,i,j] == Theta[:,j,i].
    """
    rows, cols = Shifts.shape
    no_of_dofs = V.shape[0]

    if out is None:
        Theta = np.zeros((no_of_dofs, rows, cols))
    else:
        Theta = out

    if x0 is None:
        x0 = np.zeros(no_of_dofs)

    for i in range(rows):
        for j in range(cols):
            K_dyn = K_func(x0) - Shifts[i, j]**2 * M
            solve_k_dyn = splu(K_dyn)
            if verbose:
                print('Computing finite difference K-matrix')
            dK_dv_j = jacobian_finite_difference(K_func, V[:, j], x0, h, finite_diff)
            b = - dK_dv_j.dot(V[:, i])
            if verbose:
                print('Solving linear system #', i)
            Theta[:,i,j] = solve_k_dyn.solve(b)
            if verbose:
                print('Done solving linear system #({}, {})'.format(i,j))
    if verbose:
        residual = np.linalg.norm(Theta - Theta.transpose(0, 2, 1)) / \
                   np.linalg.norm(Theta)
        print('The residual, i.e. the unsymmetric values, are', residual)
    if symmetric:
        # make Theta symmetric
        Theta = 1/2*(Theta + Theta.transpose(0, 2, 1))
    return Theta


def modal_derivatives_cruz(V, K_func, M, omega, x0=None, h=1.0,
                           verbose=True, symmetric=True,
                           finite_diff='central', out_theta=None, out_theta_tilde=None):
    """
    Compute the derivatives developed by Maria Cruz.
    These are two different shifted static derivatives.
    The first is shifted by :math:`\omega_i + \omega_j` and the second by :math:`\omega_i - \omega_j`.

    Parameters
    ----------
    V : ndarray
        array containing the eigenmodes
    K_func : function
        function returning the tangential stiffness matrix for a given
        displacement. Has to work like `K = K_func(u)`.
    M : ndarray
        mass matrix. Can be sparse or dense. If `None` is given, the mass of 0
        is assumed. Default value is `None`.
    omega : ndarray
        frequencies of the modes
    x0 : ndarray
        u vector around which the static derivatives shall be computed
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True
    symmetric : bool, optional
        flag for making the modal derivative matrix theta symmetric. Default is
        `True`.
    finite_diff : str {'central', 'forward', backward}
        Method for finite difference scheme. 'central' computes the finite
        difference based on a central difference scheme, 'forward' based on an
        forward scheme etc. Note that the upwind scheme can cause severe
        distortions of the static correction derivative.
    out_theta : ndarray, optional
        Preallocated ndarray for writing Theta if desired
    out_theta_tilde : ndarray, optional
        Preallocated ndarray for writing Theta if desired

    Returns
    -------
    Theta : ndarray
        three dimensional array of shifted modal derivatives,
        shifted by -(w_i + w_j)*M. Theta[:,i,j]
        contains the static derivative 1/2 * dx_i / dx_j. As the shifted
        derivatives are symmetric, Theta[:,i,j] == Theta[:,j,i].
    Theta_tilde : ndarray
        three dimensional array of shifted modal derivatives,
        shifted by -(w_i - w_j)*M. Theta[:,i,j]
        contains the static derivative 1/2 * dx_i / dx_j. As the shifted
        derivatives are symmetric, Theta[:,i,j] == Theta[:,j,i].

    See Also
    --------
    modal_derivatives
    static_derivatives

    """

    no_of_dofs, no_of_modes = V.shape

    if x0 is None:
        x0 = np.zeros(no_of_dofs)

    # Preallocation
    if out_theta is None:
        Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes), dtype=float)
    else:
        Theta = out_theta

    if out_theta_tilde is None:
        Theta_tilde = np.zeros((no_of_dofs, no_of_modes, no_of_modes), dtype=float)
    else:
        Theta_tilde = out_theta_tilde

    Shifts = np.zeros((len(omega), len(omega)), dtype=float)
    Shifts_tilde = np.zeros_like(Shifts)
    for i, omi in enumerate(omega):
        for j, omj in enumerate(omega):
            Shifts[i, j] = omi + omj
            Shifts_tilde[i, j] = omi - omj

    Theta = shifted_static_derivatives(V, K_func, M, Shifts, x0=x0,
                                       h=h, verbose=verbose, symmetric=symmetric,
                                       finite_diff=finite_diff, out=Theta)
    Theta_tilde = shifted_static_derivatives(V, K_func, M, Shifts_tilde, x0=x0,
                                             h=h, verbose=verbose, symmetric=symmetric,
                                             finite_diff=finite_diff, out=Theta_tilde)
    return Theta, Theta_tilde


def merge_bases(V1, V2, atol=1e-14, rtol=1E-8, deflate=True, out=None):
    if out is None:
        out = np.zeros((V1.shape[0], V1.shape[1] + V2.shape[1]), dtype=float)

    V_raw = np.concatenate((V1, V2), axis=1)
    # Deflation algorithm
    U, s, V_svd = sp.linalg.svd(V_raw, full_matrices=False)
    if deflate:
        idx_defl = s > s[0] * rtol + atol
        out[:, :len(idx_defl)] = U[:, idx_defl]
        out = out[:, :len(idx_defl)]
    else:
        out[:, :] = V_raw[:, :]
    return out


def augment_with_derivatives(V=None, theta=None, M=None, tol=1E-8, symm=True, deflate=True):
    """
    Make a linear basis containing the subspace spanned by V and a third order tensor theta
    e.g. from modal derivatives by deflation.

    Parameters
    ----------
    V : ndarray
        linear basis
    theta : ndarray
        third order tensor filled with the modal derivatives associated with V
    M : ndarray, optional
        Mass matrix. If mass matrix is passed, the reduction basis augmentation
        will be M-normalized, otherwise it will just be normalized (2-norm)
    tol : float, optional
        Tolerance for the deflation via SVD. The omitted singular values are
        at least smaller than the largest one multiplied with tol.
        Default value: 1E-8.
    symm : bool, optional
        If set to true (default), theta will be assumed to be symmetric!
    deflate : bool, optional
        Flag for removing linear dependent vectors

    Returns
    -------
    V_ret : ndarray
        linear basis containing the subspace spanned by V and theta.

    """
    if V is None:
        ndof = theta.shape[0]
        n = 0
        l = theta.shape[1]
        m = theta.shape[2]
        if symm:
            V_raw = np.zeros((ndof, n + l * (l + 1) // 2))
        else:
            V_raw = np.zeros((ndof, n + l * m))
    else:
        ndof, n = V.shape
        l = theta.shape[1]
        m = theta.shape[2]
        if symm:
            V_raw = np.zeros((ndof, n + l * (l + 1) // 2))
        else:
            V_raw = np.zeros((ndof, n + l * m))
        V_raw[:, :n] = V[:, :]

    if symm:
        for i in range(l):
            for j in range(i+1):
                idx = n + i*(i+1)//2 + j
                if M is None:
                    theta_norm = np.sqrt(theta[:, i, j].T @ theta[:, i, j])
                else:
                    theta_norm = np.sqrt(theta[:, i, j].T @ M @ theta[:, i, j])
                V_raw[:, idx] = theta[:, i, j] / theta_norm
    else:
        for i in range(l):
            for j in range(m):
                idx = n + m*i + j
                if M is None:
                    theta_norm = np.sqrt(theta[:, i, j].T.dot(theta[:, i, j]))
                else:
                    theta_norm = np.sqrt(theta[:, i, j].T.dot(M).dot(theta[:, i, j]))
                V_raw[:, idx] = theta[:, i, j] / theta_norm

    # Deflation algorithm
    if deflate:
        U, s, V_svd = sp.linalg.svd(V_raw, full_matrices=False)
        idx_defl = s > s[0]*tol
        V_ret = U[:, idx_defl]
    else:
        V_ret = V_raw
    return V_ret


def augment_with_ranked_derivatives(V, Theta, W, n, tol=1E-8, symm=True):
    """
    Build a linear basis from a linear basis V and a third order Tensor Theta
    e.g. from modal or static derivatives, with a given weighting matrix W.

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

    """
    ranking = ranking_of_weighting_matrix(W, symm=symm)
    Theta_ranked = Theta[:, ranking[:, 0], ranking[:, 1]]
    V_raw = np.concatenate((V, Theta_ranked[:, :n]), axis=1)
    U, s, V_svd = sp.linalg.svd(V_raw, full_matrices=False)
    idx_defl = s > s[0]*tol
    V_red = U[:, idx_defl]
    return V_red


def ranking_of_weighting_matrix(W, symm=True):
    """
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

    """
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

    else:  # not symmetric
        ranking = np.zeros((n*m, 2), dtype=int)
        for i, val in enumerate(ranking_list):
            ranking[i,:] = val // m, val % m
    return ranking
