'''
Reduced basis methods...
'''

import numpy as np
import scipy as sp
from scipy import linalg

from .solver import solve_sparse, PardisoSolver

__all__ = ['krylov_subspace',
           'compute_modes_pardiso',
           'vibration_modes',
           'craig_bampton',
           'pod',
           'modal_derivatives',
           'static_derivatives',
           'augment_with_derivatives',
           'augment_with_ranked_derivatives',
           'ranking_of_weighting_matrix',
           ]


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
    LU_object = PardisoSolver(K - omega**2 * M)

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


def compute_modes_pardiso(mechanical_system, n=10, u_eq=None,
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
    K_mat = PardisoSolver(K)
    K_mat.factorize()

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
        Basis consisting of static displacement modes and internal vibration
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


def modal_derivatives(V, omega, K_func, M, h=1.0, verbose=True,
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

    Returns
    -------
    Theta : ndarray
        three dimensional array of modal derivatives. Theta[:,i,j] contains
        the modal derivative 1/2 * dx_i / dx_j. The basis Theta is made symmetric, so
        that `Theta[:,i,j] == Theta[:,j,i]` if `symmetic=True`.

    See Also
    --------
    static_derivatives : modal derivative with mass neglection but much faster in computation.
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
        LU_object = PardisoSolver(K_dyn_i)

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

def static_derivatives(V, K_func, M=None, omega=0, h=1.0,
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
    LU_object = PardisoSolver(K_dyn)
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


def augment_with_derivatives(V, theta, M=None, tol=1E-8, symm=True):
    '''
    Make a linear basis containing the subspace spanned by V and theta by
    deflation.

    Parameters
    ----------
    V : ndarray
        linear basis
    theta : ndarray
        third order tensor filled with the modal derivatices associated with V
    M : ndarray, optional
        Mass matrix. If mass matrix is passed, the reduction basis augmentation
        will be M-normalized, otherwise it will just be normalized (2-norm)
    tol : float, optional
        Tolerance for the deflation via SVD. The omitted singular values are
        at least smaller than the largest one multiplied with tol.
        Default value: 1E-8.
    symm : bool, optional
        If set to true (default), theta will be assumed to be symmetric!

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


def augment_with_ranked_derivatives(V, Theta, W, n, tol=1E-8, symm=True):
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
