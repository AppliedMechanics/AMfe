#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Structural Dynamics tools
"""

import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from .linalg.linearsolvers import solve_sparse

__all__ = ['modal_assurance',
           'mac_criterion',
           'modal_analysis',
           'force_norm',
           'rayleigh_coefficients',
           'vibration_modes',
           'vibration_modes_lanczos'
           ]


def modal_assurance(U, V):
    r"""
    Compute the Modal Assurance Criterion (MAC) of the vectors stacked
    in U and V:

    .. math::
        U = [u_1, \dots, u_n], V = [v_1, \dots, v_n] \\
        MAC_{i,j} = \frac{(u_i^Hv_j)^2}{u_i^T u_i \cdot v_j^H v_j}


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

    """
    nominator = (U.conj().T @ V)**2
    diag_u_squared = np.einsum('ij, ij -> j', U, U)
    diag_v_squared = np.einsum('ij, ij -> j', V, V)
    denominator = np.outer(diag_u_squared, diag_v_squared)
    return nominator / denominator


def force_norm(F, K, M, norm='euclidean'):
    """
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
    """
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


def rayleigh_coefficients(zeta, omega_1, omega_2):
    """
    Compute the coefficients for rayleigh damping such, that the modal damping
    for the given two eigenfrequencies is zeta.

    Parameters
    ----------
    zeta : float
        modal damping for modes 1 and 2
    omega_1 : float
        first eigenfrequency in [rad/s]
    omega_2 : float
        second eigenfrequency in [rad/s]

    Returns
    -------
    alpha : float
        rayleigh damping coefficient for the mass matrix
    beta : float
        rayleigh damping for the stiffness matrix

    """
    beta = 2*zeta/(omega_1 + omega_2)
    alpha = omega_1*omega_2*beta
    return alpha, beta


def vibration_modes(K, M, n=10, shift=0.0, mass_orth=False, normalized=False):
    """
    Compute the n first vibration modes of the given stiffness and mass matrix using the ARPACK Lanczos solver

    Parameters
    ----------
    K: ndarray or csr_matrix
        Stiffness Matrix
    M: ndarray or csr_matrix
        Mass Matrix
    n: int
        number of modes to be computed.
    shift: float
        shift the eigenvalue problem such that the eigenfrequencies around the shift (omega) are found
    mass_orth: bool
        Flag if modes shall be mass orthogonalized
    normalized: bool
        Flag if modes shall be normalized to unit vectors

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
    omega, Phi = vibration_modes(K, M, 10)

    Notes
    -----
    The core command using the ARPACK library is a little bit tricky. One has
    to use the shift inverted mode for the solution of the mechanical
    eigenvalue problem with the largest eigenvalues. Generally no convergence
    is gained when the smallest eigenvalue is to be found.

    If the squared eigenvalue omega**2 is negative, as it might happen due to
    round-off errors with rigid body modes, the negative sign is traveled to
    the eigenfrequency omega, though this makes physically no sense...
    """
    sigma = shift**2

    lambda_, V = sp.sparse.linalg.eigsh(K, M=M, k=n, sigma=sigma, which='LM',
                                        maxiter=100)
    omega = np.sqrt(abs(lambda_))
    # Little bit of sick hack: The negative sign is transferred to the
    # eigenfrequencies
    omega[lambda_ < 0] *= -1

    if mass_orth and normalized:
        raise ValueError('The eigenvectors cannot be mass orthogonal AND normalized')
    if mass_orth:
        for i, v in enumerate(V.T):
            V[:, i] = v/np.sqrt(v.T @ M @ v)
    if normalized:
        for i, v in enumerate(V.T):
            V[:, i] = v/np.linalg.norm(v)
    return omega, V


def vibration_modes_lanczos(K, M, n=10, shift=0.0, Kinv_operator=None, niter_max=40, rtol=1E-14):
    r"""
    Make a modal analysis using a naive Lanczos iteration.

    Parameters
    ----------
    K : array_like
        stiffness matrix
    M : array_like
        mass matrix
    n : int
        number of modes to be computed
    shift : float
        shift the eigenvalue problem such that the eigenfrequencies around the shift (omega) are found
    Kinv_operator : LinearOperator
        LinearOperator solving Kx=b (i.e. multiplication :math:`K^{-1} b`)
        if None, the scipy eigsh solver will be used instead
    niter_max : int, optional
        Maximum number of Lanczos iterations
    rtol : float, optional
        relative tolerance

    Returns
    -------
    om : ndarray, shape(n)
        eigenfrequencies of the system
    Phi : ndarray, shape(ndim, n)
        Vibration modes of the system

    Note
    ----
    In comparison to the vibration_modes method, this method can use different linear solvers
    available via the Kinv_operator method and a naive Lanczos iteration. The modal_analysis method
    uses the Arpack solver which is more accurate but takes also much longer for
    large systems, since the factorization is very inefficient by using superLU.
    """
    if Kinv_operator is None:
        return vibration_modes(K, M, n, shift)

    if shift != 0.0:
        raise NotImplementedError('The shift has not been implemented yet in the Lanczos solver')

    k_diag = K.diagonal().sum()

    # build up Krylov sequence
    n_rand = n
    n_dim = M.shape[0]

    residual = np.zeros(n)
    b = np.random.rand(n_dim, n_rand)
    b, _ = sp.linalg.qr(b, mode='economic')
    krylov_subspace = b

    def matvec(v):
        return Kinv_operator.dot(M.dot(v))

    A = LinearOperator(shape=(n_dim, n_dim), matvec=matvec)

    n_iter = niter_max + 1

    for n_iter in range(niter_max):
        print('Lanczos iteration # {}. '.format(n_iter), end='')
        new_directions = A.dot(b)
        krylov_subspace = np.concatenate((krylov_subspace, new_directions), axis=1)
        krylov_subspace, _ = sp.linalg.qr(krylov_subspace, mode='economic')
        b = krylov_subspace[:, -n_rand:]

        # solve interaction problem
        K_red = krylov_subspace.T @ K @ krylov_subspace
        M_red = krylov_subspace.T @ M @ krylov_subspace
        lambda_r, Phi_r = sp.linalg.eigh(K_red, M_red, overwrite_a=True, overwrite_b=True)
        Phi = krylov_subspace @ Phi_r[:, :n]

        # check the tolerance to be below machine epsilon with some buffer...
        for i in range(n):
            residual[i] = np.sum(abs((-lambda_r[i] * M + K) @ Phi[:, i])) / k_diag

        print('Res max: {:.2e}'.format(np.max(residual)))
        if np.max(residual) < rtol:
            break

    if n_iter - 1 == niter_max:
        print('No convergence gained in the given iteration steps.')

    print('The Lanczos solver took ' +
          '{} iterations to solve for {} eigenvectors.'.format(n_iter+1, n))
    omega = np.sqrt(abs(lambda_r[:n]))
    V = Phi[:,:n]
    # Little bit of sick hack: The negative sign is transferred to the
    # eigenfrequencies
    omega[lambda_r[:n] < 0] *= -1

    return omega, V


modal_analysis = vibration_modes
mac_criterion = modal_assurance
