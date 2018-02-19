'''
Structural Dynamics tools
'''

import numpy as np
from .reduced_basis import vibration_modes
from .solver import solve_sparse

modal_analysis = vibration_modes

__all__ = ['modal_assurance',
           'mass_orth',
           'force_norm',
           'rayleigh_coefficients',
           'give_mass_and_stiffness',
           ]

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


def give_mass_and_stiffness(mechanical_system):
    '''
    Determine mass and stiffness matrix of a mechanical system.

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem

    Returns
    -------
    M : ndarray
        Mass matrix of the mechanical system
    K : ndarray
        Stiffness matrix of the mechanical system

    '''

    K = mechanical_system.K()
    M = mechanical_system.M()
    return M, K


def rayleigh_coefficients(zeta, omega_1, omega_2):
    '''
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

    '''
    beta = 2*zeta/(omega_1 + omega_2)
    alpha = omega_1*omega_2*beta
    return alpha, beta
