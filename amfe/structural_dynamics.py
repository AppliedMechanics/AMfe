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
from .reduced_basis import vibration_modes
from .linalg.linearsolvers import solve_sparse
from .linalg.orth import m_orthogonalize

__all__ = ['modal_assurance',
           'mac_criterion',
           'modal_analysis',
           'mass_orth',
           'force_norm',
           'rayleigh_coefficients',
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


def modal_analysis(K, M, n=10, shift=0.0, mass_norm=False):
    """
    modal analysis with stiffness matrix K and mass matrix M

    Parameters
    ----------
    K: array_like
        stiffness matrix
    M: array_like
        mass matris M
    n: int, optional
        number of modes desired
    shift: float, optional
        shift for the eigensolver to find frequencies near the shift
    mass_norm: bool
        Flag if eigenmodes shall be mass normalized

    Returns
    -------
    omega: numpy.array
        eigenfrequencies in rad/s
    V: numpy.array
        eigenshapes
    """
    omega, V = vibration_modes(K, M, n, shift)
    if mass_norm:
        V = m_orthogonalize(V, M)
    else:
        for i in range(V.shape[1]):
            V[:, i] = V[:, i]/np.linalg.norm(V[:, i])
    return omega, V


mass_orth = m_orthogonalize
mac_criterion = modal_assurance
