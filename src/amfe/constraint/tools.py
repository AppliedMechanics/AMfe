#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.linalg import eigvalsh

__all__ = ['constraints_scaling_factor',
           'validate_constraints_independent',
           ]


def constraints_scaling_factor(dt, K, M, D=None):
    r"""
    Calculates the ideal scaling factor for Lagrange Formulation

    .. math::
        s = k_r + \frac{d_r}{\Delta t} + \frac{m_r}{\Delta t^2}

    Where the index r means the mean value of the diagonal terms in the matrix
    k, d or m, respectively

    Parameters
    ----------
    dt : float
        time step width
    K : array_like
        stiffness matrix
    M : array_like
        mass matrix
    D : array_like, optional
        damping matrix

    Returns
    -------
    scaling : float
        scaling factor for the constraints to have the same order of
        magnitude as the matrices M, D and K.
    """
    ndof = M.shape[0]
    mr = M.diagonal().sum() / ndof  # is faster than calculating the norm
    # mr = sp.sparse.linalg.norm(M)

    kr = K.diagonal().sum() / ndof
    # kr = sp.sparse.linalg.norm(K)

    if D is not None:
        dr = D.diagonal().sum() / ndof
    else:
        dr = 0.0

    scaling = kr + dr/dt + mr / (dt ** 2)

    return scaling


def validate_constraints_independent(B, tol=1E-10):
    """
    Function to find out if the constraints are linearly dependent. If the
    constraints are linearly dependent then the system cannot be solved.

    Parameters
    ----------
    B: array_like
        B-matrix of the constraints, e.g. the jacobian of the holonomic residual g_holo
    tol: float, optional
        tolerance with which the check is carried out. If a squared singular value is below
        this tolerance, it is considered that there are linear dependent lines
    """
    # Gilbert Strang singular value decomposition
    B_BT = B @ B.T
    omega_2 = eigvalsh(B_BT)

    if not np.any(omega_2 < tol):
        print("YES. The constraints are linearly independent within this",
              "tolerance. That means the system most likely can be solved.")
        return True

    else:
        print("NO, WARNING! There are", np.sum(omega_2 < 1E-10),
              "constraints that are linearly dependent within",
              "this tolerance. That means the system cannot be solved",
              "with these constraints and this tolerance.")
        return False
