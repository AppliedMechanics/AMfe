#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Tools for all elements.
"""

__all__ = [
    'compute_B_matrix',
    'scatter_matrix'
]

import numpy as np

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


def scatter_matrix(Mat, ndim):
    """
    Scatter the symmetric (geometric stiffness) matrix to all dofs.

    What is basically done is to perform the np.kron(Mat, eye(ndim))

    Parameters
    ----------
    Mat : ndarray
        Matrix that should be scattered
    ndim : int
        number of dimensions of finite element. If it's a 2D element, ndim=2,
        if 3D, ndim=3

    Returns
    -------
    Mat_scattered : ndarray
        scattered matrix

    """
    dof_small_row = Mat.shape[0]
    dof_small_col = Mat.shape[1]
    Mat_scattered = np.zeros((dof_small_row*ndim, dof_small_col*ndim))

    for i in range(dof_small_row):
        for j in range(dof_small_col):
            for k in range(ndim):
                Mat_scattered[ndim*i+k,ndim*j+k] = Mat[i,j]
    return Mat_scattered


def compute_B_matrix(B_tilde, F):
    """
    Compute the B-matrix used in Total Lagrangian Finite Elements.

    Parameters
    ----------
    F : ndarray
        deformation gradient (dx_dX)
    B_tilde : ndarray
        Matrix of the spatial derivative of the shape functions: dN_dX

    Returns
    -------
    B : ndarray
        B matrix such that {delta_E} = B @ {delta_u^e}

    Notes
    -----
    When the Voigt notation is used in this Reference, the variables are
    denoted with curly brackets.
    """

    no_of_nodes = B_tilde.shape[0]
    no_of_dims = B_tilde.shape[1] # spatial dofs per node, i.e. 2 for 2D or 3 for 3D
    b = B_tilde
    B = np.zeros((no_of_dims*(no_of_dims+1)//2, no_of_nodes*no_of_dims))
    F11 = F[0,0]
    F12 = F[0,1]
    F21 = F[1,0]
    F22 = F[1,1]

    if no_of_dims == 3:
        F13 = F[0,2]
        F31 = F[2,0]
        F23 = F[1,2]
        F32 = F[2,1]
        F33 = F[2,2]

    for i in range(no_of_nodes):
        if no_of_dims == 2:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
                [F11*b[i,0], F21*b[i,0]],
                [F12*b[i,1], F22*b[i,1]],
                [F11*b[i,1] + F12*b[i,0], F21*b[i,1] + F22*b[i,0]]]
        else:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
                [F11*b[i,0], F21*b[i,0], F31*b[i,0]],
                [F12*b[i,1], F22*b[i,1], F32*b[i,1]],
                [F13*b[i,2], F23*b[i,2], F33*b[i,2]],
                [F12*b[i,2] + F13*b[i,1],
                     F22*b[i,2] + F23*b[i,1], F32*b[i,2] + F33*b[i,1]],
                [F13*b[i,0] + F11*b[i,2],
                     F23*b[i,0] + F21*b[i,2], F33*b[i,0] + F31*b[i,2]],
                [F11*b[i,1] + F12*b[i,0],
                     F21*b[i,1] + F22*b[i,0], F31*b[i,1] + F32*b[i,0]]]
    return B


# overloading routines with fortran routines
if use_fortran:
    compute_B_matrix = amfe.f90_element.compute_b_matrix
    scatter_matrix = amfe.f90_element.scatter_matrix