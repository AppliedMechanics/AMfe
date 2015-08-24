# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:37 2015

@author: johannesr
"""

import numpy as np
import scipy as sp


def scatter_geometric_matrix(Mat, ndim):
    '''
    Scatter the symmetric geometric stiffness matrix to all dofs.

    What is basically done is to perform the kron(Mat, eye(ndof))

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

    '''
    dof_small_row = Mat.shape[0]
    dof_small_col = Mat.shape[1]
    Mat_scattered = np.zeros((dof_small_row*ndim, dof_small_col*ndim))
    for i in range(dof_small_row):
        for j in range(dof_small_col):
            for k in range(ndim):
                Mat_scattered[ndim*i+k,ndim*j+k] = Mat[i,j]
    return Mat_scattered



voigt_dof_dict = {1: 1, 2: 3, 3: 6}

def compute_B_matrix(F, B_tilde):
    '''
    Compute the B-matrix used in Total Lagrangian Finite Elements.

    Parameters
    ----------
    F : ndarray
        deformation gradient
    B_tilde : ndarray
        Matrix of the spatial derivative of the shape functions
        the columns form the nodes, the lines form the directions, in which the derivatives are taken

    Returns
    -------
    B : ndarray
        B matrix such that {dE} = B @ {u^e}

    Notes
    -----
    When the Voigt notation is used in this Reference, the variables are denoted with curly brackets.
    '''
    no_of_nodes = B_tilde.shape[1]
    no_of_dims = B_tilde.shape[0] # spatial dofs per node, i.e. 2 for 2D or 3 for 3D
    b = B_tilde
    B = np.zeros((voigt_dof_dict[no_of_dims], no_of_nodes*no_of_dims))
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
            [F11*b[0,i], F21*b[0,i]],
            [F12*b[1,i], F22*b[1,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]]]
        else:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
            [F11*b[0,i], F21*b[0,i], F31*b[0,i]],
            [F12*b[1,i], F22*b[1,i], F32*b[1,i]],
            [F13*b[2,i], F23*b[2,i], F33*b[2,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]], F31*b[1,i]+F32*b[0,i],
            [F12*b[2,i] + F13*b[1,i], F22*b[2,i] + F23*b[1,i]], F32*b[2,i]+F33*b[1,i],
            [F13*b[0,i] + F11*b[2,i], F23*b[0,i] + F21*b[2,i]], F33*b[0,i]+F31*b[2,i]]
    return B

compute_B_matrix(np.eye(2), sp.rand(2,5))
