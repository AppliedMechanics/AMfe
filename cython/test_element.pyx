# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:37 2015

@author: johannesr
"""

#cimport numpy as cnp
import numpy as np

def scatter_geometric_matrix(double[:, ::1] Mat, int ndim):
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
    cdef int dof_small_row = Mat.shape[0]
    cdef int dof_small_col = Mat.shape[1]
    cdef double[:, ::1] Mat_scattered = np.zeros((dof_small_row*ndim, dof_small_col*ndim))
    cdef:
        int i
        int j
        int k
    for i in range(dof_small_row):
        for j in range(dof_small_col):
            for k in range(ndim):
                Mat_scattered[ndim*i+k,ndim*j+k] = Mat[i, j]
    return Mat_scattered

