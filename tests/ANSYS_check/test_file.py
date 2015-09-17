# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy as sp

def read_hbmat(filename):
    '''
    reads the hbmat file and returns it in the csc format. 
    
    Parameters
    ----------
    filename : string
        string of the filename
    
    Returns
    -------
    matrix : sp.sparse.csc_matrix
        matrix which is saved in harwell-boeing format
    
    Info
    ----
    Information on the Harwell Boeing format: 
    http://people.sc.fsu.edu/~jburkardt/data/hb/hb.html
    '''
    with open(filename, 'r') as infile:
        matrix_data = infile.read().splitlines()
    
    # Analsysis of further line indices
    n_total, n_indptr, n_indices, n_data, n_rhs = map(int, matrix_data[1].split())
    matrix_keys, n_rows, n_cols, _, _ = matrix_data[2].split()
    
    n_rows, n_cols = int(n_rows), int(n_cols)

    symmetric = False
    if matrix_keys[1] == 'S':
        symmetric = True

    idx_0 = 4
    if n_rhs > 0:
        idx_0 += 1

    indptr = sp.zeros(n_indptr, dtype=int)
    indices = sp.zeros(n_indices, dtype=int)
    data = sp.zeros(n_data)
    
    for i, j in enumerate(sp.arange(idx_0, idx_0 + n_indptr)):
        indptr[i] = int(matrix_data[j])
    for i, j in enumerate(sp.arange(idx_0 + n_indptr, idx_0 + n_indptr + n_indices)):
        indices[i] = int(matrix_data[j])
    for i, j in enumerate(sp.arange(idx_0 + n_indptr + n_indices, 
                                    idx_0 + n_indptr + n_indices + n_data)):
        # consider, that fortran saves double as 2.3454323D+12
        data[i] = float(matrix_data[j].replace('D', 'E'))
    
    # take care of the indexing notation of fortran
    indptr -= 1
    indices -= 1

    matrix = sp.sparse.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    if symmetric:
        diagonal = matrix.diagonal()
        matrix = matrix + matrix.T
        matrix.setdiag(diagonal)
    return matrix


#%%

matrix_data = read_hbmat('m_mat.matrix')
matrix_data.A

import matplotlib.pyplot as plt
plt.matshow(matrix_data.A)
