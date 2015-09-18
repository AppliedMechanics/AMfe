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
    
    indptr[:] = list(map(int, matrix_data[idx_0 : idx_0 + n_indptr]))
    indices[:] = list(map(int, matrix_data[idx_0 + n_indptr : idx_0 + n_indptr + n_indices]))
    # consider the fortran convention with D instead of E in double precison floats
    data[:] = [float(x.replace('D', 'E')) for x in matrix_data[idx_0 + n_indptr + n_indices : ]]
    
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


def write_apdl_file(filename, nodes, elements, ANSYS_element, rho, E, nu):
    '''
    Write an apdl file which produces mass- and stiffness matrix for a single element. 
    
    Parameters
    ----------
    filename : string
        filename of the apdl file
    nodes : ndarray
        array containing the nodes of the element. The nodes are given in matrix format
    elements : ndarray
        array containing the nodes forming the elements. The array has to be a 2D array
    ANSYS_element : string
        string containing the elment key number and eventually the keyarguments for the APDL file
    rho : float
        density of the elements
    E : float
        Yong's modulus of the elements
    nu : float
        poisson ratio of the elements
        
    Returns
    -------
    None
    
    '''
    
    # check, if nodes are flat
    if nodes.shape[-1] == 2:
        nodes3D = sp.zeros((nodes.shape[0], 3))
        nodes3D[:,:2] = nodes
    else:
        nodes3D = nodes
        
    with open(filename + '.inp', 'w') as apdl:
        apdl.write('finish \n/clear,start \n/PREP7\n\n')
        apdl.write('!Nodal values\n')
        for i, node in enumerate(nodes3D):
            apdl.write('N,' + str(i+1) + ',' + ','.join(str(x) for x in node) + '\n')
        
        apdl.write('\n!Material properties\n')
        apdl.write('MP,DENS,1,' + str(rho) + '\n')
        apdl.write('MP,EX,1,' + str(E) + '\n')
        apdl.write('MP,NUXY,1,' + str(nu) + '\n')
        apdl.write('MAT,1\n')
        
        apdl.write('\n!Elements\n')
        apdl.write('ET,1,' + ANSYS_element + '\n')
        apdl.write('TYPE,1\n')
        for i, element in enumerate(elements):
            apdl.write('E,' + ','.join(str(x+1) for x in element) + '\n')
        apdl.write('\n!Solver\n')
        apdl.write('/SOLU \nantype,modal \nmodopt,lanb,1 \nmxpand,1 \nsolve\n')
        apdl.write('\n!Postprocessor\n')
        apdl.write('/POST1 \n/AUX2 \nFILE,,full\n')
        apdl.write('HBMAT,element' + ANSYS_element + '_m,ansmat,,ASCII,MASS,NO,NO\n')
        apdl.write('HBMAT,element' + ANSYS_element + '_k,ansmat,,ASCII,STIFF,NO,NO\n')
    
    print('File', filename + '.inp', 'written successfully')



matrix_data = read_hbmat('m_mat.matrix')
matrix_data.A

nodes = np.array([0,0,1,0,1,1,0,1.]).reshape((-1,2))
elements = np.array([0, 1, 2, 3]).reshape((1,-1))
write_apdl_file('my_testfile', nodes,  elements, '182', 1, 60, 0.3)

import matplotlib.pyplot as plt
plt.matshow(matrix_data.A)











