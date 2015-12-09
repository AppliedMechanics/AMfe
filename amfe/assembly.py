#!/bin/env python
# -*- coding: utf-8 -*-
"""
Basic assembly module for the finite element code. Assumes to have all elements 
in the inertial frame.
Created on Tue Apr 21 11:13:52 2015

@author: Johannes Rutzmoser
"""

import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg
import time

fortran_use = False
try:
    import amfe.f90_assembly
    fortran_use = True
except:
    print('''
Python was not able to load the fast fortran element routines.
run the script 

f2py/install_fortran_routines.sh 

in order to get the full speed! 
''')

def get_index_of_csr_data(i,j, indptr, indices):
    '''Get the value index of the i,j element of a matrix in CSR format.

    Parameters
    ----------

    i : int
        row index of the CSR-matrix
    j : int
        column index of the CSR-Matrix
    indptr : ndarray
        index-ptr-Array of the CSR-Matrix.
    indices : ndarray
        indices of the row entries to the given value matrix.

    Returns
    -------

    k : int
        index of the value array of the CSR-matrix, in which value [i,j] is 
        stored.

    Notes
    -----

    This routine works only, if the tuple i,j is acutally a real entry of the 
    Matrix. Otherwise the value k=0 will be returned and an Error Message will 
    be provided.
    '''

    k = indptr[i]
    while j != indices[k]:
        k += 1
        if k > indptr[i+1]:
            print('ERROR! The index in the csr matrix is not preallocated!')
            k = 0
            break
    return k

def fill_csr_matrix(indptr, indices, vals, K, k_indices):
    '''
    Fill the values of K into the vals-array of a sparse CSR Matrix given the 
    k_indices array.

    Parameters
    ----------

    indptr : ndarray
        indptr-array of a preallocated CSR-Matrix
    indices : ndarray
        indices-array of a preallocated CSR-Matrix
    vals : ndarray
        vals-array of a preallocated CSR-Marix
    K : ndarray
        'small' sqare array which values will be distributed into the CSR-Matrix
        Shape is (n,n)
    k_indices : ndarray
        mapping array of the global indices for the 'small' K array.
        The (i,j) entry of K has the global indices (k_indices[i], k_indices[j])
        Shape is (n,)

    Returns
    -------

    None

    '''
    ndof_l = K.shape[0]
    for i in range(ndof_l):
        for j in range(ndof_l):
            l = get_index_of_csr_data(k_indices[i], k_indices[j], indptr, indices)
            vals[l] += K[i,j]
    pass


def compute_csr_assembly_indices(global_element_indices, indptr, indices):
    '''
    Computes the assembly-indices for matrices in andvance.

    This function is deprecated. It is not clear, if the function will be used
    in the future, but it seems that it could make sense for small systems
    when FORTRAN is not aviailible.

    '''
    no_of_elements, dofs_per_element = global_element_indices.shape
    matrix_assembly_indices = np.zeros((no_of_elements, dofs_per_element, dofs_per_element))
    for i in range(no_of_elements):
        for j in range(dofs_per_element):
            for k in range(dofs_per_element):
                row_idx = global_element_indices[i,j]
                col_idx = global_element_indices[i,k]
                matrix_assembly_indices[i, j, k] = \
                    get_index_of_csr_data(row_idx, col_idx, indptr, indices)
    return matrix_assembly_indices


if fortran_use:
    '''
    Fortran routine that will override the functions above for massive speedup.
    '''
    get_index_of_csr_data = amfe.f90_assembly.get_index_of_csr_data
    fill_csr_matrix = amfe.f90_assembly.fill_csr_matrix



class Assembly():
    '''
    Class for the more fancy assembly of meshes with non-heterogeneous elements.
    '''
    def __init__(self, mesh): 
        '''
        Parameters
        ----
        mesh : instance of the Mesh-class

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.mesh = mesh
        self.save_stresses = False
        pass

    def preallocate_csr(self):
        '''
        Precompute the values and allocate the matrices for efficient assembly.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        
        
        Internal variables computed:
        ----------------------------

        C_csr : scipy.sparse.csr.csr_matrix
            Matrix containing the sparsity pattern of the problem
        csr_assembly_indices : np.ndarray
            Array containing the indices for the csr matrix assembly routine.
            The entry [i,j,k] contains the index of the csr-value-matrix
            for the i-th element and the j-th row and the k-th column
            of the local stiffness matrix of the i-th element.
            The dimension is (n_elements, ndof_element, ndof_element).
        element_indices : list
            List containing the global indices for the local variables of an 
            element. The entry [i,j] gives the index in the global vector of 
            element i with dof j
        nodes_voigt : np.ndarray
            vector of all nodal coordinates in voigt-notation. 
            Dimension is (ndofs_total, )

        Notes
        -----
        This preallocation routine can take some while for small matrices.

        '''
        print('Preallocating the stiffness matrix')
        t1 = time.clock()
        # computation of all necessary variables:
        ele_nodes = self.mesh.ele_nodes
        no_of_dofs_per_node = self.mesh.no_of_dofs_per_node
        no_of_elements = self.mesh.no_of_elements
        no_of_dofs = self.mesh.no_of_dofs
        self.nodes_voigt = self.mesh.nodes.reshape(-1)
        
#        nodes_per_element = ele_nodes.shape[-1]
#        dofs_per_element = nodes_per_element*no_of_dofs_per_node
#        dofs_total = self.node_coords.shape[0]
#        no_of_local_matrix_entries = dofs_per_element**2
        # 
        self.element_indices = \
        [np.array([(np.arange(no_of_dofs_per_node) + no_of_dofs_per_node*i) for i in nodes], 
                   dtype=int).reshape(-1) for nodes in ele_nodes]
        
        max_dofs_per_element = np.max([len(i) for i in self.element_indices])


        # Auxiliary Help-Matrix H
        H = np.zeros((max_dofs_per_element, max_dofs_per_element))

        # preallocate the CSR-matrix
        row_global = np.zeros(no_of_elements*max_dofs_per_element**2, dtype=int)
        col_global = row_global.copy()
        vals_global = np.zeros_like(col_global, dtype=float)

        for i, indices_of_one_element in enumerate(self.element_indices):
            l = len(indices_of_one_element)
            H[:l,:l] = indices_of_one_element
            row_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.reshape(-1)
            col_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.T.reshape(-1)

        self.C_csr = sp.sparse.csr_matrix((vals_global, (row_global, col_global)),
                                          shape=(no_of_dofs, no_of_dofs))
        t2 = time.clock()
        print('Done preallocating stiffness matrix with', no_of_elements, 'elements', 
              'and', no_of_dofs, 'dofs.')
        print('Time taken for preallocation:', t2 - t1, 'seconds.')


    def assemble_matrix_and_vector(self, u, decorated_matrix_func):
        '''
        Assembles the matrix and the vector of the decorated matrix func.

        Parameters
        ----------

        u : ndarray
            global displacement array
        decorated_matrix_func : function
            function which works like

            K_local, f_local = func(index, X_local, u_local)

        Returns
        -------

        K_csr : sp.sparse.csr_matrix
            Assembled matrix in csr-format (Compressed sparse row)
        f : ndarray
            array of the assembled vector
        '''
        K_csr = self.C_csr.copy()
        f_glob = np.zeros(self.mesh.no_of_dofs)

        # Schleife ueber alle Elemente 
        # (i - Elementnummer, indices - DOF-Nummern des Elements)
        for i, indices in enumerate(self.element_indices): 
            # X - zu den DOF-Nummern zugehoerige Koordinaten (Positionen)
            X = self.nodes_voigt[indices] 
            # Auslesen der localen Elementverschiebungen
            u_local = u[indices] 
            # K wird die Elementmatrix und f wird der Elementlastvektor zugewiesen
            K, f = decorated_matrix_func(i, X, u_local) 
            # Einsortieren des lokalen Elementlastvektors in den globalen Lastvektor
            f_glob[indices] += f 
            # Einsortieren der lokalen Elementmatrix in die globale Matrix
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices) 

        return K_csr, f_glob

    def assemble_k_and_f(self, u):
        '''
        Assembles the stiffness matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            unconstrained assembled force vector
        '''
        # define the function that returns K, f for (i, X, u)
        # sort of a decorator approach! 
        def k_and_f_func(i, X, u):
            return self.mesh.ele_obj[i].k_and_f_int(X, u)
            
        return self.assemble_matrix_and_vector(u, k_and_f_func)

    def assemble_m(self, u=None):
        '''
        Assembles the mass matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        M : ndarray
            unconstrained assembled mass matrix in sparse matrix csr-format.

        Examples
        ---------
        TODO
        '''
        def m_and_vec_func(i, X, u):
            return self.mesh.ele_obj[i].m_and_vec_int(X, u)
            
        if u == None:
            u = np.zeros_like(self.nodes_voigt)
        M, _ = self.assemble_matrix_and_vector(u, m_and_vec_func) 
        return M

