#!/bin/env python
# -*- coding: utf-8 -*-
"""
Basic assembly module for the finite element code. Assumes to have all elements in the inertial frame.
Created on Tue Apr 21 11:13:52 2015

@author: Johannes Rutzmoser
"""


import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg

import multiprocessing as mp
from multiprocessing import Pool

fortran_use = False
try:
    import amfe.f90_assembly
    fortran_use = True
except:
    print('''
Python was not able to load the fast fortran routines (assembly).
run

(TODO)

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
        index of the value array of the CSR-matrix, in which value [i,j] is stored.

    Note
    ----

    This routine works only, if the tuple i,j is acutally a real entry of the Matrix.
    Otherwise the value k=0 will be returned and an Error Message will be provided.
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
    Fill the values of K into the vals-array of a sparse CSR Matrix given the k_indices array.

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
                matrix_assembly_indices[i, j, k] = get_index_of_csr_data(row_idx, col_idx, indptr, indices)
    return matrix_assembly_indices


if fortran_use:
    '''
    Fortran routine that will override the functions above for massive speedup.
    '''
    get_index_of_csr_data = amfe.f90_assembly.get_index_of_csr_data
    fill_csr_matrix = amfe.f90_assembly.fill_csr_matrix
    pass


class Assembly():
    '''
    Class for the more fancy assembly of meshes with non-heterogeneous elements.
    '''
    def __init__(self, mesh, element_class_dict): # element_class_dict enthaelt die Elementtypen, welche im Modul 'mechanical_system' definiert werden
        '''
        Parameters
        ----
        mesh : instance of the Mesh-class

        element_class_dict : dict
            dict where the official keyword and the the Element-objects are linked

        Returns
        --------
        None

        Examples
        ---------
        TODO

        '''
        self.mesh = mesh
        self.element_class_dict = element_class_dict
        self.save_stresses = False
        pass

    def preallocate_csr(self):
        '''
        Precompute the values and allocate the matrices for efficient assembly.

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
        global_element_indices : np.ndarray
            Array containing the global indices for the local variables of an element.
            The entry [i,j] gives the index in the global vector of element i with dof j
        node_coords : np.ndarray
            vector of all nodal coordinates. Dimension is (ndofs_total, )

        Notes
        -----
        This preallocation routine can take some while for small matrices.

        '''
        # computation of all necessary variables:
        dofs_per_node = self.mesh.node_dof
        self.node_coords = self.mesh.nodes.reshape(-1)
        elements = self.mesh.elements
        nodes_per_element = elements.shape[-1]
        dofs_per_element = nodes_per_element*dofs_per_node
        no_of_elements = len(elements)
        dofs_total = self.node_coords.shape[0]
        no_of_local_matrix_entries = dofs_per_element**2

        # compute the global element indices
        self.global_element_indices = np.zeros((no_of_elements, dofs_per_element), dtype=int)
        for i, element in enumerate(elements):
            self.global_element_indices[i,:] = np.array(
                [(np.arange(dofs_per_node) + dofs_per_node*i)  for i in element]).reshape(-1)


        # Auxiliary Help-Matrix H
        H = np.zeros((dofs_per_element, dofs_per_element))

        # preallocate the CSR-matrix
        row_global = np.zeros(no_of_elements*dofs_per_element**2, dtype=int)
        col_global = row_global.copy()
        vals_global = np.zeros(no_of_elements*dofs_per_element**2)

        for i, element_indices in enumerate(self.global_element_indices):
            H[:,:] = element_indices
            row_global[i*no_of_local_matrix_entries:(i+1)*no_of_local_matrix_entries] = \
                H.reshape(-1)
            col_global[i*no_of_local_matrix_entries:(i+1)*no_of_local_matrix_entries] = \
                H.T.reshape(-1)

        self.C_csr = sp.sparse.csr_matrix((vals_global, (row_global, col_global)),
                                          shape=(dofs_total, dofs_total))



    def assemble_matrix_and_vector(self, u, decorated_matrix_func):
        '''
        Assembles the matrix and the vector of the decorated matrix func.

        Parameters
        ----------

        u : ndarray
            displacement array
        decorated_matrix_func : function
            function which works like

            K, f = func(X_local, u_local)

        Returns
        -------

        K_csr : sp.sparse.csr_matrix
            Assembled matrix in csr-format (Compressed sparse row)
        f : ndarray
            array of the assembled vector
        '''
        K_csr = self.C_csr.copy()
        f_glob = np.zeros_like(self.node_coords)

        for i, indices in enumerate(self.global_element_indices):
            X = self.node_coords[indices]
            u_local = u[indices]
            K, f = decorated_matrix_func(X, u_local)
            f_glob[indices] += f
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
        # This is only working for one element type!
        element = self.element_class_dict[self.mesh.elements_type[0]]
        return self.assemble_matrix_and_vector(u, element.k_and_f_int)

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

        if u == None:
            u = np.zeros_like(self.node_coords)
        element = self.element_class_dict[self.mesh.elements_type[0]] # Zuweisen der Elementklasse
        M, _ = self.assemble_matrix_and_vector(u, element.m_and_vec_int) # element.m_and_vec_in
        return M

    def _assemble_matrix(self, u, decorated_matrix_func):
        '''
        This is an old function!!!


        Assembly routine for any matrices.

        Parameters
        ----------
        u : ndarray
            global displacement vector; if set to None, u will be assumed to be zero displacement
        decorated_matrix_func : function
            function with input variables (X, u_local, k, global_element_indices)
            the input variables are
            -----------------------
            X : ndarray
                local coordinates in the reference configuration
            u_local : ndarray
                local displacements
            k : int
                global index of the element (is needed in order to find the element type out of a global list)
            global_element_indices : ndarray
                global indices of the element (is needed for the force assembly)


        Returns
        -------
        Matrix : coo sparse array
            sparse assembled array in coo format
        element_props : list
            list of all saved element props that are pumped out of the decorated matrix func

        Note
        ----


        '''
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        self.element_props_global = []
        node_dof = self.mesh.node_dof
        if u is None:
            u = np.zeros(self.mesh.no_of_dofs)
        # loop over all elements
        for k, element in enumerate(self.mesh.elements):
            # coordinates of element in 1-D array
            X = np.array([self.mesh.nodes[i] for i in element]).reshape(-1)
            # corresponding global coordinates of element in 1-D array
            global_element_indices = np.array([(np.arange(node_dof) + node_dof*i)  for i in element]).reshape(-1)
            u_local = u[global_element_indices]
            # evaluation of element matrix (k_local, stresses)
            element_matrix, element_props = decorated_matrix_func(X, u_local, k, global_element_indices)
            self.row = np.zeros(element_matrix.shape)
            # build a matrix with constant columns and the rows representing the global_element_indices
            self.row[:,:] = global_element_indices
            self.row_global.append(self.row.reshape(-1))
            self.col_global.append((self.row.T).reshape(-1))
            # Attention! Here, the matrix is not copied! Make sure that the
            # element matrix is an object of its own
            self.vals_global.append(element_matrix.reshape(-1))
            self.element_props_global.append(element_props)
        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo, self.element_props_global


    def assemble_k(self, u=None):
        '''
        Assembles the stiffness matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        K : ndarray
            unconstrained assembled stiffness matrix in sparse matrix coo-format.
        '''

        def decorated_k_func(X, u_local, k, global_element_indices=None):
            '''
        Assembles the stiffness matrix of the given mesh and element.

            Parameters
            -----------
            X : ndarray
                local coordinates in the reference configuration
            u_local : ndarray
                local displacements
            k : int
                global index of the element
            global_element_indices : ndarray
                global dofs (corresponding to the dofs of X)                

            Returns
            --------
            k_local : TODO
                TODO
            stresses: TODO
                TODO
            '''
            # self.mesh.elements_type[k] gives type of element, e.g. 'Tri3'            
            element = self.element_class_dict[self.mesh.elements_type[k]]
            k_local = element.k_int(X, u_local)
            if self.save_stresses:
                stresses = element.S_voigt
            else:
                stresses = ()
            return k_local, stresses

        K, self.stress_list = self._assemble_matrix(u, decorated_k_func)
        return K


    def old_assemble_m(self, u=None):
        '''
        Assembles the mass matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        M : ndarray
            unconstrained assembled mass matrix in sparse matrix coo-format.

        Examples
        ---------
        TODO
        '''
        def decorated_m_func(X, u_local, k, global_element_indices=None):
            return self.element_class_dict[self.mesh.elements_type[k]].m_int(X, u_local), ()

        M, _ = self._assemble_matrix(u, decorated_m_func)
        return M


    def assemble_f(self, u):
        '''
        Assembles the force vector of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation

        Returns
        --------
        f : ndarray
            unconstrained force vector

        '''
        self.global_force = np.zeros(self.mesh.no_of_dofs)
        node_dof = self.mesh.node_dof
        for k, element in enumerate(self.mesh.elements):
            X = np.array([self.mesh.nodes[i] for i in element]).reshape(-1)
            # global coordinates of element
            global_element_indices = np.array([(np.arange(node_dof) + node_dof*i)  for i in element]).reshape(-1)
            self.global_force[global_element_indices] += \
                self.element_class_dict[self.mesh.elements_type[k]].f_int(X, u[global_element_indices])
        return self.global_force

    def assemble_k_and_f_old(self, u=None):
        '''
        Assembles the tangential stiffness matrix and the force matrix in one
        run as it is very often needed by an implicit integration scheme.

        Takes the advantage, that some element properties only have to be
        computed once.

        Parameters
        -----------
        u : ndarray, optional
            displacement of the unconstrained system in voigt notation

        Returns
        --------
        K : ndarray
            tangential stiffness matrix
        f : ndarray
            nonlinear force vector

        Examples
        ---------
        TODO
        '''
        def decorated_f_and_k_func(X, u_local, k, global_element_indices):
            element = self.element_class_dict[self.mesh.elements_type[k]]
            k_local, f_local = element.k_and_f_int(X, u_local)
            self.global_force[global_element_indices] += f_local
            if self.save_stresses:
                stresses = element.S_voigt
            else:
                stresses = ()
            return k_local, stresses

        self.stress_list = []
        self.global_force = np.zeros(self.mesh.no_of_dofs)
        K, _ = self._assemble_matrix(u, decorated_f_and_k_func)
        return K, self.global_force





class PrimitiveAssembly():
    '''
    Assembly class working directly on the tables of node coordinates and element nodes

    Came historically before more advanced assembly routines and have the status of being for test cases
    '''

    # Hier muessen wir uns mal genau ueberlegen, was alles dem assembly uebergeben werden soll
    # ob das ganze Mesh, oder nur ein paar Attribute
    def __init__(self, nodes=None, elements=None, matrix_function=None, node_dof=2, vector_function=None):
        '''
        Verlangt ein dreispaltiges Koordinatenarray, indem die Koordinaten in x, y, und z-Koordinaten angegeben sind
        Anzahl der Freiheitsgrade für einen Knotenfreiheitsgrad: node_dof gibt an, welche Koordinaten verwendet werden sollen;
        Wenn mehr Koordinaten pro Knoten nötig sind (z.B. finite Rotationen), werden Nullen hinzugefügt
        '''
        self.nodes = nodes
        self.elements = elements
        self.matrix_function = matrix_function
        self.vector_function = vector_function
        self.node_dof = node_dof

        self.row_global = []
        self.col_global = []
        self.vals_global = []

        self.no_of_nodes = len(self.nodes)
        self.no_of_elements = len(self.elements)
        self.no_of_dofs = self.no_of_nodes*self.node_dof
        self.no_of_element_nodes = len(self.elements[0])

        self.ndof_global = self.no_of_dofs
        pass


    def assemble_matrix(self, u=None):
        '''
        assembliert die matrix_function für die Ursprungskonfiguration X und die Verschiebung u.
        '''
        # deletion of former variables
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        # number of dofs per element (6 for triangle since no_of_element_nodes = 3 and node_dof = 2)
        ndof_local = self.no_of_element_nodes*self.node_dof
        # preset for u_local; necessary, when u=None
        u_local = np.zeros(ndof_local)

        for element in self.elements:
            # Koordinaten des elements
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            # element_indices have to be corrected in order respect the dimensions
            element_indices = np.array([[self.node_dof*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            if u is not None:
                u_local = u[element_indices]
            element_matrix = self.matrix_function(X, u_local)
            row = np.zeros((ndof_local, ndof_local))
            row[:,:] = element_indices
            self.row_global.append(row.reshape(-1))
            self.col_global.append((row.T).reshape(-1))
            self.vals_global.append(element_matrix.reshape(-1))

        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo

    def assemble_vector(self, u):
        '''
        Assembliert die Force-Function für die Usprungskonfiguration X und die Verschiebung u
        '''
        global_force = np.zeros(self.ndof_global)
        for element in self.elements:
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            element_indices = np.array([[2*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            global_force[element_indices] += self.vector_function(X, u[element_indices])
        return global_force







class MultiprocessAssembly():
    '''
    Klasse um schnell im Multiprozess zu assemblieren; Verteilt die Assemblierung auf alle Assemblierungsklassen und summiert die anschließend alles auf
    - funktioniert nicht so schnell, wie ich es erwartet hätte; genauere Analysen bisher noch nicht vorhanden, da profile-Tool nich zuverlässig für multiprocessing-Probleme zu funktionieren scheint.
    - ACHTUNG: Diese Klasse ist derzeit nicht in aktiver Nutzung. Möglicherweise macht es Sinn, diese Klasse zu überarbeiten, da sich die gesamte Programmstruktur gerade noch ändert.
    '''
    def __init__(self, assembly_class, list_of_matrix_functions, nodes_array, element_array):
        '''
        ???
        '''
        self.no_of_processes = len(list_of_matrix_functions)
        self.nodes_array = nodes_array
        self.element_array = element_array
        self.list_of_matrix_functions = list_of_matrix_functions
        domain_size = self.nodes_array.shape[0]//self.no_of_processes
        element_domain_list = []
        for i in range(self.no_of_processes - 1):
            element_domain_list.append(self.element_array[i*domain_size:(i+1)*domain_size,:])
        element_domain_list.append(self.element_array[(i+1)*domain_size:,:]) # assemble last domain to the end in order to consider flooring above
        self.assembly_class_list = [assembly_class(self.nodes_array, element_domain_list[i], matrix_function) for i, matrix_function in enumerate(list_of_matrix_functions)]
        pass

    def assemble(self):
        '''
        assembles the mesh with a multiprocessing routine
        '''
        pool = mp.Pool(processes=self.no_of_processes)
        results = [pool.apply_async(assembly_class.assemble) for assembly_class in self.assembly_class_list]
        matrix_coo_list = [j.get() for j in results]
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        data_global = np.array([], dtype=float)
        for matrix_coo in matrix_coo_list:
            row_global = np.append(row_global, matrix_coo.row)
            col_global = np.append(col_global, matrix_coo.col)
            data_global = np.append(data_global, matrix_coo.data)
        matrix_coo = sp.sparse.coo_matrix((data_global, (row_global, col_global)), dtype=float)
        return matrix_coo



