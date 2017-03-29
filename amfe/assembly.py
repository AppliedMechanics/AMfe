"""
Basic assembly module for the finite element code. Assumes to have all elements
in the inertial frame.

Provides an Assembly class which knows the mesh. It can assemble the vector of nonlinear forces, the mass matrix and the tangential stiffness matrix. Some parts of the code -- mostly the indexing of the sparse matrices -- are substituted by fortran routines, as they allow for a huge speedup.


"""

__all__ = ['Assembly']

import time

import numpy as np
import scipy as sp

from scipy import sparse
from scipy import linalg


# Trying to import the fortran routines
use_fortran = False
try:
    import amfe.f90_assembly
    use_fortran = True
except ImportError:
    print('Python was not able to load the fast fortran assembly routines.')
#use_fortran = False


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
    return


if use_fortran:
    ###########################################################################
    # Fortran routine that will override the functions above for massive speedup.
    ###########################################################################
    get_index_of_csr_data = amfe.f90_assembly.get_index_of_csr_data
    fill_csr_matrix = amfe.f90_assembly.fill_csr_matrix



class Assembly():
    '''
    Class for the more fancy assembly of meshes with non-heterogeneous
    elements.

    Attributes
    ----------
    C_csr : scipy.sparse.csr.csr_matrix
        Matrix containing the sparsity pattern of the problem
    csr_assembly_indices : np.ndarray
        Array containing the indices for the csr matrix assembly routine.
        The entry [i,j,k] contains the index of the csr-value-matrix
        for the i-th element and the j-th row and the k-th column
        of the local stiffness matrix of the i-th element.
        The dimension is (n_elements, ndof_element, ndof_element).
    element_indices : list
        Ragged list containing the global indices for the local variables
        of an element. The entry [i,j] gives the index in the global vector
        of element i with dof j
    neumann_indices : list
        Ragged list equivalently to element_indices for the neumann
        boundary skin elements.
    nodes_voigt : np.ndarray
        vector of all nodal coordinates in voigt-notation.
        Dimension is (ndofs_total, )
    elements_on_node : np.ndarray
        array containing the number of adjacent elements per node. Is necessary
        for stress recovery.

    '''
    def __init__(self, mesh):
        '''
        Parameters
        ----------
        mesh : instance of the Mesh-class

        Returns
        -------
        None

        Examples
        --------
        TODO

        '''
        self.mesh = mesh
        self.element_indices = []
        self.neumann_indices = []
        self.C_csr = sp.sparse.csr_matrix([[]])
        self.nodes_voigt = sp.array([])
        self.elements_on_node = None

    def preallocate_csr(self):
        '''
        Compute the sparsity pattern of the assembled matrices and store an
        empty matrix in self.C_csr.

        The matrix self.C_csr serves as a 'blueprint' matrix which is filled in
        the assembly process.

        Parameters
        ----------
        None

        Returns
        -------
        None


        Notes
        -----
        This preallocation routine can take some while for large matrices. Furthermore it is not implemented memory-efficient, so for large systems and low RAM this might be an issue...

        '''
        print('Preallocating the stiffness matrix')
        t1 = time.clock()
        # computation of all necessary variables:
        no_of_elements = self.mesh.no_of_elements
        no_of_dofs = self.mesh.no_of_dofs
        self.nodes_voigt = self.mesh.nodes.reshape(-1)

        self.compute_element_indices()
        max_dofs_per_element = np.max([len(i) for i in self.element_indices])

        # Auxiliary Help-Matrix H which is the blueprint of the local
        # element stiffness matrix
        H = np.zeros((max_dofs_per_element, max_dofs_per_element))

        # preallocate the CSR-matrix
        
        # preallocate row_global with maximal possible size for prealloc. C_csr
        row_global = np.zeros(no_of_elements*max_dofs_per_element**2, dtype=int)
        # preallocate col_global with maximal possible size for prealloc. C_csr
        col_global = row_global.copy()
        # set 'dummy' values
        vals_global = np.zeros_like(col_global, dtype=bool)

        # calculate row_global and col_global
        for i, indices_of_one_element in enumerate(self.element_indices):
            l = len(indices_of_one_element)
            # insert global-dof-ids in l rows (l rows have equal entries)
            H[:l,:l] = indices_of_one_element
            # calculate row_global and col_global such that every possible
            # combination of indices_of_one_element can be returned by
            # (row_global[k], col_global[k]) for all k
            row_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.reshape(-1)
            col_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.T.reshape(-1)

        # fill C_csr matrix with dummy entries in those places where matrix
        # will be filled in assembly
        self.C_csr = sp.sparse.csr_matrix((vals_global, (row_global, col_global)),
                                          shape=(no_of_dofs, no_of_dofs), dtype=float)
        t2 = time.clock()
        print('Done preallocating stiffness matrix with', no_of_elements,
              'elements and', no_of_dofs, 'dofs.')
        print('Time taken for preallocation: {0:2.2f} seconds.'.format(t2 - t1))

    def compute_element_indices(self):
        '''
        Compute the element indices which are the global dofs of every element.

        The element_indices are a list, where every element of the list denotes
        the dofs of the element in the correct order.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        connectivity = self.mesh.connectivity
        nm_connectivity = self.mesh.neumann_connectivity
        no_of_dofs_per_node = self.mesh.no_of_dofs_per_node

        # Explanation of following expression:
        # for each element in connectivity
        # and for each node-id of each element
        # take [0,1] (2D-problem) or [0,1,2] (3D-problem)
        # and add 2*node_id (2D-problem) or 3*node_id (3D-problem)
        # and reshape the result...
        # Result (self.element_indices:)
        # the rows are the elements, the columns are the local element dofs
        # the values are the global dofs
        self.element_indices = \
        [np.array([(np.arange(no_of_dofs_per_node) + no_of_dofs_per_node*node_id)
                   for node_id in elements], dtype=int).reshape(-1)
         for elements in connectivity]

        self.neumann_indices = \
        [np.array([(np.arange(no_of_dofs_per_node) + no_of_dofs_per_node*i)
                   for i in nodes], dtype=int).reshape(-1)
         for nodes in nm_connectivity]

        # compute nodes_frequency for stress recovery
        nodes_vec = np.concatenate(self.mesh.connectivity)
        self.elements_on_node = np.bincount(nodes_vec)


    def assemble_k_and_f(self, u, t):
        '''
        Assembles the stiffness matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            unconstrained assembled force vector
        '''
        if u is None:
            u = np.zeros_like(self.nodes_voigt)

        K_csr = self.C_csr.copy()
        f_glob = np.zeros(self.mesh.no_of_dofs)

        # Loop over all elements
        # (i - element index, indices - DOF indices of the element)
        for i, indices in enumerate(self.element_indices):
            # X - The undeformed positions of the i-th element
            X_local = self.nodes_voigt[indices]
            # the displacements of the i-th element
            u_local = u[indices]
            # K computation of the element tangential stiffness matrix and
            # nonlinear force
            K, f = self.mesh.ele_obj[i].k_and_f_int(X_local, u_local, t)
            # adding the local force to the global one
            f_glob[indices] += f

            # this is equal to
            # K_csr[indices, indices] += K
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)

        return K_csr, f_glob


    def assemble_m(self, u=None, t=0):
        '''
        Assembles the mass matrix of the given mesh and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        M : ndarray
            unconstrained assembled mass matrix in sparse matrix csr-format.

        Examples
        ---------
        TODO
        '''
        if u is None:
            u = np.zeros_like(self.nodes_voigt)

        M_csr = self.C_csr.copy()

        for i, indices in enumerate(self.element_indices):
            X_local = self.nodes_voigt[indices]
            u_local = u[indices]
            M = self.mesh.ele_obj[i].m_int(X_local, u_local, t)
            fill_csr_matrix(M_csr.indptr, M_csr.indices, M_csr.data, M, indices)

        return M_csr


    def assemble_k_and_f_neumann(self, u=None, t=0):
        '''
        Assembles the stiffness matrix and the force of the Neumann skin
        elements.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            unconstrained assembled force vector
        '''
        if u is None:
            u = np.zeros_like(self.nodes_voigt)

        K_csr = self.C_csr.copy()
        f_glob = np.zeros(self.mesh.no_of_dofs)

        for i, indices in enumerate(self.neumann_indices):
            X_local = self.nodes_voigt[indices]
            u_local = u[indices]
            K, f = self.mesh.neumann_obj[i].k_and_f_int(X_local, u_local, t)
            f_glob[indices] += f
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)

        return K_csr, f_glob

    def assemble_k_f_S_E(self, u, t):
        '''
        Assemble the stiffness matrix with stress recovery of the given mesh
        and element.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        K : sparse.csr_matrix
            unconstrained assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            unconstrained assembled force vector
        S : ndarray
            unconstrained assembled stress tensor
        E : ndarray
            unconstrained assembled strain tensor

        '''

        K_csr = self.C_csr.copy()
        f_glob = np.zeros(self.mesh.no_of_dofs)
        no_of_nodes = len(self.mesh.nodes)
        E_global = np.zeros((no_of_nodes, 6))
        S_global = np.zeros((no_of_nodes, 6))

        for i, indices in enumerate(self.element_indices):
            node_indices = self.mesh.connectivity[i]
            X_local = self.nodes_voigt[indices]
            u_local = u[indices]
            K, f, E, S = self.mesh.ele_obj[i].k_f_S_E_int(X_local, u_local, t)

            # Assembly of force, stiffness, strain and stress
            f_glob[indices] += f
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)
            E_global[node_indices, :] += E
            S_global[node_indices, :] += S

        # Correct strains such, that average is taken at the elements
        E_global = (E_global.T/self.elements_on_node).T
        S_global = (S_global.T/self.elements_on_node).T

        return K_csr, f_glob, E_global, S_global
