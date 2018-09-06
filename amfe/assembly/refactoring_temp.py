# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Basic assembly module for the finite element code. Assumes to have all elements
in the inertial frame.

Provides an Assembly class which knows the mesh. It can assemble the vector of
nonlinear forces, the mass matrix and the tangential stiffness matrix. Some
parts of the code -- mostly the indexing of the sparse matrices -- are
substituted by fortran routines, as they allow for a huge speedup.


"""

__all__ = ['Assembly',
           'AssemblyConstraint',
           ]

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
        row index which is asked to get the CSR-index for
    j : int
        column index which is asked to get the CSR-index for
    indptr : ndarray
        index-ptr-Array of the CSR-Matrix.
    indices : ndarray
        indices array of CSR-matrix (represents the nonzero column indices)

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

    # indices for row i are stored in indices[indptr[k]:indptr[k+1]]
    # Thus the indptr marks the start and end of the part of the indices
    # and val vector where all entries of a row are stored
    # --
    # Set k to the start of data of row k:
    k = indptr[i]
    # Search for appearance of j in the nonzero column indices which are
    # stored in indices[k] till indices[k+indptr[i+1]]
    while j != indices[k]:
        # while column j not found search for j in next entry
        k += 1
        # Check if next search would be in next (wrong) row
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
        'small' square array whose values will be distributed into the
        CSR-Matrix, Shape is (n,n)
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
    C_csr_hyper : scipy.sparse.csr.csr_matrix
        Matrix containing the sparsity pattern of the ECSW-hyperreduced problem
    C_csr_deim : scipy.sparse.csr.csr_matrix
        Matrix containing the sparsity pattern of the DEIM-hyperreduced problem
    element_indices : list
        Ragged list containing the global indices for the local variables
        of an element. The entry [i,j] gives the index in the global vector
        of element i with dof j
    mesh : amfe.Mesh
        Mesh-Class the Assembly is associated with
    neumann_indices : list
        Ragged list equivalently to element_indices for the neumann
        boundary skin elements.
    nodes_voigt : np.ndarray
        vector of all nodal coordinates in voigt-notation.
        Dimension is (ndofs_total, )
    elements_on_node : np.ndarray
        array containing the number of adjacent elements per node. Is necessary
        for stress recovery.
    observers : list
        list with objects of class Observer, observers that observe changes of the
        assembly object

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
        self.C_csr_hyper = None
        self._nodes_voigt = sp.array([])
        self.elements_on_node = None
        self.C_deim = None
        self._observers = list()

    def add_observer(self, observer, verbose=True):
        self._observers.append(observer)
        if verbose:
            print('Added observer to assembly')

    def remove_observer(self, observer, verbose=True):
        self._observers.remove(observer)
        if verbose:
            print('Removed observer from assembly')

    def notify(self):
        for observer in self._observers:
            observer.update()

    @property
    def nodes_voigt(self):
        return self._nodes_voigt

    @nodes_voigt.setter
    def nodes_voigt(self, nodes_voigt):
        self._nodes_voigt = nodes_voigt
        self.notify()

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


    def preallocate_hyper_csr(self, idxs):
        '''
        Preallocate a sparse matrix for a reduced mesh.

        Parameters
        ----------
        idxs : ndarray, shape (n_ele_hyper, )
            indices of the elements in the active element set.

        Returns
        -------
        None

        '''
        K_csr = self.C_csr.copy()

        # Loop over all elements
        for i, idx in enumerate(idxs):
            indices = self.element_indices[idx]
            K_ele = np.ones((len(indices), len(indices)), )
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data,
                            K_ele, indices)
        K_csr.eliminate_zeros()
        self.C_csr_hyper = K_csr * 0

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

        # Allocate K and f
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
        u : ndarray or None, optional
            nodal displacement of the nodes in Voigt-notation. Default: None.
        t : float, optional
            time. Default: 0.

        Returns
        --------
        M : sparse.csr_matrix
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
            K, f, S, E = self.mesh.ele_obj[i].k_f_S_E_int(X_local, u_local, t)

            # Assembly of force, stiffness, strain and stress
            f_glob[indices] += f
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)
            E_global[node_indices, :] += E
            S_global[node_indices, :] += S

        # Correct strains such, that average is taken at the elements
        E_global = (E_global.T/self.elements_on_node).T
        S_global = (S_global.T/self.elements_on_node).T

        return K_csr, f_glob, E_global, S_global

    def assemble_g_and_b(self, V, S, verbose=False):
        '''
        Assembles the element contributin matrix G for the given basis V and
        the snapshots S.

        This function is needed for cubature bases Hyper reduction methods
        like the ECSW.

        Parameters
        ----------
        V : ndarray, shape (N_unconstr, n)
            Reduction basis
        S : ndarray, shape (N_unconstr, m)
            Snapshots gathered as column vectors.
        verbose : print dots for the

        Returns
        -------
        G : ndarray, shape (n*m, no_of_elements)
            Contribution matrix of internal forces. The columns form the
            internal force contributions on the basis V for the m snapshots
            gathered in S.
        b : ndarray, shape (n*m, )
            summed force contribution

        Note
        ----
        This assembly works on the unconstrained variables, so both, V and S
        have to contain the unassembled nodal displacements.

        '''
        # Check the raw dimension
        assert(V.shape[0] == S.shape[0])

        # n : dimension of reduces basis
        # m : number of snapshots

        N_unconstr, n = V.shape
        __, m = S.shape

        if verbose:
            print('Start building large selection matrix G.',
                  'In total {0:d} elements are treated:'.format(
                      len(self.element_indices)))
        G = np.zeros((n*m, len(self.element_indices)))

        # loop over all elements
        for i, indices in enumerate(self.element_indices):
            # node_indices = self.mesh.connectivity[i]
            X_local = self.nodes_voigt[indices]
            V_ele = V[indices, :]
            if verbose:
                print('.', sep='', end='')
            # loop over all snapshots
            for j, u in enumerate(S.T):
                u_local = u[indices]
                f = self.mesh.ele_obj[i].f_int(X_local, u_local)
                G[j*n:(j+1)*n,i] = V_ele.T @ f

        b = np.sum(G, axis=1)
        return G, b

    def assemble_k_and_f_red(self, V, u, t):
        '''
        Assembly routine for reduces systems. Note, that V has to be
        unconstrained and u is the reduced displacement field.

        Parameters
        ----------
        V : ndarray, shape: (N_unconstr, n_red)
            unconstrained reduction basis
        u : ndarray, shape: (n_red,)
            unconstrained displacement field
        t : ndarray
            current time

        Returns
        -------
        K : ndarray, shape: (n_red, n_red)
            reduced stiffness matrix
        f_int : ndarray, shape (n_red,)
            reduced internal force vector

        '''
        n_red = V.shape[1]
        K_red = np.zeros((n_red, n_red))
        f_red = np.zeros(n_red)

        if u is None:
            u_full = np.zeros(V.shape[0])
        else:
            u_full = V @ u

        # Loop over all elements
        for i, indices in enumerate(self.element_indices):
            X_local = self.nodes_voigt[indices]
            u_local = u_full[indices]
            K_ele, f_ele = self.mesh.ele_obj[i].k_and_f_int(X_local, u_local, t)

            # Group the element contribution into the reduced K and f
            V_ele = V[indices,:]
            f_red += V_ele.T @ f_ele
            K_red += V_ele.T @ K_ele @ V_ele

        return K_red, f_red


    def assemble_k_and_f_hyper(self, V, idxs, xi, u, t):
        '''
        Assembly routine for a hyper reduced ECSW system.

        Parameters
        ----------
        V : ndarray, shape: (N_unconstr, n_red)
            unconstrained reduction basis
        idxs : ndarray, shape (n_ele_hyper, )
            indices of the elements in the active element set.
        xi : ndarray, shape (n_ele_hyper, )
            array of weights for the hyper reduced system
        u : ndarray, shape: (n_red,)
            unconstrained displacement field
        t : ndarray
            current time

        Returns
        -------
        K : ndarray, shape: (n_red, n_red)
            reduced stiffness matrix
        f_int : ndarray, shape (n_red,)
            reduced internal force vector

        See also
        --------
        assemble_k_and_f_hyper_no_inplace

        '''
        n_red = V.shape[1]
        K_red = np.zeros((n_red, n_red))
        f_red = np.zeros(n_red)

        if u is None:
            u_full = np.zeros(V.shape[0])
        else:
            u_full = V @ u

        # Loop over all elements
        for i, idx in enumerate(idxs):
            indices = self.element_indices[idx]
            X_loc = self.nodes_voigt[indices]
            u_loc = u_full[indices]
            K_ele, f_ele = self.mesh.ele_obj[idx].k_and_f_int(X_loc, u_loc, t)

            # this seems to be very slow for large bases
            V_ele = V[indices,:]
            f_red += V_ele.T @ f_ele * xi[i]
            K_red += V_ele.T @ K_ele @ V_ele * xi[i]

        return K_red, f_red


    def assemble_k_and_f_hyper_no_inplace(self, V, idxs, xi, u, t):
        '''
        Assembly routine for a hyper reduced ECSW system.

        The difference to the function `assemble_k_and_f_hyper` is, that here
        the sparse matrix is assembled first and then the reduction is
        performed.

        Parameters
        ----------
        V : ndarray, shape: (N_unconstr, n_red)
            unconstrained reduction basis
        idxs : ndarray, shape (n_ele_hyper, )
            indices of the elements in the active element set.
        xi : ndarray, shape (n_ele_hyper, )
            array of weights for the hyper reduced system
        u : ndarray, shape: (n_red,)
            unconstrained displacement field
        t : ndarray
            current time

        Returns
        -------
        K : ndarray, shape: (n_red, n_red)
            reduced stiffness matrix
        f_int : ndarray, shape (n_red,)
            reduced internal force vector

        See also
        --------
        assemble_k_and_f_hyper

        '''
        if self.C_csr_hyper is None:
            self.preallocate_hyper_csr(idxs)
        K_csr = self.C_csr_hyper.copy()
        f_glob = np.zeros(self.mesh.no_of_dofs)

        if u is None:
            u_full = np.zeros(V.shape[0])
        else:
            u_full = V @ u

        # Loop over all elements
        for i, idx in enumerate(idxs):
            indices = self.element_indices[idx]
            X_loc = self.nodes_voigt[indices]
            u_loc = u_full[indices]
            K_ele, f_ele = self.mesh.ele_obj[idx].k_and_f_int(X_loc, u_loc, t)

            f_glob[indices] += f_ele * xi[i]
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data,
                            K_ele*xi[i], indices)

        K_red = V.T @ K_csr @ V
        f_red = V.T @ f_glob

        return K_red, f_red

    def f_nl_unassembled(self, u=None, t=0):
        '''
        Computes the unassemebled nonlinear force for DEIM.

        Parameters
        ----------
        u : ndarray
            Initial solution or solution from previous iteration
        t : float
            Time

        Returns
        -------
        f : ndarray
            Unassembled nonlinear internal force

        '''
#         #this works with inhomogeneous meshes
        dofs_unassembled = len(np.concatenate(self.element_indices))
        f = np.zeros(dofs_unassembled)

        if u is None:
            u = np.zeros(self.mesh.no_of_dofs)

        dof_idx = 0 # necessary to work with inhomogeneous meshes

        # Loop over all elements
        for i, indices in enumerate(self.element_indices):

            X_local = self.nodes_voigt[indices]
            u_local = u[indices]
            K_ele, f_ele = self.mesh.ele_obj[i].k_and_f_int(X_local, u_local, t)
            K0_ele, __    = self.mesh.ele_obj[i].k_and_f_int(X_local, u_local*0, t)
            f_ele_nl = f_ele - K0_ele @ u_local

            # assemble the force into the global force vector
            dofs_ele = indices.shape[0]
            f[dof_idx : dof_idx+dofs_ele] = f_ele_nl
            dof_idx += dofs_ele

        return f

    def assemble_k_and_f_DEIM(self, E_tilde, proj_list, V, u_red=None, t=0,
                                symmetric=False):
        '''
        Assemble the DEIM matrix and vector. This function is suited for large
        systems, as only hte proj list is used.

        '''
        ndim_red, ele_dofs = proj_list[0].shape
#        ndim_red, no_of_force_modes = oblique_proj.shape
#        ele_dofs = len(self.mesh_class.element_indices[0])

        f = np.zeros(ndim_red)
        K = np.zeros((ndim_red, ndim_red))

        if u_red is None and not symmetric:
            u_full = np.zeros(V.shape[0])
        elif not symmetric:
            u_full = V @ u_red

        if u_red is None and symmetric:
            u_red = np.zeros(V.shape[1])

#        if u_red is None:
#            u_full = np.zeros(V.shape[0])
#        elif symmetric:
#            u_unass = P @ oblique_proj.T @ u_red
#        else:
#            u_full = V @ u_red

        for i, ele in enumerate(E_tilde):
            indices = self.element_indices[ele]
            X_local = self.nodes_voigt[indices]
            proj = proj_list[i]
            if symmetric:
                u_local = proj.T @ u_red
                K_ele, f_ele = self.mesh.ele_obj[i].k_and_f_int(X_local,
                                                                u_local, t)
                f += proj @ f_ele
                K += proj @ K_ele @ proj.T
            else:
                u_local = u_full[indices]
                K_ele, f_ele = self.mesh.ele_obj[i].k_and_f_int(X_local,
                                                                u_local, t)
                f += proj @ f_ele
                K += proj @ K_ele @ V[indices, :]
        return K, f


    def compute_c_deim(self):
        '''
        Assembles the matrix C_deim which assembles the unassembled system
        C_matrix: f_a = C_deim @ f_u

        Parameters:
        ----------
        None

        Return:
        -------
        C_matrix : sp.sparse.csr_matrix, ndim (n_assbld, n_unassbld)
            Matrix operator which assembles unassembled variables

        '''
        row = np.concatenate(self.element_indices)
        col = np.arange(len(row))
        data = np.ones_like(row, dtype=bool)

        no_of_dofs = self.mesh.no_of_dofs
        dofs_unassembled = len(row)
        self.C_deim = sp.sparse.csr_matrix((data, (row, col)),
                                           shape=(no_of_dofs, dofs_unassembled))
        return self.C_deim


##### Constrained Mechanical System ###########################################

class AssemblyConstraint(Assembly):
    '''
    Class for the assembly of constraints.

    Attributes
    ----------
    B_csr : scipy.sparse.csr.csr_matrix
        Matrix containing the sparsity pattern of the constraints

    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        self.B_csr = sp.sparse.csr_matrix([[]])

    def preallocate_B_csr(self):
        '''
        Precompute the values and allocate the matrices for efficient assembly.

        Parameters
        ----------
        None

        Returns
        -------
        None


        Notes
        -----
        This preallocation routine can take some time.

        '''
        print('Preallocating the Jacobian matrix of constraints')
        t1 = time.clock()
        # computation of all necessary variables:

        # create B_csr with the number of unconstrained dofs and multiply
        # it like a vector to get the constrained B: B.dot(B_constraints.T)
        #        constraints_list = self.mesh.constraints_list
        constraints_dofs_related = self.mesh.constraints_dofs_related
        no_of_dofs = self.mesh.no_of_dofs
        ndof_const = len(constraints_dofs_related)

        # find out the number of nonzero entries in B
        nonzero_in_B = 0
        for dofs in constraints_dofs_related:
            nonzero_in_B += len(dofs)
        # preallocate the CSR-matrix
        row_global = np.zeros(nonzero_in_B, dtype=int)
        col_global = row_global.copy()

        vals_global = np.zeros_like(col_global, dtype=bool)

        # build csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)])
        # where data, row_ind and col_ind satisfy the relationship
        # a[row_ind[k], col_ind[k]] = data[k].
        i = 0
        for j, constraint_dofs in enumerate(constraints_dofs_related):
            for dof in constraint_dofs:
                row_global[i] = j
                col_global[i] = dof
                i += 1

        self.B_csr = sp.sparse.csr_matrix((vals_global, (row_global, col_global)),
                                          shape=(ndof_const, no_of_dofs), dtype=float)
        t2 = time.clock()
        print('Done preallocating the Jacobian matrix of constraints with',
              ndof_const, 'constraints', 'and', no_of_dofs, 'dofs.')
        print('Time taken for preallocation:', t2 - t1, 'seconds.')

    def assemble_B(self, u, t):
        '''
        Assembles the jacobian of the constraints vector for the given mesh
        and element and constraint.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        B_csr : ndarray
            unconstrained jacobian matrix of the constraints in sparse matrix
            csr-format.

        Examples
        ---------
        TODO
        '''
        if u is None:
            u = np.zeros_like(self.nodes_voigt)

        B_csr = self.B_csr.copy()

        constraints_list = self.mesh.constraints_list
        constraints_dofs_related = self.mesh.constraints_dofs_related

        for i, constraint in enumerate(constraints_list):
            X_local = self.nodes_voigt[constraints_dofs_related[i]]
            u_local = u[constraints_dofs_related[i]]

            # This small b vector refers ONLY to the nonzero entries in B for a
            # specific constraint. E.g. for Dirichlet BC it is always [1].
            # For this to work it is important that constraint_dofs_related
            # already has the correct DOFs that will be constrained.
            b = constraint.vector_b(X_local, u_local, t)

            fill_B_csr_matrix(B_csr.indptr, B_csr.indices, B_csr.data, b, i,
                              constraints_dofs_related[i])

        return B_csr

    def assemble_C(self, u, t):
        '''
        Assembles the constraints vector for the given mesh, element and
        constraints.

        Parameters
        -----------
        u : ndarray
            nodal displacement of the nodes in Voigt-notation
        du: ndarray
            nodal velocities of the nodes in Voigt-notation
        t : float
            time

        Returns
        --------
        C : ndarray
            array of constraint equations

        Examples
        ---------
        TODO
        '''
        if u is None:
            u = np.zeros_like(self.nodes_voigt)

        constraints_list = self.mesh.constraints_list
        constraints_dofs_related = self.mesh.constraints_dofs_related
        ndof_const = len(constraints_list)
        C = np.zeros((ndof_const))

        for i, constraint in enumerate(constraints_list):
            X_local = self.nodes_voigt[constraints_dofs_related[i]]
            u_local = u[constraints_dofs_related[i]]
            # build C for each constraint, displacement and X.
            C[i] += constraint.c_equation(X_local, u_local, t=t)
        return C


def fill_B_csr_matrix(indptr, indices, vals, b, i, b_indices):
    '''
    Fill the values of b into the vals-array of a sparse CSR Matrix given the
    b_indices array.

    Parameters
    ----------

    indptr : ndarray
        indptr-array of a preallocated CSR-Matrix
    indices : ndarray
        indices-array of a preallocated CSR-Matrix
    vals : ndarray
        vals-array of a preallocated CSR-Marix
    b : ndarray
        'small' line array which values will be distributed into the CSR-Matrix
    i : int
        represents the line of B_csr that will be filled
    b_indices : ndarray
        indices for the colums which b represents inside B_csr

    Returns
    -------

    None

    '''

    ndof_l = b.shape[0]
    for j in range(ndof_l):
        l = get_index_of_csr_data(i, b_indices[j], indptr, indices)
        vals[l] += b[j]
    return