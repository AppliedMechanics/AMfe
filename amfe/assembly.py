"""
Basic assembly module for the finite element code. Assumes to have all elements
in the inertial frame.
"""

__all__ = ['Assembly']

import time

import numpy as np
import scipy as sp

from scipy import sparse
from scipy import linalg



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
    Class for the more fancy assembly of meshes with non-heterogeneous elements.

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
        self.save_stresses = False
        self.element_indices = []
        self.neumann_indices = []
        self.C_csr = sp.sparse.csr_matrix([[]])
        self.nodes_voigt = sp.array([])

    def preallocate_csr(self):
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
        This preallocation routine can take some while.

        '''
        print('Preallocating the stiffness matrix')
        t1 = time.clock()
        # computation of all necessary variables:
        no_of_elements = self.mesh.no_of_elements
        no_of_dofs = self.mesh.no_of_dofs
        self.nodes_voigt = self.mesh.nodes.reshape(-1)

        self.compute_element_indices()
        max_dofs_per_element = np.max([len(i) for i in self.element_indices])


        # Auxiliary Help-Matrix H
        H = np.zeros((max_dofs_per_element, max_dofs_per_element))

        # preallocate the CSR-matrix
        row_global = np.zeros(no_of_elements*max_dofs_per_element**2, dtype=int)
        col_global = row_global.copy()
        vals_global = np.zeros_like(col_global, dtype=bool)

        for i, indices_of_one_element in enumerate(self.element_indices):
            l = len(indices_of_one_element)
            H[:l,:l] = indices_of_one_element
            row_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.reshape(-1)
            col_global[i*max_dofs_per_element**2:(i+1)*max_dofs_per_element**2] = \
                H.T.reshape(-1)

        self.C_csr = sp.sparse.csr_matrix((vals_global, (row_global, col_global)),
                                          shape=(no_of_dofs, no_of_dofs), dtype=float)
        t2 = time.clock()
        print('Done preallocating stiffness matrix with', no_of_elements, 'elements',
              'and', no_of_dofs, 'dofs.')
        print('Time taken for preallocation:', t2 - t1, 'seconds.')

    def compute_element_indices(self):
        '''
        Compute the element indices which are necessary for assembly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        ele_nodes = self.mesh.ele_nodes
        nm_nodes = self.mesh.neumann_nodes
        no_of_dofs_per_node = self.mesh.no_of_dofs_per_node

        self.element_indices = \
        [np.array([(np.arange(no_of_dofs_per_node) + no_of_dofs_per_node*i)
                   for i in nodes], dtype=int).reshape(-1)
         for nodes in ele_nodes]

        self.neumann_indices = \
        [np.array([(np.arange(no_of_dofs_per_node) + no_of_dofs_per_node*i)
                   for i in nodes], dtype=int).reshape(-1)
         for nodes in nm_nodes]



    def assemble_matrix_and_vector(self, u, decorated_matrix_func,
                                   element_indices, t):
        '''
        Assembles the matrix and the vector of the decorated matrix func.

        Parameters
        ----------

        u : ndarray
            global displacement array
        decorated_matrix_func : function
            function which works like

                >>> K_local, f_local = func(index, X_local, u_local)

        element_indices : list
            List containing the indices mappint the local dofs to the global dofs.
            element_indices[i][j] returns the global dof of element i's dof j.
        t : float
            Time
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
        for i, indices in enumerate(element_indices):
            # X - zu den DOF-Nummern zugehoerige Koordinaten (Positionen)
            X = self.nodes_voigt[indices]
            # Auslesen der localen Elementverschiebungen
            u_local = u[indices]
            # K wird die Elementmatrix und f wird der Elementlastvektor zugewiesen
            K, f = decorated_matrix_func(i, X, u_local, t)
            # Einsortieren des lokalen Elementlastvektors in den globalen Lastvektor
            f_glob[indices] += f
            # Einsortieren der lokalen Elementmatrix in die globale Matrix
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)

        return K_csr, f_glob

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

        # Schleife ueber alle Elemente
        # (i - Elementnummer, indices - DOF-Nummern des Elements)
        for i, indices in enumerate(self.element_indices):
            # X - zu den DOF-Nummern zugehoerige Koordinaten (Positionen)
            X_local = self.nodes_voigt[indices]
            # Auslesen der localen Elementverschiebungen
            u_local = u[indices]
            # K wird die Elementmatrix und f wird der Elementlastvektor zugewiesen
            K, f = self.mesh.ele_obj[i].k_and_f_int(X_local, u_local, t)
            # Einsortieren des lokalen Elementlastvektors in den globalen Lastvektor
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
            node_indices = self.mesh.ele_nodes[i]
            # X - zu den DOF-Nummern zugehoerige Koordinaten (Positionen)
            X_local = self.nodes_voigt[indices]
            # Auslesen der localen Elementverschiebungen
            u_local = u[indices]
            # K wird die Elementmatrix und f wird der Elementlastvektor zugewiesen
            K, f, E, S = self.mesh.ele_obj[i].k_f_S_E_int(X_local, u_local, t)
            # Einsortieren des lokalen Elementlastvektors in den globalen Lastvektor
            f_glob[indices] += f
            # Einsortieren der lokalen Elementmatrix in die globale Matrix
            fill_csr_matrix(K_csr.indptr, K_csr.indices, K_csr.data, K, indices)
            # Assemble the stresses and strains
            E_global[node_indices, :] += E
            S_global[node_indices, :] += S

        # Correct strains such, that average is taken at the elements
        nodes_vec = np.array(self.mesh.ele_nodes).ravel
        # calculate the frequency of every nodes in ele_nodes list
        nodes_frequency = np.bincount(nodes_vec)
        E_global = (E_global.T/nodes_frequency).T
        S_global = (S_global.T/nodes_frequency).T

        return K_csr, f_glob, E_global, S_global


###########################Hyper Reduction implementation######################




    def assemble_g_and_b(self, u, t, Vq, BV):
        '''
        Assembles the G matrix and b vector

        Parameters
        -----------
        u   : ndarray
              nodal displacement of the nodes in Voigt-notation
        t   : float
              time
        Vq  : ndarray
              The trained dofs for ECSW
        BV  : ndarray
              Constraining matrix B multiplied with constrained basis V

        Returns
        -------
        G   : ndarray
              Refer Farhats paper on ECSW
        b   : ndarray
              Refer Farhats paper on ECSW
        '''
        def k_and_f_func(i, X, u, t):
            '''
            Decorated function picking the element object from the mesh and
            returning k and f out of it.

            Parameters
            ----------
            i : int
                index of the element
            X : ndarray
                reference configuration of nodes
            u : ndarray
                displacement of nodes
            t : float
                time

            Returns
            -------
            K : ndarray
                Stiffness matrix.
            f : ndarray
                Force vector.

            '''

            return self.mesh.ele_obj[i].k_and_f_int(X, u, t)

        return self.assemble_g_matrix_and_b_vector(u, k_and_f_func,
                                               self.element_indices, t, Vq,BV)
#    @profile
    def assemble_g_matrix_and_b_vector(self, u, decorated_matrix_func,
                                   element_indices, t,Vq,BV):

        '''
        Assembles the matrix and the vector of the decorated matrix func.

        Parameters
        ----------

        u : ndarray
            global displacement array
        decorated_matrix_func : function
            function which works like

                >>> K_local, f_local = func(index, X_local, u_local)

        element_indices : list
            List containing the indices mappint the local dofs to the global dofs.
            element_indices[i][j] returns the global dof of element i's dof j.
        t : float
            Time
        Vq  : ndarray
              The trained dofs for ECSW
        BV  : ndarray
              Constraining matrix B multiplied with constrained basis V

        Returns
        -------
        G     : ndarray
                Refer Farhats paper on ECSW
        b_g   : ndarray
                Refer Farhats paper on ECSW
        '''


        # Set the dimensions of V, G, B into variables
        m = BV.shape[1] #no. of modes
        n_t = Vq.shape[1] # no. of training vectors
        n_e = self.mesh.no_of_elements #no. of elements

        # Initialize G and (b_g -> b related to G)

        G = np.zeros((m*n_t, n_e))
        b_g= np.zeros((m*n_t))
        g1sum = np.zeros(m)

        # Loop over j(alles elemente) and i(alles training vetorin)
        # i->  element number, indices- DOF-number j ->  training vec number
        for j in np.arange(n_t): #(row)
            g1sum = np.zeros(m)
            for i, indices in enumerate(element_indices):  #(column)

                # coordinates of the nodes corresponding to the dofs
                X = self.nodes_voigt[indices]

                #u_local = u[indices]
                #v_local = V[indices,:]

                u_proj  = Vq[:,j] # Kinematically admissible u
                u_local = u_proj[indices]
                #Somehow Sky Daddy ou get the element matrix and force,
                _, f = decorated_matrix_func(i, X, u_local, t)

                bv_local = BV[indices,:]
#                g1  = v_local.T @ f # \in R**n
                g1  =  bv_local.T @ f # \in R**{n-dirchdofs}
                g1sum = g1sum + g1
#                g2  = v_local.T @ K @ v_local # not sure if this is used.
                G[j*m:(j+1)*m,i] = g1 # \in R**{mn_t*n_e}

            b_g[j*m:(j+1)*m] = g1sum # \in R**m

        return G,b_g

    def assemble_hr_k_and_f(self, u, t, xi, E_tilde, BV):
        '''
        Assembles the Hyper-reduced stiffness matrix of the given mesh and element.

        Parameters
        -----------
        u       : ndarray
                  nodal displacement of the nodes in Voigt-notation
        t       : float
                  time
        G       : ndarray
                  Refer Farhats paper on ECSW
        b       : ndarray
                  Refer Farhats paper on ECSW
        xi      : ndarray
                  Weights corresponding to row number, i.e., element number
        E_tilde : list
                  List of all elements that have non zero weights
        BV      : ndarray
                  Constraining matrix B multiplied with constrained basis V

        Returns
        --------
        K : Dense ndarray
            Constrained Hyper-reduced assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            Constrained Hyper-reduced force vector
        '''
        # define the function that returns K, f for (i, X, u)
        # sort of a decorator approach!
        def k_and_f_func(i, X, u, t):
            '''
            Decorated function picking the element object from the mesh and
            returning k and f out of it.

            Parameters
            ----------
            i : int
                index of the element
            X : ndarray
                reference configuration of nodes
            u : ndarray
                displacement of nodes
            t : float
                time

            Returns
            -------
            K : ndarray
                Stiffness matrix.
            f : ndarray
                Force vector.

            '''
            return self.mesh.ele_obj[i].k_and_f_int(X, u, t)

        return self.assemble_hr_matrix_and_vector(u, k_and_f_func,
                                    self.element_indices, t, xi, E_tilde, BV)

#    @profile
    def assemble_hr_matrix_and_vector(self, u, decorated_matrix_func,\
                                element_indices, t, xi=None, E_tilde=None, BV=None):
        '''
        Assembles the Hyper-reduced matrix and the vector of the decorated matrix func.

        Parameters
        ----------

        u : ndarray
            global displacement array
        decorated_matrix_func : function
            function which works like

                >>> K_local, f_local = func(index, X_local, u_local)

        element_indices : list
            List containing the indices mappint the local dofs to the global dofs.
            element_indices[i][j] returns the global dof of element i's dof j.
        t       : float
                  Time
        xi      : ndarray
                  Weights corresponding to row number, i.e., element number
        E_tilde : list
                  List of all elements that have non zero weights
        BV      : ndarray
                  Constraining matrix B multiplied with constrained basis V

        Returns
        --------
        K : Dense ndarray
            Constrained Hyper-reduced assembled stiffness matrix in sparse matrix csr format.
        f : ndarray
            Constrained Hyper-reduced force vector
        '''

        n_t = BV.shape[1]
        K_red = np.zeros((n_t,n_t))
        f_red = np.zeros((n_t))

        #try 3 will not work as we have element_indices everywhere
#        for j, indices in enumerate(element_indices):
#            i = E_tilde[j] #element nummer!
#            indices = element_indices[j]   # Correspondin indices
         #try2 not faster than try 1
#        for j in np.arange(len(E_tilde)):
#            i = E_tilde[j]
#            indices = element_indices[i]
        for i in E_tilde:
            indices  = element_indices[i]
#
         #try 1
#        # i->  element number, indices- DOF-number
#        for i, indices in enumerate(element_indices):
#            if xi[i] != 0:

            # coordinates of the nodes corresponding to the dofs
            X = self.nodes_voigt[indices]

            u_local = u[indices]
            bv_local = BV[indices,:]

            #Somehow you get the element matrix and force,
            K, f = decorated_matrix_func(i, X, u_local, t)

            K_mod = xi[i] * bv_local.T @ K @ bv_local
            f_mod = xi[i] * bv_local.T @ f

            # Einsortieren des lokalen Elementlastvektors in den globalen Lastvektor
            f_red += f_mod
            # Einsortieren der lokalen Elementmatrix in die globale Matrix
            K_red += K_mod

        return K_red,f_red
