"""Module for handling the Dirichlet and Neumann boundary. """

__all__ = ['DirichletBoundary', 'NeumannBoundary']

import numpy as np
import scipy as sp


class DirichletBoundary():
    '''
    Class responsible for the Dirichlet Boundary conditions

    The boundary-information is stored in the master_slave_list, which forms
    the interface for all homogeneous Dirichlet boundary condtions.
    '''
    def __init__(self, no_of_unconstrained_dofs, master_slave_list=[]):
        '''
        Parameters
        ----------
        ndof_unconstrained_system : int
            Number of dofs of the unconstrained system.
        master_slave_list : list
            list containing the dirichlet-boundary triples (DBT)

            >>> [DBT_1, DBT_2, DBT_3, ]

        Returns
        -------
        None

        Notes
        -----
        each dirchilet_boundary_triple is itself a list containing

        >>> DBT = [master_dof=None, [list_of_slave_dofs], B_matrix=None]

        master_dof : int / None
            the dof onto which the slave dofs are projected. The master_dof
            will be overwritten at the end, i.e. if the master dof should
            participate at the end, it has to be a member in teh list of
            slave_dofs. If the master_dof is set to None, the slave_dofs will
            be fixed
        list_of_slave_dofs : list containing ints
            The list of the dofs which will be projected onto the master dof;
            the weights of the projection are stored in the B_matrix
        B_matrix : ndarras / None
            The weighting-matrix which gives enables to apply complicated
            boundary conditions showing up in symmetry-conditions or rotational
            dofs. The default-value for B_matrix is None, which weighs all
            members of the slave_dof_list equally with 1.

        Examples
        --------

        The dofs 0, 2 and 4 are fixed:

        >>> DBT = [None, [0, 2, 4], None]
        >>> my_boundary = DirichletBoundary([DBT, ])

        The dofs 0, 1, 2, 3, 4, 5, 6 are fixed and the dofs 100, 101, 102, 103
        have all the same displacements:

        >>> DBT_fix = [None, np.arange(7), None]
        >>> DBT_disp = [100, [100, 101, 102, 103], None]
        >>> my_boundary = DirichletBoundary([DBT_fix, DBT_disp])

        Symmetry: The displacement of dof 21 is negativ equal to the
        displacement of dof 20, i.e. u_20 + u_21 = 0

        >>> DBT_symm = [20, [20, 21], np.array([1, -1])]
        >>> my_boundary = DirichletBoundary([DBT_symm, ])
        '''
        # number of all dofs of the full system without boundary conditions
        self.no_of_unconstrained_dofs = no_of_unconstrained_dofs
        self.master_slave_list = master_slave_list  # boundary list
        self.B = None
        self.no_of_constrained_dofs = no_of_unconstrained_dofs

    def update(self):
        '''
        update internal variables according to internally saved boundary list.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.b_matrix()
        self.no_of_unconstrained_dofs, self.no_of_constrained_dofs = self.B.shape

    def constrain_dofs(self, dofs):
        '''
        Take the dofs to the constrained dofs.

        Parameters
        ----------
        dofs : ndarray
            Array containing the dofs to be constrained

        Returns
        -------
        None
        '''
        self.master_slave_list.append([None, dofs, None])
        self.update()

    def b_matrix(self):
        '''
        Parameters
        ----------
        None


        Returns
        -------
        B : scipy.sparse.csr_matrix
            Matrix (is usually Boolean, when no weightingfactors are assigned),
            which links the dofs of the constrained system to the dofs of the
            unconstrained system:

            >>> u = B @ u_constr

            If no Dirichlet Boundaries are chosen, B is identity.

        Examples
        --------
        Apply the constraints to a random stiffness matrix:

        >>> ndim = 100
        >>> K = sp.sparse.random(100,100, format='csr')
        >>> my_dirichlet_boundary = DirichletBoundary(ndim, [[None, [0,1,2,3,4], None]])
        >>> B = my_dirichlet_boundary.b_matrix()
        >>> B.T.dot(K.dot(B)) # B.T @ K @ B
        ... <95x95 sparse matrix of type '<class 'numpy.float64'>'
                with 93 stored elements in Compressed Sparse Column format>

        Notes
        -----

        Die globalen Freiheitsgrade differieren daher von den Freiheitsgraden
        des beschränkten Systems;
        Eingabeparameter ist eine Liste mit Dirichlet-Randbedingungen:
        [Master-DOF, [Liste_von_Sklaven-DOFs], Gewichtungsvektor]

        Master-DOF: (typ: int) Der DOF, auf den die Sklaven-DOFs projiziert
        werden. Der Master-DOF wird am ende eliminiert, d.h. er sollte
        üblicherweise auch in den Sklaven-DOFs auftauchen

        [Liste_von_Sklaven-DOFs]: (typ: liste mit ints) Die DOFs, die auf den
        Master-DOF projiziert werden. Zur Gewichtung wird der Gewichtungsvektor
        angewendet, der genauso viele Einträge haben muss wie die Sklaven-DOF-Liste

        Gewichtungsvektor: (typ: np.array oder None) TODO Beschreibung


        Wichtig: Für die Dirichlet-Randbedingungen werden Freiheitsgrade des
        globalen Systems und nicht die Knotenfreiheitsgrade berücksichtigt.
        Die Indexwerte der Knoten müssen stets in DOFs des globalen Sytems
        umgerechnet werden
        '''
        dofs_uncstr = self.no_of_unconstrained_dofs
        B = sp.sparse.eye(dofs_uncstr).tocsr()

        if self.master_slave_list == []:  # no boundary conditions
            self.B = B
            return B
        B_tmp = B*0
        global_slave_node_list = np.array([], dtype=int)
        global_master_node_list = np.array([], dtype=int)

        # Loop over all boundary items; the boundary information is stored in
        # the _tmp-Variables
        for master_node, slave_node_list, b_matrix in self.master_slave_list:

            # a little hack in order to get the types right
            if (type(b_matrix) != type(np.zeros(1))):
                # Make a B-Matrix, if it's not there
                b_matrix = np.ones(len(slave_node_list))

            if len(slave_node_list) != len(b_matrix):
                raise ValueError('Die Dimension der Sklaven-Knotenliste \
                entspricht nicht der Dimension des Gewichtungsvektors!')

            # check, if the master node is existent; otherwise the columns will
            # only be deleted
            if master_node != None:
                for i in range(len(slave_node_list)):
                    ## This is incredible slow!!!, but it's exactly what is done:
                    # B_tmp[:,master_node] += B[:,i]*b_matrix[i]
                    # so here's the alternative:
                    col = B[:,slave_node_list[i]]*b_matrix[i]
                    row_indices = col.nonzero()[0]
                    no_of_nonzero_entries = row_indices.shape[0]
                    col_indices = np.ones(no_of_nonzero_entries)*master_node
                    B_tmp = B_tmp + sp.sparse.csr_matrix((col.data,
                            (row_indices, col_indices)), shape=(dofs_uncstr, dofs_uncstr))

                global_master_node_list = np.append(global_master_node_list, master_node)
            global_slave_node_list = np.append(global_slave_node_list, slave_node_list)

        # Remove the master-nodes from B as they have to be overwritten by the slaves
        for i in global_master_node_list:
            B[i, i] -= 1
        B = B + B_tmp

        # Remove the master-nodes from the slave_node_list and mast the matrix such,
        # that the slave_nodes are removed
        global_slave_node_list = np.array(
            [i for i in global_slave_node_list if i not in global_master_node_list])
        mask = np.ones(dofs_uncstr, dtype=bool)
        mask[global_slave_node_list] = False
        B = B[:,mask]
        self.B = B
        return B

    def constrain_matrix(self, M_unconstr):
        '''
        Constrain a matrix with the given Dirichlet boundary conditions.

        Parameters
        ----------
        M_unconstr : sp.sparse.sparse_matrix
            Sparse unconstrained matrix

        Returns
        -------
        M : sp.sparse.sparse_matrix
            Sparse constrained matrix
        '''
        if not sp.sparse.issparse(self.B):
            B = self.b_matrix()
        else:
            B = self.B
        return B.T.dot(M_unconstr.dot(B))

    def constrain_vec(self, vec_unconstr):
        '''
        Constrain a vector with the given Dirichlet boundary conditions.

        Parameters
        ----------
        vec_unconstr : ndarray
            vector of unconstrained system

        Returns
        -------
        vec : ndarray
            vector of constrained system

        Notes
        -----
        The dimension of the returned `vec` is smaller than of `vec_unconstr`,
        as the fixed dofs are removed from the vector.
        '''
        if not sp.sparse.issparse(self.B):
            B = self.b_matrix()
        else:
            B = self.B
        return B.T.dot(vec_unconstr)

    def unconstrain_vec(self, vec):
        '''
        Remove the constraints of a vector.

        Parameters
        ----------
        vec : ndarray
            vector of the finite element system where constraints are imposed on.

        Returns
        -------
        vec_unconstr : ndarray
            Vector of hte finite element system where no constraints are imposed
            on. All dofs correspond to the dofs of the mesh.

        Notes
        -----
        The dimension of vec become larger, as the constrained dofs are added
        to the vector `vec`.
        '''
        if not sp.sparse.issparse(self.B):
            B = self.b_matrix()
        else:
            B = self.B
        return B.dot(vec)


class NeumannBoundary():
    '''
    Class for application of von Neumann boundary conditions.
    Works a little bit crazy but it's working.
    '''

    def __init__(self, no_of_dofs, neumann_boundary_list):
        self.neumann_boundary_list = neumann_boundary_list
        self.no_of_dofs = no_of_dofs
        pass

    def _harmonic_excitation(self, amplitude, frequency):
        '''
        function returning the harmonic function
        '''

        def internal_harmonic(t):
            return amplitude*np.cos(frequency*2*np.pi*t)

        return internal_harmonic


    def _step_function(self, step_size, time):
        '''
        function returning a step function
        '''
        def internal_step_function(t):
            if t > time:
                return step_size
            else:
                return 0

        return internal_step_function

    def _dirac_impulse(self, amplitude, time, time_interval):
        '''
        returns a dirac impulse as excitation
        '''

        def internal_dirac(t):
            if abs(t - time) < time_interval:
                return amplitude / time_interval
            else:
                return 0

        return internal_dirac

    def _ramp_function(self, slope, time):
        '''returns a ramp function '''

        def internal_ramp(t):
            delta_t = t - time
            if delta_t < 0:
                return 0
            else:
                return delta_t*slope

        return internal_ramp

    def _constant_function(self, amplitude):
        '''returns a constant function; this makes most sense for static applications'''

        def internal_constant(t):
            return amplitude

        return internal_constant

    function_dict = {'harmonic': _harmonic_excitation,
                     'stepload': _step_function,
                     'dirac'   : _dirac_impulse,
                     'ramp'    : _ramp_function,
                     'static'  : _constant_function}

    def f_ext(self):
        '''
        Parameters
        ----------
        None

        Returns
        -------
        a function f, which can be called with f(t) giving back the von neumann bcs

        '''
        self.function_list = []
        counter = 0
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        vals_global = np.array([], dtype=float)

        for dofs, type_, props, B_matrix in self.neumann_boundary_list:
            # constructing the indices for the boolean matrix grouping
            # the functions in the right place
            col = np.ones(len(dofs), dtype=int)*counter
            row = np.array(dofs)
            if B_matrix:
                vals = B_matrix
            else:
                vals = np.ones(len(dofs))

            row_global  = np.append(row_global, row)
            col_global  = np.append(col_global, col)
            vals_global = np.append(vals_global, vals)
            counter += 1

            # construct_the_list
            self.function_list.append(self.function_dict[type_](self, *props))

        self.boolean_force_matrix = sp.sparse.csr_matrix((vals_global,
            (row_global, col_global)), shape=(self.no_of_dofs, len(self.function_list)))

        # export external forcing function
        def external_forcing_function(t):
            return self.boolean_force_matrix.dot(np.array([i(t) for i in self.function_list]))
        return external_forcing_function


# TEST
if __name__ == '__main__':
    my_neumann_bc = NeumannBoundary(2000, [[[0,], 'dirac', (5, 0.5, 1E-3), None],])
    f = my_neumann_bc.f_ext()
    T = np.arange(-1, 5, 0.1)
    res = np.array([f(t) for t in T])
    from matplotlib import pyplot
    pyplot.plot(T, res[:, 0])
