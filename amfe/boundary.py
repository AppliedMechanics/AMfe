# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""Module for handling the Dirichlet boundary. """

__all__ = ['DirichletBoundary',
           ]

import numpy as np
import scipy as sp


class DirichletBoundary():
    '''
    Class responsible for the Dirichlet Boundary conditions

    Attributes
    ----------
    B : sparse matrix
        Matrix for mapping the constrained dofs to the unconstrained dofs:
            u_unconstr = B @ u_constr
        The matrix is a sparse CSR matrix
    slave_dofs : ndarray
        Array of unique slave dof indices which are eliminated in the
        constrained dofs
    row : ndarray, dtype: int
        row indices of triplet sparse matrix description for C (constraints)
    col : ndarray, dtype: int
        col indices of triplet sparse matrix description for C (constraints)
    val : ndarray, dtype: float
        vals of triplet sparse matrix description for C (constraints)
    no_of_unconstrained_dofs : int
        number of unconstrained dofs
    no_of_constrained_dofs : int
        number of constrained dofs

    '''
    def __init__(self, no_of_unconstrained_dofs=np.nan):
        '''
        Parameters
        ----------
        ndof_unconstrained_system : int
            Number of dofs of the unconstrained system.

        '''
        self.B = None
        self.slave_dofs = np.array([])
        self.row = np.array([])
        self.col = np.array([])
        self.val = np.array([])
        # number of all dofs of the full system without boundary conditions
        self.no_of_unconstrained_dofs = no_of_unconstrained_dofs
        self.no_of_constrained_dofs = no_of_unconstrained_dofs
        return

    def update(self):
        '''
        update internal variables according to internally saved boundary list:
            - calls self.b_matrix()
            - update self.no_of_constrained_dofs, self.no_of_unconstrained_dofs

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.b_matrix()
        self.no_of_unconstrained_dofs, self.no_of_constrained_dofs = self.B.shape
        return

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
        slave_dofs = np.append(self.slave_dofs, dofs)
        self.slave_dofs = np.array(np.unique(slave_dofs), dtype=int)
        self.update()
        return


    def add_constraints(self, slave_dofs, row, col, val):
        '''
        Add constraints to the system.

        The slave_dofs are eliminated. The triple row, col and val expresses the
        triplet entries in the B-matrix mapping the constrained dofs to the
        unconstrained dofs:

            u_unconstr = B @ u_constr

        Parameters
        ----------
        slave_dofs : array-like, dtype int
            array or list of slave dofs which should be eliminated
        row : ndarray, dtype int, shape(n)
            row index list of master-slave mapping
        col : ndarray, dtype int, shape(n)
            col index list of master-slave mapping
        val : ndarray, shape(n)
            values of master-slave mapping

        Returns
        -------
        B : sparse_matrix, shape(ndim_unconstr, ndim_constr)
            sparse matrix performing the mapping from the constrained dofs to the unconstrained dofs.

        '''
        assert(len(row) == len(col) == len(val))

        slave_dofs_tmp = np.append(self.slave_dofs, slave_dofs)
        self.slave_dofs = np.array(np.unique(slave_dofs_tmp), dtype=int)

        self.row = np.array(np.append(self.row, row), dtype=int)
        self.col = np.array(np.append(self.col, col), dtype=int)
        self.val = np.append(self.val, val)

        self.update()
        return

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
        # TODO: CHECK: The next line can be deleted because deprecated, right?
        >>> my_dirichlet_boundary = DirichletBoundary(ndim, [[None, [0,1,2,3,4], None]])
        >>> B = my_dirichlet_boundary.b_matrix()
        >>> B.T.dot(K.dot(B)) # B.T @ K @ B
        ... <95x95 sparse matrix of type '<class 'numpy.float64'>'
                with 93 stored elements in Compressed Sparse Column format>

        Notes
        -----

        Wichtig: Für die Dirichlet-Randbedingungen werden Freiheitsgrade des
        globalen Systems und nicht die Knotenfreiheitsgrade berücksichtigt.
        Die Indexwerte der Knoten müssen stets in DOFs des globalen Sytems
        umgerechnet werden
        '''
        ndof = self.no_of_unconstrained_dofs
        B_raw = sp.sparse.eye(ndof) \
                   + sp.sparse.csr_matrix((self.val, (self.row, self.col)),
                                          shape=(ndof, ndof))

        if len(self.slave_dofs) > 0:
            mask = np.ones(ndof, dtype=bool)
            mask[self.slave_dofs] = False
            self.B = B_raw[:,mask]
        else:
            self.B = B_raw

        return self.B

    def apply_master_slave_list(self, master_slave_list):
        '''
        Apply a master-slave list of the form

        Parameters
        ----------
        master_slave_list : list
            list containing the dirichlet-boundary triples (DBT)

            >>> [DBT_1, DBT_2, DBT_3, ]

        Returns
        -------
        None

        Notes
        -----
        each dirchilet_boundary_triple is itself a list containing

        >>> DBT = [master_dof=None, [list_of_slave_dofs], weighting_matrix=None]

        master_dof : int / None
            the dof onto which the slave dofs are projected. The master_dof
            will be overwritten at the end, i.e. if the master dof should
            participate at the end, it has to be a member in the list of
            slave_dofs. If the master_dof is set to None, the slave_dofs will
            be fixed
        list_of_slave_dofs : list containing ints
            The list of the dofs which will be projected onto the master dof;
            the weights of the projection are stored in the B_matrix
        weighting_matrix : ndarray / None
            The weighting-matrix which enables to apply complicated
            boundary conditions showing up in symmetry-conditions or rotational
            dofs. The default-value for weighting_matrix is None, which weighs all
            members of the list_of_slave_dofs equally with 1.

        Examples
        --------

        The dofs 0, 2 and 4 are fixed:

        >>> DBT = [None, [0, 2, 4], None]
        >>> my_boundary = DirichletBoundary([DBT, ])

        The dofs 0, 1, 2, 3, 4, 5, 6 are fixed and the dofs 100, 101, 102, 103
        have all the same displacements:

        >>> DBT_fix = [None, np.arange(7), None]
        >>> DBT_disp = [100, [101, 102, 103], None]
        >>> my_boundary = DirichletBoundary([DBT_fix, DBT_disp])

        Symmetry: The displacement of dof 21 is negativ equal to the
        displacement of dof 20, i.e. u_20 + u_21 = 0

        >>> DBT_symm = [20, [21], np.array([-1])]
        >>> my_boundary = DirichletBoundary([DBT_symm, ])

        '''

        # Diese Funktion funktioniert, ist aber verbesserungswürdig.
        # Beispielsweise wird hier immer nur von einem einzigen Master_dof in einem constraint ausgegangen.
        # Das ist eigentlich der falsche Ansatz. Es sollten mehrere Master-Dofs erlaubt sein.
        # Vergleich mit Nastran: Mit diesem Ansatz ist ein RBE2-Element möglich, aber kein RBE3-Element.

        for line in master_slave_list:
            master_dof = line[0]
            slave_dofs = line[1]
            weighting_matrix = line[2]
            if master_dof is None:
                self.constrain_dofs(slave_dofs)
                continue
            else:
                col = np.ones_like(slave_dofs) * master_dof
                row = slave_dofs
                if weighting_matrix is None:
                    weighting_matrix = np.ones_like(slave_dofs)
                else:
                    assert(len(weighting_matrix) == len(slave_dofs))
                val = weighting_matrix
                self.add_constraints(slave_dofs, row, col, val)
        return

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
        Returns B @ u

        Parameters
        ----------
        vec : ndarray
            vector of the finite element system where constraints are imposed on.

        Returns
        -------
        vec_unconstr : ndarray
            Vector of the finite element system where no constraints are imposed
            on. All dofs correspond to the dofs of the mesh.

        Notes
        -----
        The dimension of vec becomes larger, as the constrained dofs are added
        to the vector `vec`.
        '''
        if not sp.sparse.issparse(self.B):
            B = self.b_matrix()
        else:
            B = self.B
        return B.dot(vec)

