#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.sparse import identity, issparse

from amfe.linalg.tools import isboolean
from .constraint_formulation import ConstraintFormulationBase


class BooleanEliminationConstraintFormulation(ConstraintFormulationBase):
    """
    Works only with holonomic scleronomic constraints that result in a constant Boolean B matrix
    (Jacobian of the constraint function)

    Attributes
    ----------
    _L: csr_matrix
        Matrix that is able to eliminate the constrained dofs by applying :math:`L^T A L` to a matrices A
    _L_changed: bool
        Internal flag that indicates if L must be updated when it is asked for the next time

    Notes
    -----
    Currently there is no check if this formulation is allowed to use!
    It may only be used for constraints defined by Bu = 0 with boolean matrix B
    """
    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, jac_h_u=None, jac_h_du=None, g_func=None,
                 b_func=None, a_func=None):
        super().__init__(no_of_dofs_unconstrained, M_func, h_func, B_func, jac_h_u, jac_h_du, g_func, b_func, a_func)
        self._L = None
        self._L_changed = True  # Setting flag for lazy evaluation

    @property
    def dimension(self):
        """
        Returns the dimension of the system after constraints have been applied

        Returns
        -------
        dim: int
            dimension of the system after constraints are applied
        """
        return self.L.shape[1]

    @property
    def L(self):
        """
        Returns the L matrix that is able to eliminate the constrained dofs by applying :math:`L^T A L` to a matrices A

        Returns
        -------
        L: csr_matrix
            The matrix L
        """
        if self._L_changed:
            self._compute_L()
            self._L_changed = False
        return self._L

    def update(self):
        """
        Function that is called by observers if state has changed

        Returns
        -------
        None
        """
        # This class assumes that the C matrix is constant and Boolean
        self._L_changed = True

    def _compute_L(self):
        """
        Internal function that computes the matrix L

        The function is called when L must be updated
        L is the nullspace of B

        Returns
        -------
        None
        """
        # Boolean elimination assumes that C is constant (scleronomic) and independent on q!
        # Thus, C is called by just calling for any arbitrary values, q and t
        q = np.zeros(self._no_of_dofs_unconstrained, dtype=float)
        t = 0.0
        B = self._B_func(q, t)
        constrained_dofs = self._get_constrained_dofs_by_B(B)
        if issparse(B):
            self._L = self._get_L_by_constrained_dofs(constrained_dofs, B.shape[1], format='csr')
        else:
            self._L = self._get_L_by_constrained_dofs(constrained_dofs, B.shape[1], format='dense')

    @staticmethod
    def _get_constrained_dofs_by_B(B):
        """
        Static method that computes the indices of those dofs that are constrained when a matrix B is given that
        is boolean

        Parameters
        ----------
        B: csr_matrix
            B is a matrix coming from the constraint definitions: B q + b = 0

        Returns
        -------

        """
        # Check if B is boolean
        # later also for substructuring coupling: if np.array_equal(np.abs(B_boolean), np.abs(B_boolean).astype(bool)):
        if isboolean(B):
            # check if only one 1 is in each row:
            if issparse(B):
                Bcsr = B.tocsr()
                if np.array_equal(Bcsr.indptr, np.arange(len(Bcsr.indices)+1)):
                    constrained_dofs = B.tocsr().indices.tolist()
                else:
                    raise ValueError('B_boolean must have exactly one 1-entry per row')
            else:
                counts = np.count_nonzero(B, axis=1)
                if np.all(counts == 1):
                    constrained_dofs = list()
                    for row in B:
                        index = np.where(row == 1)[0][0]
                        constrained_dofs.append(index)
                else:
                    raise ValueError('B_boolean must have exactly one 1-entry per row')
            return constrained_dofs

        else:
            raise ValueError('B_boolean must be a Boolean array')

    @staticmethod
    def _get_L_by_constrained_dofs(constrained_dofs, ndof_unconstrained, format='csr'):
        """
        Internal static function that computes L by given indices of constrained dofs

        Parameters
        ----------
        constrained_dofs: list or ndarray
            list containing the indices of the constrained dofs
        ndof_unconstrained: int
            number of dofs of the unconstrained system
        format: str
            format = 'csr' or 'dense' describes the format of L

        Returns
        -------
        L: csr_matrix
            computed L matrix
        """
        L = identity(ndof_unconstrained, format='csr')
        col_idxs_not_to_remove = np.arange(0, ndof_unconstrained)
        col_idxs_not_to_remove = np.delete(col_idxs_not_to_remove, constrained_dofs)
        if format == 'csr':
            return L[:, col_idxs_not_to_remove]
        elif format == 'dense':
            return L[:, col_idxs_not_to_remove].toarray()
        else:
            raise ValueError('Only csr or dense format allowed')

    def u(self, x, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        t: float
            time

        Returns
        -------
        u: numpy.array
            recovered displacements of the unconstrained system

        """
        return self.L.dot(x)

    def du(self, x, dx, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        du: numpy.array
            recovered velocities of the unconstrained system

        """
        return self.L.dot(dx)

    def ddu(self, x, dx, ddx, t):
        """

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        ddx: numpy.array
            Second time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        ddu: numpy.array
            recovered accelerations of the unconstrained system

        """
        return self.L.dot(ddx)

    def lagrange_multiplier(self, x, t):
        """
        Recovers the lagrange multipliers of the unconstrained system

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        t: float
            time

        Returns
        -------
        lambda_: numpy.array
            recovered displacements of the unconstrained system

        """
        return np.array([], ndmin=1, dtype=float)

    def M(self, x, dx, t):
        r"""
        Returns the constrained mass matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        M: csr_matrix
            Constrained mass matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T M_{raw} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        return self.L.T.dot(self._M_func(u, du, t)).dot(self.L)

    def F(self, x, dx, t):
        r"""
        Returns the constrained F vector

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        F: numpy.array
            Constrained F vector

        Notes
        -----
        In this formulation this returns

        .. math::
            L^T h(u, du, t)

        """

        u = self.u(x, t)
        du = self.du(x, dx, t)
        return self.L.T.dot(self._h_func(u, du, t))

    def K(self, x, dx, t):
        r"""
        Returns the constrained stiffness matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        K: csr_matrix
            Constrained mass matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            - L^T \frac{\mathrm{d}h}{\mathrm{d} u} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_u is not None:
            return -self.L.T.dot(self._jac_h_u(u, du, t)).dot(self.L)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')

    def D(self, x, dx, t):
        r"""
        Returns the constrained damping matrix

        Parameters
        ----------
        x: numpy.array
            Global state vector of the system
        dx: numpy.array
            First time derivative of global state vector of the constrained system
        t: float
            time

        Returns
        -------
        D: csr_matrix
            Constrained damping matrix

        Notes
        -----
        In this formulation this returns

        .. math::
            - L^T \frac{\mathrm{d}h}{\mathrm{d} \dot{u}} L

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_u is not None:
            return -self.L.T.dot(self._jac_h_du(u, du, t)).dot(self.L)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')
