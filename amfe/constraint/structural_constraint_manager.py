#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module handling structural constraints
"""

import numpy as np
from scipy.sparse import eye as eyes

from .constraint_manager import ConstraintManager

__all__ = [
    'StructuralConstraintManager',
]


class StructuralConstraintManager(ConstraintManager):
    """
    Class responsible for constraints in the structural component

    Attributes
    ----------
    _no_of_unconstrained_dofs : int
        number of dofs of the unconstrained component
    _constraints : list
        list of dicts with keys 'dofsarg', 'obj', 'strategy'
        the dofsarg value is an iterable with global degrees of freedom that must be passed to the constraint
        the obj is a constraint object
        the strategy is a string that describes the strategy (e.g. 'elim' for elimination)
    _slave_dofs : ndarray, dtype:int
        array with global ids of the slave dofs that are eliminated by the constraintmanager
    _L : ndarray
        matrix with boolean values to eliminate dofs form vectors and matrices
    """

    # Strategies to apply constraints
    # currently: elimination and lagrange multiplier
    STRATEGIES = [
        'elim',
        'lagrmult'
    ]

    def __init__(self, ndof_unconstrained_system=0):
        """
        Parameters
        ----------
        ndof_unconstrained_system : int
            number of dofs of the unconstrained system.
        """
        super().__init__()
        self._no_of_unconstrained_dofs = ndof_unconstrained_system
        self._constraints = []
        self._slave_dofs = np.array([], dtype=int)
        self._L = None
        return

    def add_constraint(self, constraint, dofsarg, strategy):
        """
        Method for adding a structural constraint

        Parameters
        ----------
        constraint : StructuralConstraint
            constraint object, describing the constraint
        dofsarg : tuple
            dofs' indices that must be passed to the constraint
        strategy : str
            strategy how the constraint shall be applied (e.g. via elimination or lagrange multiplier)

        Returns
        -------
        None
        """

        if strategy not in self.STRATEGIES:
            raise ValueError('strategy must be \'elim\' or \'lagrmult\'')

        self._constraints.append({'dofsarg': np.array(dofsarg, dtype=int), 'obj': constraint, 'strategy': strategy})
        # FIXME: Prüfung ob slave-dofs passen sehr aufwendig, bitte schlauer programmieren!
        # if [dof for dof in dofs if dof in self._slave_dofs]:
        #    raise ValueError('some of the dofs are already constrained')
        if strategy == 'elim':
            slave_dofs = constraint.slave_dofs(dofsarg)
            self._slave_dofs = np.append(self._slave_dofs, slave_dofs)
        return

    def constrain_m(self, M_unconstr, t=0.):
        """
        Constrain the mass matrix of the structural component

        Parameters
        ----------
        M_unconstr : csr_matrix
            Unconstrained mass matrix

        Returns
        -------
        M_constr : csr_matrix
            Constrained mass matrix
        """

        return self.L.T @ M_unconstr @ self.L

    def constrain_k(self, K_unconstr, t=0.):
        """
        Constrain the linear stiffness matrix of the structural component

        Parameters
        ----------
        K_unconstr : csr_matrix
            Unconstrained linear stiffness matrix

        Returns
        -------
        K_constr : csr_matrix
            Constrained linear stiffness matrix
        """

        return self.L.T @ K_unconstr @ self.L

    def constrain_d(self, D_unconstr, t=0.):
        """
        Constrain the viscous damping matrix of the structural component

        Parameters
        ----------
        D_unconstr : csr_matrix
            Unconstrained viscous damping matrix

        Returns
        -------
        D_constr : csr_matrix
            Constrained viscous damping matrix
        """

        return self.L.T @ D_unconstr @ self.L

    def constrain_f_int(self, f_int_unconstr, t=0.):
        """
        Constrain the internal force vector of the structural component

        Parameters
        ----------
        f_int_unconstr : ndarray
            Unconstrained internal force vector

        Returns
        -------
        f_int_constr : ndarray
            Constrained internal force vector
        """

        return self.L.T @ f_int_unconstr

    def constrain_f_ext(self, f_ext_unconstr, t=0.):
        """
        Constrain the external force vector of the structural component

        Parameters
        ----------
        f_ext_unconstr : ndarray
            Unconstrained external force vector

        Returns
        -------
        f_ext_constr : ndarray
            Constrained external force vector
        """

        return self.L.T @ f_ext_unconstr

    def unconstrain_u(self, u_constr, t):
        """
        Extend the u_constrained vector by the constrained displacements at time t

        Parameters
        ----------
        u_constr : ndarray
            Displacement vector of the unconstrained system
        t : float
            time

        Returns
        -------
        u_unconstr : ndarray
            Unconstrained displacements = u_constr extended by constrained displacements at time t
        """

        u_unconstr = self.L @ u_constr
        for constraint in self._constraints:
            u_unconstr[constraint['dofsarg']] = constraint['obj'].u(t)
        return u_unconstr

    def constrain_u(self, u_unconstr, t=0.):
        """
        Constrain the u_unconstr = eliminate the dofs that must be eliminated

        Parameters
        ----------
        u_unconstr : ndarray
            Displacement vector of the unconstrained system
        t : float
            time

        Returns
        -------
        u_unconstr : ndarray
            Unconstrained displacements = u_constr extended by constrained displacements at time t
        """

        return self.L.T @ u_unconstr

    def get_rhs_nl(self, t, M_unconstr, D_unconstr=None):
        """
        Returns an additional rhs for the nonlinear equation of motion when constraints are applied by elimination

        Parameters
        ----------
        t : int
            time
        M_unconstr : csr_matrix
            Unconstrained csr_matrix
        D_unconstr : csr_matrix
            Unconstrained viscous damping matrix

        Returns
        -------
        f_nl_rhs : ndarray
            Additional RHS for the nonlinear equation of motion when constraints are applied by elimination
        """

        ndof = self._no_of_unconstrained_dofs
        result = np.zeros(self.no_of_constrained_dofs)
        for constraint in self._constraints:
            if constraint['strategy'] == 'elim':
                constraintobj = constraint['obj']
                mask = np.zeros(ndof, dtype=bool)
                mask[constraintobj.slave_dofs(constraint['dofsarg'])] = True
                result -= (self.L.T @ M_unconstr)[:, mask] @ constraintobj.ddu(t)
                if D_unconstr is not None:
                    result -= (self.L.T @ D_unconstr)[:, mask] @ constraintobj.du(t)
        return result

    def get_rhs_nl_static(self, t):
        """
        Returns an additional rhs for static investigations for the nonlinear equation of motion
        when constraints are applied by elimination

        Parameters
        ----------
        t : int
            time

        Returns
        -------
        f_nl_rhs_static : ndarray
            Additional RHS for the static nonlinear equation of motion when constraints are applied by elimination
        """

        return np.zeros(self.no_of_constrained_dofs)

    def get_rhs_lin(self, t, M_unconstr, K_unconstr, D_unconstr=None):
        """
        Returns an additional rhs for the linear equation of motion
        when constraints are applied by elimination

        Parameters
        ----------
        t : int
            time
        M_unconstr : csr_matrix
            Unconstrained csr_matrix
        K_unconstr : csr_matrix
            Unconstrained csr_matrix
        D_unconstr : csr_matrix
            Unconstrained viscous damping matrix

        Returns
        -------
        f_lin_rhs : ndarray
            Additional RHS for the linear equation of motion when constraints are applied by elimination
        """

        ndof = self._no_of_unconstrained_dofs
        result = np.zeros(self.no_of_constrained_dofs)
        for constraint in self._constraints:
            if constraint['strategy'] == 'elim':
                constraintobj = constraint['obj']
                mask = np.zeros(ndof, dtype=bool)
                mask[constraintobj.slave_dofs(constraint['dofsarg'])] = True
                result -= (self.L.T @ M_unconstr)[:, mask] @ constraintobj.ddu(t) \
                          + (self.L.T @ K_unconstr)[:, mask] @ constraintobj.u(t)
                if D_unconstr is not None:
                    result -= (self.L.T @ D_unconstr)[:, mask] @ constraintobj.du(t)
        return result

    def get_rhs_lin_static(self, t, K_unconstr):
        """
        Returns an additional rhs for the linear equation of motion
        when constraints are applied by elimination

        Parameters
        ----------
        t : int
            time
        K_unconstr : csr_matrix
            Unconstrained csr_matrix

        Returns
        -------
        f_lin_rhs_static : ndarray
            Additional RHS for the static linear equation of motion when constraints are applied by elimination
        """

        ndof = self._no_of_unconstrained_dofs
        result = np.zeros(self.no_of_constrained_dofs)
        for constraint in self._constraints:
            if constraint['strategy'] == 'elim':
                constraintobj = constraint['obj']
                mask = np.zeros(ndof, dtype=bool)
                mask[constraintobj.slave_dofs(constraint['dofsarg'])] = True
                result -= (self.L.T @ K_unconstr)[:, mask] @ constraintobj.u(t)
        return result

    def update_l(self):
        """
        Update the L matrix that eliminates the dofs that are eliminated by constraints

        Returns
        -------
        None
        """

        ndof = self._no_of_unconstrained_dofs
        L_raw = eyes(ndof, format='csr', dtype=bool)
        if len(self._slave_dofs) > 0:
            mask = np.ones(ndof, dtype=bool)
            mask[self._slave_dofs] = False
            self._L = L_raw[:, mask]
        else:
            self._L = L_raw
        return

    @property
    def L(self):
        """
        Returns
        -------
        L : ndarray
            Retuns the L matrix that eliminates the dofs that are eliminated by constraints
        """

        if self._L is None:
            self.update_l()
        return self._L

    @property
    def no_of_constrained_dofs(self):
        """
        Gives the number of dofs of the constrained system

        Returns
        -------
        no_of_constrained_dofs : int
            Number of dofs of the constrained system
        """

        return self._no_of_unconstrained_dofs - len(self._slave_dofs)
