#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.linalg import null_space
from scipy.sparse import csr_matrix, identity
from scipy.sparse import vstack as spvstack

from .constraint_formulation import ConstraintFormulationBase


class NullspaceConstraintFormulation(ConstraintFormulationBase):
    r"""
    The Nullspace Constraint Formulation belongs to the class of Acceleration level formulations.

    This formulation transforms the system

    .. math::
        M_{\mathrm{raw}}(u, \dot{u}, t) \ddot{u} + h(u, \dot{u}, t) + B^T \lambda &= p(u, \dot{u}, t) \\
        B(u, t) \ddot{u} + a(u, \dot{u}, t) &= 0

    to

    .. math::
        M(x, dx, t) \ddot{x} + f_{int}(x, \dot{x}, t) = f_{ext}(x, \dot{x}, t)

    In detail:

    .. math::
        \begin{bmatrix} L^T M_{raw} \\
        s B(u, t) \end{bmatrix} \ddot{u}
        + \begin{bmatrix} L^T h(u, \dot{u}, t)\\
            s a(u, \dot{u}, t) \end{bmatrix}
        = \begin{bmatrix} L^T p(u, \dot{u}, t)\\
            0
        \end{bmatrix}

    where L is the nullspace of B in the acceleration level constraint equation
    :math:`B(u, t) \ddot{u} + a(u, \dot{u}, t) = 0`.

    Attributes
    ----------
    _L: csr_matrix
        internal storage of the L matrix (nullspace of last asked B)
    _D_full: csr_matrix
        internal storage of the linearized viscous damping matrix D
    _scaling: float
        scaling factor that scales the constraint equation in the formulation (default 1.0)
    """
    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, p_func=None,
                 jac_h_u=None, jac_h_du=None, jac_p_u=None, jac_p_du=None,
                 g_func=None, b_func=None, a_func=None):
        super().__init__(no_of_dofs_unconstrained, M_func, h_func, B_func, p_func,
                         jac_h_u, jac_h_du, jac_p_u, jac_p_du, g_func, b_func, a_func)
        self._no_of_constraints = len(self._g_func(np.zeros(self._no_of_dofs_unconstrained), 0.0))
        if self._a_func is None:
            z = np.zeros(self._no_of_constraints)

            def a_func_zero(u, du, t):
                return z

            self._a_func = a_func_zero
        self._L = None
        self._D_full = None
        self._scaling = 1.0

    def _preallocate_D(self, D):
        """
        Preallocates the D function

        Parameters
        ----------
        D

        Returns
        -------

        """
        if not isinstance(D, csr_matrix):
            D = D.tocsr()
        indptr = np.concatenate((D.indptr, np.ones(self._no_of_constraints, dtype=D.indptr.dtype)*D.indptr[-1]))
        return csr_matrix((D.data*0.0, D.indices, indptr), shape=(D.shape[0] + self._no_of_constraints,
                          D.shape[1]))

    def L(self, x, t):
        """
        Returns the Nullspace of the B matrix

        Parameters
        ----------
        x: ndarray
            System vector x
        t: float
            time

        Returns
        -------
        L: csr_matrix
            nullspace of B(x, t)
        """
        u = self.u(x, t)
        B = self._B_func(u, t)
        if B.shape[0] > 0:
            self._L = csr_matrix(null_space(B.todense()))
        else:
            self._L = identity(self._no_of_dofs_unconstrained, format='csr')
        return self._L

    @property
    def dimension(self):
        """
        Returns the dimension of the system after constraints have been applied

        Returns
        -------
        dim: int
            dimension of the system after constraints are applied
        """
        return self.no_of_dofs_unconstrained

    def set_options(self, **options):
        """
        Sets options for the Nullspace Elimination formulation

        Parameters
        ----------
        options: dict
            Key value dict describing the options to apply

        Returns
        -------

        Notes
        -----
        Available Options:
            - 'scaling': float (scaling factor for constraint function)

        """
        self._scaling = options.get('scaling', self._scaling)

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
        return x

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
        return dx

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
        return ddx

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
            \begin{bmatrix} L^T M_{raw} \\
            s B(u, t)
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        M = self._M_func(u, du, t)
        return spvstack((csr_matrix(self.L(x, t).T.dot(M)),
                         self._scaling*self._B_func(u, t)), format='csr')

    def f_int(self, x, dx, t):
        r"""
        Returns the constrained f_int vector

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
        f_int: numpy.array
            Constrained f_int vector

        Notes
        -----
        In this formulation this returns

        .. math::
            \begin{bmatrix} L^T h(u, \dot{u}, t)   \\
            s a(u, \dot{u}, t)
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        return np.concatenate((self.L(x, t).T.dot(self._h_func(u, du, t)),
                               self._scaling*self._a_func(u, du, t)), axis=0)

    def f_ext(self, x, dx, t):
        r"""
        Returns the constrained f_ext vector

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
        f_ext: numpy.array
            Constrained f_ext vector

        Notes
        -----
        In this formulation this returns

        .. math::
            \begin{bmatrix} L^T p(u, \dot{u}, t)   \\
            0
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        return np.concatenate((self.L(x, t).T.dot(self._p_func(u, du, t)),
                               np.zeros(self._no_of_constraints)), axis=0)

    def K(self, x, dx, t):
        r"""
        Returns the constrained stiffness matrix

        This is an approximation! The upper part, namely the B of the internal and external forces
        is exactly evaluated. But the B of the a_function is not available and set to zero!

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
            \begin{bmatrix} L^T K_{raw}\\
            0
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        K = self._jac_h_u(u, du, t)
        if self._jac_p_u is not None:
            K -= self._jac_p_u(u, du, t)

        return spvstack((self.L(x, t).T.dot(K), csr_matrix((self._no_of_constraints, self._no_of_dofs_unconstrained))),
                        format='csr')

    def D(self, x, dx, t):
        r"""
        This is an approximation of D. The upper part, namely the B of the internal and external forces is exactly
        evaluated. But the B of the c_function w.r.t. the velocities is not available and set to zero!

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
            \begin{bmatrix} D_{raw} & 0 \\
            0 & 0
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_u is not None:
            if self._jac_p_du is not None:
                D = self._jac_h_du(u, du, t) - self._jac_p_du(u, du, t)
            else:
                D = self._jac_h_du(u, du, t)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')
        D = self.L(x, t).T.dot(D)
        if self._D_full is None:
            self._D_full = self._preallocate_D(D)
        if not isinstance(D, csr_matrix):
            D = D.tocsr()
        self._D_full.indptr = np.concatenate((D.indptr, np.ones(self._no_of_constraints,
                                                                dtype=D.indptr.dtype)*D.indptr[-1]))
        self._D_full.indices = D.indices
        self._D_full.data = D.data
        return self._D_full
