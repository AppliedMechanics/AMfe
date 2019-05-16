#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import hstack as sphstack
from scipy.sparse import vstack as spvstack

from .constraint_formulation import ConstraintFormulationBase

__all__ = ['SparseLagrangeMultiplierConstraintFormulation']


class SparseLagrangeMultiplierConstraintFormulation(ConstraintFormulationBase):
    r"""
    Sparse Lagrange Multiplier Formulation for Sparse Matrices including scaling and Augmentation (Penalty)

    returns sparse matrices, although it can be used with a non-sparse unconstrained system

    This formulation transforms the system

    .. math::
        M_{\mathrm{raw}}(u, \dot{u}, t) \ddot{u} + h(u, \dot{u}, t) + B^T \lambda &= p(u, \dot{u}, t) \\
        g_{holo}(u, t) &= 0

    to

    .. math::
        M(x, dx, t) \ddot{x} + f_{int}(x, \dot{x}, t) = f_{ext}(x, \dot{x}, t)

    In detail:

    .. math::
        \begin{bmatrix} M_{raw} & 0 \\
        0 & 0
        \end{bmatrix} \begin{bmatrix} \ddot{u} \\
        \ddot{\lambda} \end{bmatrix} + \begin{bmatrix} h(u, \dot{u}, t) + s \cdot B^T \lambda + ps B^T g(u, t)  \\
            s g(u, t)
            \end{bmatrix} =
            \begin{bmatrix}
            p(u, \dot{u}, t) \\
            0
            \end{bmatrix}

    and the linearization

    .. math::
        \begin{bmatrix} M_{raw} & 0 \\
        0 & 0
        \end{bmatrix} \
        \begin{bmatrix} \ddot{u} \\
        \ddot{\lambda} \end{bmatrix} + \
         \begin{bmatrix} D_{raw} & 0 \\
            0 & 0
            \end{bmatrix} \
        \begin{bmatrix} \Delta \dot{u} \\
        \Delta \dot{\lambda} \end{bmatrix} + \
        \begin{bmatrix} K_{raw} + psB^T B & sB^T \\
        sB & 0
        \end{bmatrix} \
        \begin{bmatrix} \Delta u \\
        \Delta \lambda \end{bmatrix} = \
        \begin{bmatrix} p(\bar{u}, \dot{\bar{u}}, t) - h(\bar{u}, \dot{\bar{u}}, t) - s \cdot B^T \bar{\lambda}
        - ps B^T g(\bar{u}, t) \\
            - s g(\bar{u}, t)
            \end{bmatrix}
        = f_{ext}(\bar{u}, \dot{\bar{u}}, t) - f_{int}(\bar{u}, \dot{\bar{u}}, t)

    with

    .. math::
        K_{\mathrm{raw}} &= \frac{\partial (h-p)}{\partial u} \\
        D_{\mathrm{raw}} &= \frac{\partial (h-p)}{\partial {\dot u}}

    It includes a scaling factor s that scales the constraint equations and a penalization term in the tangential
    stiffness matrix (Augmentation or Penalization) that is scaled by a penalty factor p.

    Attributes
    ----------
    _M_full: csr_matrix
        Preallocated csr_matrix for M
    _D_full: csr_matrix
        Preallocated csr_matrix for D
    _K_full: csr_matrix
        Preallocated csr_matrix for K
    _f_int_full: csr_matrix
        Preallocated ndarray for f_int
    _f_ext_full: csr_matrix
        Preallocated ndarray for f_ext
    _scaling: float
        Scaling factor for scaling the constraint equation
    _penalty: float or None
        Penalty factor for Penalization of stiffness matrix K to achieve better conditioning
    """

    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, p_func=None,
                 jac_h_u=None, jac_h_du=None, jac_p_u=None, jac_p_du=None,
                 g_func=None, b_func=None, a_func=None):
        super().__init__(no_of_dofs_unconstrained, M_func, h_func, B_func, p_func,
                         jac_h_u, jac_h_du, jac_p_u, jac_p_du,
                         g_func, b_func, a_func)
        self._no_of_constraints = len(self._g_func(np.zeros(self._no_of_dofs_unconstrained), 0.0))
        self._M_full = None
        self._D_full = None
        self._K_full = None
        self._f_int_full = None
        self._f_ext_full = None
        self._scaling = 1.0
        self._penalty = None

    def _preallocate_M(self, M):
        """
        internal function for preallocation of Mass matrix

        Parameters
        ----------
        M: csr_matrix
            matrix containing the pattern of the M matrix before constraint formulation is carried out

        Returns
        -------
        M_full: csr_matrix
            preallocated matrix that will be returned after constraints are applied
        """
        if not isinstance(M, csr_matrix):
            if issparse(M):
                M = M.tocsr()
            else:
                M = csr_matrix(M)
        indptr = np.concatenate((M.indptr, np.ones(self._no_of_constraints, dtype=M.indptr.dtype) * M.indptr[-1]))
        return csr_matrix((M.data * 0.0, M.indices, indptr), shape=(M.shape[0] + self._no_of_constraints,
                                                                    M.shape[1] + self._no_of_constraints))

    def _preallocate_D(self, D):
        """
        internal function for preallocation of linear damping matrix

        Parameters
        ----------
        D: csr_matrix
            matrix containing the pattern of the D matrix before constraint formulation is carried out

        Returns
        -------
        D_full: csr_matrix
            preallocated matrix that will be returned after constraints are applied
        """
        return self._preallocate_M(D)

    def _preallocate_f(self):
        """
        internal function for preallocation of f_int and f_ext vector

        Returns
        -------
        F_full: numpy.array
            preallocated F array that will be returned after constraints are applied
        """
        return np.zeros(self._no_of_dofs_unconstrained + self._no_of_constraints)

    @property
    def dimension(self):
        """
        Returns the dimension of the system after constraints have been applied

        Returns
        -------
        dim: int
            dimension of the system after constraints are applied
        """
        return self._no_of_dofs_unconstrained + self._no_of_constraints

    def set_options(self, **options):
        """
        Sets options for the Lagrange formulation

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
            - 'penalty': float or None (scaling factor for Penalty Augmentation (if None, not applied))

        """
        self._scaling = options.get('scaling', self._scaling)
        self._penalty = options.get('penalty', self._penalty)

    def update(self):
        """
        Function that is called by observers if state has changed

        Returns
        -------
        None
        """
        self._no_of_constraints = len(self._g_func(np.zeros(self._no_of_dofs_unconstrained), 0.0))

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
        return x[:self._no_of_dofs_unconstrained]

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
        return dx[:self._no_of_dofs_unconstrained]

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
        return ddx[:self._no_of_dofs_unconstrained]

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
        return x[self.no_of_dofs_unconstrained:]

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
            \begin{bmatrix} M_{raw} & 0 \\
            0 & 0
            \end{bmatrix}

        """
        M = self._M_func(self.u(x, t), self.du(x, dx, t), t)
        if self._M_full is None:
            self._M_full = self._preallocate_M(M)
        if not isinstance(M, csr_matrix):
            if issparse(M):
                M = M.tocsr()
            else:
                M = csr_matrix(M)
        self._M_full.indptr = np.concatenate((M.indptr, np.ones(self._no_of_constraints,
                                                                dtype=M.indptr.dtype) * M.indptr[-1]))
        self._M_full.indices = M.indices
        self._M_full.data = M.data
        return self._M_full

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
            \begin{bmatrix} D_{raw} & 0 \\
            0 & 0
            \end{bmatrix}

        """
        u = self.u(x, t)
        du = self.du(x, dx, t)
        if self._jac_h_du is not None:
            if self._jac_p_du is not None:
                D = self._jac_h_du(u, du, t) - self._jac_p_du(u, du, t)
            else:
                D = self._jac_h_du(u, du, t)
        else:
            raise NotImplementedError('Numerical differentiation of h is not implemented yet')
        if self._D_full is None:
            self._D_full = self._preallocate_D(D)
        if not isinstance(D, csr_matrix):
            D = D.tocsr()
        self._D_full.indptr = np.concatenate((D.indptr, np.ones(self._no_of_constraints,
                                                                dtype=D.indptr.dtype) * D.indptr[-1]))
        self._D_full.indices = D.indices
        self._D_full.data = D.data
        return self._D_full

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
            \begin{bmatrix} h(u, \dot{u}, t) + s \cdot B^T \lambda   \\
            s g(u, t)
            \end{bmatrix}

        """
        if self._f_int_full is None:
            self._f_int_full = self._preallocate_f()
        u = self.u(x, t)
        du = self.du(x, dx, t)
        B = self._B_func(u, t)
        g = self._g_func(u, t)
        self._f_int_full *= 0.0
        self._f_int_full[:self._no_of_dofs_unconstrained] = self._h_func(u, du, t) + \
                                                            self._scaling * B.T.dot(x[self._no_of_dofs_unconstrained:])
        if self._penalty is not None:
            self._f_int_full[:self.no_of_dofs_unconstrained] += self._penalty * self._scaling * B.T.dot(g)
        self._f_int_full[self._no_of_dofs_unconstrained:] = self._scaling * g

        return self._f_int_full

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
            \begin{bmatrix} p(u, \dot{u}, t)   \\
            0
            \end{bmatrix}

        """
        if self._f_ext_full is None:
            self._f_ext_full = self._preallocate_f()
        u = self.u(x, t)
        du = self.du(x, dx, t)
        self._f_ext_full *= 0.0
        if self._p_func is not None:
            self._f_ext_full[:self._no_of_dofs_unconstrained] = self._p_func(u, du, t)
        return self._f_ext_full

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
            \begin{bmatrix} K_{raw} + psB^T B & sB^T \\
            sB & 0
            \end{bmatrix}

        Attention: d(B.T@g)/dq is evaluated as = B.T@dg/dq, which means that dB/dq is assumed to be zero.
        This is done because dB/dq could be expensive to evaluate.
        """
        B = self._B_func(self.u(x, t), t)
        K = self._jac_h_u(self.u(x, t), self.du(x, dx, t), t)
        if self._penalty is not None:
            K += self._penalty * self._scaling * B.T.dot(B)

        return spvstack((sphstack((K, self._scaling * B.T), format='csr'),
                         sphstack((self._scaling * B, csr_matrix((self._no_of_constraints,
                                                                  self._no_of_constraints))), format='csr')),
                        format='csr')
