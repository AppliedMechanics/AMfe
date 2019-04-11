#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
r"""
Constraint Formulations

It expects a system of this type:

.. math::
    M(u, \dot{u}, t) \ddot{u} + B^T \lambda &= h(u, \dot{u}, t) \\
    g_{holo}(u, t) &= 0 \\
    g_{nonholo}(u, \dot{u}, t) &= B_{nonholo}(u, t) * \dot{u} + b(u, t) = 0

And returns this type of system:

.. math::
    M(x, dx, t) \ddot{x} = F(x, \dot{x}, t)

Here x is the full state vector of the system. Depending on the constraint formulation this can e.g. be a mixture of
displacements :math:`u` and Lagrange Multipliers :math:`\lambda`

Furthermore, it provides the following entities:

- Linear Damping matrix
.. math::
    D = -\frac{\mathrm{d}F}{\mathrm{d} \dot{x}}

- Linear Stiffness matrix
.. math::
    K = -\frac{\mathrm{d}F}{\mathrm{d} x}

It also can recover the u, du, ddu of the unconstrained system from the system states x of the constrained system
"""


class ConstraintFormulationBase:
    r"""
    Applies constraints to general system

    .. math::
        M(u, \dot{u}, t) \ddot{u} + B^T \lambda = h(u, \dot{u}, t)

    Attributes
    ----------
    _no_of_dofs_unconstrained: int
        number of dofs of the unconstrained system, number of entries in vector u
    _M_func: function
        function with signature M(u, du, t) returning the mass matrix
    _h_func: function
        function with signature h(u, du, t) returning the nonlinear forces
    _B_func: function
        function with signature B(u, t) returning the linear map of the constraint function on velocity level
        mapping the velocities to the residual of the constraint equation (B*dq + b) = 0
    _jac_h_u: function
        Jacobian of the _h_func w.r.t. displacements u
    _jac_h_du: function
        Jacobian of the _h_func w.r.t. velocities du
    _g_func: function
        Function with signature g(u, t) returning the residual of the holonomic constraints on displacement level
    _b_func: function
        Function returning the nonhomogeneous part of the constraint equation on velocity level
        :math:`B(u, t) \dot{u} + b(u, t) = 0` (last term b)
    _a_func: function
        Function returning the nonhomogeneous part of the constraint equation on acceleration level
        :math:`B(u, t) \ddot{u} + a(u, t) = 0` (last term a)

    """
    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, jac_h_u=None, jac_h_du=None, g_func=None,
                 b_func=None, a_func=None):
        self._no_of_dofs_unconstrained = no_of_dofs_unconstrained
        self._M_func = M_func
        self._h_func = h_func
        self._B_func = B_func
        self._jac_h_u = jac_h_u
        self._jac_h_du = jac_h_du
        self._g_func = g_func
        self._b_func = b_func
        self._a_func = a_func
        return

    @property
    def no_of_dofs_unconstrained(self):
        """
        Returns the number of dofs of the system that shall be constrained

        Returns
        -------
        n: int
            number of dofs of the system that shall be constrained
        """
        return self._no_of_dofs_unconstrained

    @no_of_dofs_unconstrained.setter
    def no_of_dofs_unconstrained(self, val):
        """

        Parameters
        ----------
        val: int
            number of dofs of the system that shall be constrained

        Returns
        -------
        None
        """
        self._no_of_dofs_unconstrained = val
        self.update()

    @property
    def dimension(self):
        """
        Returns the dimension of the system after constraints have been applied

        Returns
        -------
        dim: int
            dimension of the system after constraints are applied
        """
        raise NotImplementedError('dimension was not implemented in subclass')

    def set_options(self, **kwargs):
        """
        Sets options for the specific formulation
        """
        return

    def update(self):
        """
        Function that is called by observers if state has changed

        Returns
        -------
        None
        """
        return

    def recover(self, x, dx, ddx, t):
        """

        Parameters
        ----------
        x : ndarray
            final ode variables
        dx : ndarray
            1st time derivative of final ode variables
        ddx : ndarray
            2nd time derivative of final ode variables

        Returns
        -------
        u : ndarray
            displacements
        du : ndarray
            velocities
        ddu : ndarray
            accelerations
        t : float
            time
        """
        return self.u(x, t), self.du(x, dx, t), self.ddu(x, dx, ddx, t)

    def u(self, x, t):
        """
        Recovers the displacements of the unconstrained system

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
        raise NotImplementedError('u recovery is not implemented')

    def du(self, x, dx, t):
        """
        Recovers the velocities of the unconstrained system

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
        raise NotImplementedError('du recovery is not implemented')

    def ddu(self, x, dx, ddx, t):
        """
        Recovers the accelerations of the unconstrained system

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
        raise NotImplementedError('ddu recovery is not implemented')

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
        raise NotImplementedError('lagrange multiplier recovery is not implemented')

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
        """
        raise NotImplementedError('M is not implemented')

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

        """
        raise NotImplementedError('F is not implemented')

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
        """
        raise NotImplementedError('K is not implemented')

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
        """
        raise NotImplementedError('D is not implemented')

