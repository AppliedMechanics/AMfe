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
    M(u, \dot{u}, t) \ddot{u} + h(u, \dot{u}, t) + B^T \lambda = p(u, \dot{u}, t) \\
    g_{holo}(u, t) &= 0 \\
    g_{nonholo}(u, \dot{u}, t) &= B_{nonholo}(u, t) \cdot \dot{u} + b(u, t) = 0

where :math:`h` are internal forces and :math:`p` are external forces
And returns this type of system:

.. math::
    M(x, dx, t) \ddot{x} + f_{int}(x, \dot{x}, t) = f_{ext}(x, \dot{x}, t)

Here x is the full state vector of the system. Depending on the constraint formulation this can e.g. be a mixture of
displacements :math:`u` and Lagrange Multipliers :math:`\lambda`

Furthermore, it provides the following entities:

- Linear Damping matrix
.. math::
    D = \frac{\mathrm{d}(f_{int} - f_{ext})}{\mathrm{d} \dot{x}}

- Linear Stiffness matrix
.. math::
    K = \frac{\mathrm{d}(f_{int} - f_{ext})}{\mathrm{d} x}

It also can recover the u, du, ddu of the unconstrained system from the system states x of the constrained system
"""


class ConstraintFormulationBase:
    r"""
    Applies constraints to general system

    .. math::
        M(u, \dot{u}, t) \ddot{u} + h(u, \dot{u}, t) + B^T \lambda = p(u, \dot{u}, t)

    where :math:`h` are internal forces, :math:`p` are external forces and :math:`\lambda` Lagrange
    Multipliers

    Attributes
    ----------
    _no_of_dofs_unconstrained: int
        number of dofs of the unconstrained system, number of entries in vector u
    _M_func: function
        function with signature M(u, du, t) returning the mass matrix
    _h_func: function
        function with signature h(u, du, t) returning the nonlinear internal forces
    _p_func: function
        function with signature p(u, du, t) returning the nonlinear external forces
    _B_func: function
        function with signature B(u, t) returning the linear map of the constraint function on velocity level
        mapping the velocities to the residual of the constraint equation (B*dq + b) = 0
    _jac_h_u: function
        Jacobian of the _h_func w.r.t. displacements u
    _jac_h_du: function
        Jacobian of the _h_func w.r.t. velocities du
    _jac_p_u: function
        Jacobian of the _p_func w.r.t. displacements u
    _jac_p_du: function
        Jacobian of the _p_func w.r.t. velocities du
    _g_func: function
        Function with signature g(u, t) returning the residual of the holonomic constraints on displacement level
    _b_func: function
        Function returning the nonhomogeneous part of the constraint equation on velocity level
        :math:`B(u, t) \dot{u} + b(u, t) = 0` (last term b)
    _a_func: function
        Function returning the nonhomogeneous part of the constraint equation on acceleration level
        :math:`B(u, t) \ddot{u} + a(u, t) = 0` (last term a)

    """
    def __init__(self, no_of_dofs_unconstrained, M_func, h_func, B_func, p_func=None,
                 jac_h_u=None, jac_h_du=None, jac_p_u=None, jac_p_du=None,
                 g_func=None, b_func=None, a_func=None):
        self._no_of_dofs_unconstrained = no_of_dofs_unconstrained
        self._M_func = M_func
        self._h_func = h_func
        self._B_func = B_func
        self._p_func = p_func
        self._jac_h_u = jac_h_u
        self._jac_h_du = jac_h_du
        self._jac_p_u = jac_p_u
        self._jac_p_du = jac_p_du
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
        t : float
            time

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

        """
        raise NotImplementedError('f_int is not implemented')

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

        """
        raise NotImplementedError('f_ext is not implemented')

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
