#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
r"""
This module describes different nonholonomic and holonomic constraints and provides a base classes for them

This module follows the following conventions:

Holonomic Constraint:

.. math::
    g(u, t) = 0

The differential of the constraint (Pfaffian Form) is:

.. math::
    \mathrm{d}g = B(u, t) \mathrm{d}u + b(u, t) \mathrm{d}t = 0

The above is the starting point for nonholonomic constraints but can simply be derived for holonomic constraints
via the derivatives of the constraint function g(u, t)
The total time derivative of the constraint describes the constraints on velocity level

.. math::
    \frac{\mathrm{d}g}{\mathrm{d}t} = B(u, t) \cdot \dot{u}  +  b(u, t) = 0


The second time derivative of the constraint function describes the constraints on acceleration level:

.. math::
    \frac{\mathrm{d}^2 g}{\mathrm{d} t^2} &= B(u, t) \cdot \ddot{u} + \frac{\mathrm{d}B(u, t)}{\mathrm{d} t} \cdot \
    \dot{u} + \frac{\mathrm{d}b(u, t)}{\mathrm{d} t} \\
    &= B(u, t) \cdot \ddot{u} + \frac{\partial B(u, t)}{\partial u} \cdot \dot{u}^2 + \
    \frac{\partial B(u, t)}{\partial t} \cdot \dot{u} + \frac{\partial b(u, t)}{\partial u} \dot{u} + \
    \frac{\partial b(u, t)}{\partial t} \\
    &= B(u, t) \cdot \ddot{u} + a(u, du, t) \\
    &= 0

"""

import numpy as np

from ..linalg.norms import vector_norm


class NonholonomicConstraintBase:

    NO_OF_CONSTRAINTS = 0

    def __init__(self):
        return

    def after_assignment(self, dofids):
        """
        Method that is called after assignment in Constraint Manager

        No changes are needed in this function for most cases.
        But it provides the opportunity to change the state of the constraint after it has been assigned to the
        constraint manager

        Parameters
        ----------
        dofids: list or numpy.array
            list or numpy.array containing the dofids of the dofs which are passed to the constraint

        Returns
        -------
        None
        """
        return

    def B(self, X, u, t):
        """
        Linear map from velocities to constraint function (c.f. module description)

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            current time

        Returns
        -------
        B: ndarray
            Linear map from velocities to constraint function
        """
        raise NotImplementedError('The B has not been implemented for this constraint')

    def b(self, X, u, t):
        """
        Part of the nonholonomic constraint equation that is independent of the velocities

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time

        Returns
        -------
        b: ndarray
            Velocity independent part of the constraint function on velocity level
        """
        raise NotImplementedError('The partial time derivative of the constraint function is not implemented for this'
                                  'constraint')

    def a(self, X, u, du, t):
        r"""
        It computes the inhomogeneous part on acceleration level

        .. math::
            \frac{\partial B(u, t)}{\partial u} \cdot \dot{u}^2 + \
            \frac{\partial B(u, t)}{\partial t} \cdot \dot{u} + \frac{\partial b(u, t)}{\partial u} \dot{u} + \
            \frac{\partial b(u, t)}{\partial t} \\

        Parameters
        ----------
        X: numpy.array
            Empty numpy array because dirichlet constraints do not need information about node coordinates
        u: numpy.array
            current displacements for the dofs that shall be constrained
        du: numpy.array
            current velocities for the dofs that schall be constrained
        t: float
            time

        Returns
        -------
        a: numpy.array
            The above described entity (inhomogeneous part of acceleration level constraint)

        """
        raise NotImplementedError('The total time derivative of the partial time derivative is not implemented for this'
                                  'constraint')


class HolonomicConstraintBase(NonholonomicConstraintBase):

    NO_OF_CONSTRAINTS = 0

    def __init__(self):
        super().__init__()
        return

    def B(self, X, u, t):
        """
        Partial Derivative of holonomic constraint function g w.r.t. displacements u

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """
        raise NotImplementedError('The B has not been implemented for this constraint')

    def b(self, X, u, t):
        """
        Partial Derivative of holonomic constraint function g w.r.t. time t

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time

        Returns
        -------
        b: ndarray
            Partial derivative of the constraint function g w.r.t. time t
        """
        raise NotImplementedError('The partial time derivative of the constraint function is not implemented for this'
                                  'constraint')

    def a(self, X, u, du, t):
        r"""
        It computes the inhomogeneous part on acceleration level

        .. math::
            \frac{\partial B(u, t)}{\partial u} \cdot \dot{u}^2 + \
            \frac{\partial B(u, t)}{\partial t} \cdot \dot{u} + \frac{\partial b(u, t)}{\partial u} \dot{u} + \
            \frac{\partial b(u, t)}{\partial t} \\

        Parameters
        ----------
        X: numpy.array
            Empty numpy array because dirichlet constraints do not need information about node coordinates
        u: numpy.array
            current displacements for the dofs that shall be constrained
        du: numpy.array
            current velocities for the dofs that schall be constrained
        t: float
            time

        Returns
        -------
        a: numpy.array
            The above described entity (inhomogeneous part of acceleration level constraint)

        """
        raise NotImplementedError('The total time derivative of the partial time derivative is not implemented for this'
                                  'constraint')

    def g(self, X, u, t):
        """
        Residual of holonomic constraint-function g
        
        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time
        
        Returns
        -------
        g: ndarray
            residual of holonomic constraint function
        """
        raise NotImplementedError('The constraint function has not been implemented for this constraint')


class DirichletConstraint(HolonomicConstraintBase):
    """
    Class to define a Dirichlet constraints on several dofs.

    Attributes
    ----------
    _U: function
        contains the function of enforced displacements
    _dU: function
        contains the function of enforced velocities (time derivative of _U)
    _ddU: function
        contains the function of enforced accelerations (time derivative of _dU)
    """

    NO_OF_CONSTRAINTS = 1

    def __init__(self, U=(lambda t: 0.), dU=(lambda t: 0.), ddU=(lambda t: 0.)):
        """
        A Dirichlet Constraint can be initialized with predefined displacements

        Parameters
        ----------
        U: function
            function with signature float U: f(float: t)
            describing enforced displacements
        dU: function
            function with signature float dU: f(float: t)
            describing enforced velocities
        ddU: function
            function with signature float ddU: f(float: t)
            describing enforced accelerations
        """
        super().__init__()
        self._U = U
        self._dU = dU
        self._ddU = ddU
        return

    def after_assignment(self, dofids):
        """
        In this case the number of constraints is set after assignment because this is unknown before

        Parameters
        ----------
        dofids: list or numpy.array
            list or numpy.array containing the dof-IDs of the dofs that are constrained by this Dirichlet Constraint

        Returns
        -------
        None
        """
        return

    def g(self, X_local, u_local, t):
        """
        Constraint-function for a fixed dirichlet constraint.

        Parameters
        ----------
        X_local: numpy.array
            Empty numpy array because Dirichlet Constraints to not need node coordinates
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        g: ndarray
            Residual of the holonomic constraint function
        """
        return np.array(u_local - self._U(t), dtype=float)
    
    def B(self, X_local, u_local, t):
        """
        Jacobian of constraint-function w.r.t. displacements u

        Parameters
        ----------
        X_local: numpy.array
            Empty numpy array because dirichlet constraints do not need information about node coordinates
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """
        return np.array([1], dtype=float)

    def b(self, X, u, t):
        """
        Partial Derivative of holonomic constraint function g w.r.t. time t

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time

        Returns
        -------
        b: ndarray
            Partial derivative of the constraint function g w.r.t. time t
        """
        return np.array([-self._dU(t)], ndmin=1)

    def a(self, X, u, du, t):
        r"""
        It computes the inhomogeneous part on acceleration level

        .. math::
            \frac{\partial B(u, t)}{\partial u} \cdot \dot{u}^2 + \
            \frac{\partial B(u, t)}{\partial t} \cdot \dot{u} + \frac{\partial b(u, t)}{\partial u} \dot{u} + \
            \frac{\partial b(u, t)}{\partial t} \\

        Parameters
        ----------
        X: numpy.array
            Empty numpy array because dirichlet constraints do not need information about node coordinates
        u: numpy.array
            current displacements for the dofs that shall be constrained
        du: numpy.array
            current velocities for the dofs that schall be constrained
        t: float
            time

        Returns
        -------
        a: numpy.array
            The above described entity (inhomogeneous part of acceleration level constraint)

        """
        return np.array([-self._ddU(t)], ndmin=1)


class FixedDistanceConstraint(HolonomicConstraintBase):
    """
    Class to define a fixed distance between two nodes.
    """
    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        super().__init__()
        return

    def g(self, X_local, u_local, t):
        """
        Return residual of constraint function for a fixed distance constraint between two nodes.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time

        Returns
        -------
        g : numpy.array
            Residual of constraint function
        """

        dofs_per_node = len(u_local) // 2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = vector_norm(X2 - X1)

        return np.array((vector_norm(x2 - x1) - vector_norm(X2 - X1)) * 10. / scaling, dtype=float, ndmin=1)

    def B(self, X_local, u_local, t):
        """
        Return derivative of c_equation with respect to u for a Fixed Distance constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for degrees of freedom
            e.g. [x1 x2 y3 y4 z5] if x-direction of node 1 and 2, y-direction node 3 and 4 and z-direction of node 5 is
            constrained
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """

        dofs_per_node = len(u_local) // 2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = vector_norm(X2 - X1)
        l_current = vector_norm(x2 - x1)

        return 10.0 * np.concatenate((-(x2 - x1) / (l_current * scaling), (x2 - x1) / (l_current * scaling)))

    def b(self, X, u, t):
        """
        Partial Derivative of holonomic constraint function g w.r.t. time t

        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            local displacements
        t: float
            time

        Returns
        -------
        b: ndarray
            Partial derivative of the constraint function g w.r.t. time t
        """
        return np.array([0.0], dtype=float, ndmin=1)

    def a(self, X, u, du, t):
        r"""
        It computes the inhomogeneous part on acceleration level

        .. math::
            \frac{\partial B(u, t)}{\partial u} \cdot \dot{u}^2 + \
            \frac{\partial B(u, t)}{\partial t} \cdot \dot{u} + \frac{\partial b(u, t)}{\partial u} \dot{u} + \
            \frac{\partial b(u, t)}{\partial t} \\

        Parameters
        ----------
        X: numpy.array
            Empty numpy array because dirichlet constraints do not need information about node coordinates
        u: numpy.array
            current displacements for the dofs that shall be constrained
        du: numpy.array
            current velocities for the dofs that schall be constrained
        t: float
            time

        Returns
        -------
        a: numpy.array
            The above described entity (inhomogeneous part of acceleration level constraint)

        """
        # a consists of four terms:
        # 1. partial derivative dB/du * du^2
        no_of_dofs = len(u)
        delta = 1e-8*vector_norm(u) + 1e-8
        dBdu = np.zeros((1, no_of_dofs))
        uplus = u.copy()
        uminus = u.copy()
        for i in range(no_of_dofs):
            uplus[:] = u
            uplus[i] = uplus[i] + delta
            uminus[:] = u
            uminus[i] = uminus[i] - delta
            jplus = self.B(X, uplus, t)
            jminus = self.B(X, uminus, t)
            dBdu[:] += (jplus - jminus)/(2*delta)*du[i]

        # 2. partial derivative dB/dt
        delta = 1e-8
        dBdt = (self.B(X, u, t + delta) - self.B(X, u, t - delta)) / (2 * delta)

        # 3. partial derivative db/du is zero
        # 4. partial derivative db/dt is zero

        return dBdu.dot(du) + dBdt.dot(du)
