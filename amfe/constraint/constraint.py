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


__all__ = ['NonholonomicConstraintBase',
           'HolonomicConstraintBase',
           'DirichletConstraint',
           'FixedDistanceConstraint',
           'FixedDistanceToLineConstraint',
           'NodesCollinear2DConstraint',
           'EqualDisplacementConstraint',
           'FixedDistanceToPlaneConstraint',
           'NodesCoplanarConstraint'
           ]


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


class FixedDistanceToLineConstraint(HolonomicConstraintBase):

    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        """
        """
        super().__init__()
        return

    def g(self, X_local, u_local, t):
        """
        Constraint-function for a fixed distance to line constraint.

        This function calculates the residuum of the constraints for a Fixed
        Distance To Line Constraint. The idea is that I will have a lot of
        nodes, forming a Line (not necessarily a straight line), and a point x3.
        This point is then constrained to keep a fixed distance from this line,
        based on linear approximations of the line by its nodes.


        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for 2 points forming a line
            and a third point that shall keep the same distance to this line
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3], [x1 y1 x2 y2 x3 y3] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        t: float
            time

        Returns
        -------
        g: ndarray
            Residual of the holonomic constraint function
        """
        dofs_per_node = len(u_local)//3

        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:2*dofs_per_node]
        X3 = X_local[-dofs_per_node:]

        x = X_local + u_local
        x1 = x[:dofs_per_node]
        x2 = x[dofs_per_node:2*dofs_per_node]
        x3 = x[-dofs_per_node:]

        # The vector of direction of the line is
        line_dir = x2 - x1
        # x3_dir is the vector from x1 to x3, so that we can find the distance
        x3_dir = x3 - x1

        # The norm of the rejection of x3_dir relative to line_dir gives us the
        # distance from x3 to the small line we have
        # rejection current is the perpendicular line from line to x3
        # therefore the norm of rejection_current is the distance of x3 to the line
        rejection_current = x3_dir - ((x3_dir.dot(line_dir))
                                      /(np.linalg.norm(line_dir))**2)*line_dir

        # Calculate the initial rejection vector
        initial_dir = X2 - X1
        X3_dir_initial = X3 - X1

        rejection_initial = X3_dir_initial - ((X3_dir_initial.dot(initial_dir)) /
                                              (np.linalg.norm(initial_dir))**2)*initial_dir

        # the squared current distance must be equal to the squared initial distance
        return np.array([rejection_current.dot(rejection_current) - rejection_initial.dot(rejection_initial)],
                        ndmin=1)

    def B(self, X_local, u_local, t):
        """
        Jacobian of constraint-function w.r.t. displacements u

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for 2 points forming a line
            and a third point that shall keep the same distance to this line
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3], [x1 y1 x2 y2 x3 y3] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for the dofs
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """
        dofs_per_node = len(u_local)//3

        if dofs_per_node > 2:

            x = X_local + u_local
            x1 = x[:dofs_per_node]
            x2 = x[dofs_per_node:2*dofs_per_node]
            x3 = x[-dofs_per_node:]

            r_12 = x2 - x1
            r_13 = x3 - x1

            a1 = x1[0]
            b1 = x1[1]
            c1 = x1[2]
            a2 = x2[0]
            b2 = x2[1]
            c2 = x2[2]
            # a3 = x3[0]
            # b3 = x3[1]
            # c3 = x3[2]

            # r_13   = [a3-a1, b3-b1, c3-c1]
            # r_12 = [a2-a1, b2-b1, c2-c1]

            # coef is s in documentation
            coef = ((r_13.dot(r_12))/(r_12.dot(r_12)))

            v = r_13 - coef*r_12

            # dcoefdP1 = ( r_12 @ (-I) ) / (r_12.dot(r_12))
            # + ( r_13 @  ( (-I) / (r_12.dot(r_12))
            # + (2*r_12.T @ r_13)/(r_12.dot(r_12)**2))
            # drejdP1 = -I - (dcoefdP1.T @ r_12) + coef * I
            # bP1 = 2 * v @ drejdP1.T

            # -------------------------------------------------------------------
            # Derivatives of coeff with respect to...
            # ... x1
            dcoefda1 = np.array([-1,0,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([-1,0,0])/(r_12.dot(r_12)) \
                       +r_12*(2*(a2-a1)/(r_12.dot(r_12)**2)))

            # ... y1
            dcoefdb1 = np.array([0, -1, 0]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, -1, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... z1
            dcoefdc1 = np.array([0, 0, -1]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, 0, -1]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (c2 - c1) / (r_12.dot(r_12) ** 2)))

            # ... x2
            dcoefda2 = r_13.dot(np.array([1, 0, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (a2 - a1) / (r_12.dot(r_12) ** 2)))

            # ... y2
            dcoefdb2 = r_13.dot(np.array([0, 1, 0]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... z2
            dcoefdc2 = r_13.dot(np.array([0, 0, 1]) / (r_12.dot(r_12)) \
                                  + r_12 * (2 * (c2 - c1) / (r_12.dot(r_12) ** 2)))

            # ... x3
            dcoefda3 = np.array([1, 0, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... y3
            dcoefdb3 = np.array([0, 1, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... z3
            dcoefdc3 = np.array([0, 0, 1]).dot(r_12) / (r_12.dot(r_12))

            # END of derivatives of coeff
            # All formulas checked by Meyer
            # ----------------------------------------------------------------

            # ----------------------------------------------------------------
            # Comment by Meyer: THIS SECTION IS PROBABLY WRONG!
            #
            drejda1 = np.array([-1, 0, 0]) - dcoefda1*r_12 \
                      + np.array([coef, 0, 0])

            drejdb1 = np.array([0, -1, 0]) - dcoefdb1*r_12 \
                      + np.array([0, coef, 0])


            drejdc1 = np.array([0, 0, -1]) - dcoefdc1*r_12 \
                      + np.array([0, 0, coef])

            drejda2 = - dcoefda2*r_12 - np.array([coef, 0, 0])

            drejdb2 = - dcoefdb2*r_12 - np.array([0, coef, 0])

            drejdc2 = - dcoefdc2*r_12 - np.array([0, 0, coef])

            drejda3 = np.array([1,0,0]) - dcoefda3*r_12

            drejdb3 = np.array([0,1,0]) - dcoefdb3*r_12

            drejdc3 = np.array([0,0,1]) - dcoefdc3*r_12

            bx1 = np.array([
                            2*v.dot(drejda1),
                            2*v.dot(drejdb1),
                            2*v.dot(drejdc1)
                            ])
            bx2 = np.array([
                            2*v.dot(drejda2),
                            2*v.dot(drejdb2),
                            2*v.dot(drejdc2)
                            ])
            bx3 = np.array([
                            2*v.dot(drejda3),
                            2*v.dot(drejdb3),
                            2*v.dot(drejdc3)
                            ])

            b = np.concatenate((bx1,bx2,bx3))

        else:

            x = X_local + u_local
            x1 = x[:dofs_per_node]
            x2 = x[dofs_per_node:2*dofs_per_node]
            x3 = x[-dofs_per_node:]

            r_12 = x2 - x1
            r_13 = x3 - x1

            a1 = x1[0]
            b1 = x1[1]
            a2 = x2[0]
            b2 = x2[1]

            # coef is s in documentation
            coef = ((r_13.dot(r_12))/(r_12.dot(r_12)))

            v = r_13 - coef*r_12

            # -------------------------------------------------------------------
            # Derivatives of coeff with respect to...
            # ... x1
            dcoefda1 = np.array([-1,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([-1,0])/(r_12.dot(r_12)) \
                       +r_12*(2*(a2-a1)/(r_12.dot(r_12)**2)))

            # ... y1
            dcoefdb1 = np.array([0, -1]).dot(r_12) / (r_12.dot(r_12)) \
                       + r_13.dot(np.array([0, -1]) / (r_12.dot(r_12)) \
                       + r_12 * (2 * (b2 - b1) / (r_12.dot(r_12) ** 2)))

            # ... x2
            dcoefda2 = r_13.dot(np.array([1, 0]) / (r_12.dot(r_12)) \
                       + r_12 * (2 * (a2 - a1) / (r_12.dot(r_12) ** 2)))

            # ... y2
            dcoefdb2 = np.array([0,0]).dot(r_12)/(r_12.dot(r_12)) \
                       + r_13.dot(np.array([0,1])/(r_12.dot(r_12)) \
                       +r_12*(2*(b2-b1)/(r_12.dot(r_12)**2)))

            # ... x3
            dcoefda3 = np.array([1, 0]).dot(r_12) / (r_12.dot(r_12))

            # ... y3
            dcoefdb3 = np.array([0, 1]).dot(r_12) / (r_12.dot(r_12))

            # END of derivatives of coeff
            # All formulas checked by Meyer
            # -----------------------------------------------------------------------

            # Comment by Meyer: CAUTION: The following formulas seem to be wrong!
            # The formulas in the thesis of Gruber seem to be correct but are not the same as here
            drejda1 = np.array([-1, 0]) - dcoefda1*r_12 + np.array([coef, 0])

            drejdb1 = np.array([0, -1]) - dcoefdb1*r_12 + np.array([0, coef])

            drejda2 = - dcoefda2*r_12 - np.array([coef, 0])

            drejdb2 = - dcoefdb2*r_12 - np.array([0, coef])

            drejda3 = np.array([1,0]) - dcoefda3*r_12

            drejdb3 = np.array([0,1]) - dcoefdb3*r_12

            bx1 = np.array([
                            2*v.dot(drejda1),
                            2*v.dot(drejdb1)
                            ])
            bx2 = np.array([
                            2*v.dot(drejda2),
                            2*v.dot(drejdb2)
                            ])
            bx3 = np.array([
                            2*v.dot(drejda3),
                            2*v.dot(drejdb3)
                            ])

            b = np.concatenate((bx1,bx2,bx3))

        return b

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
        raise NotImplementedError('The a entity has not been implemented for the fixed distance to line'
                                  'constraint yet')


class NodesCollinear2DConstraint(HolonomicConstraintBase):
    """
    Class to define collinearity (three points on a line).

    This function works with two given coordinates of the nodes and makes
    them collinear. If you want a 3D effect, you have to make two constraints,
    one for (X and Y) and the other for (X and Z or Y and Z).
    This is made in this manner because each constraint can only remove one
    degree of freedom of the system, not two, at a time.

    Caution: Only three points are allowed to be passed to the functions
    Otherwise the results will be wrong."""

    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        return

    def g(self, X_local, u_local, t):
        """
        Constraint-function for a 2d collinear constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for three points just concatenated but only for 2 dimensions
            e.g. x and y [x1 y1 x2 y2 x3 y3] or y and z [y1 z1 y2 z2 y3 z3]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time

        Returns
        -------
        g: ndarray
            Residual of the holonomic constraint function
        """
        x = X_local + u_local
        x1 = x[:2]
        x2 = x[2:4]
        x3 = x[-2:]

        # Three points are collinear if and only if the determinant of the matrix
        # A here is zero.
        A = np.hstack((np.vstack((x1, x2, x3)), np.ones((3, 1))))

        return np.array([np.linalg.det(A)], ndmin=1)  # **2

    def B(self, X_local, u_local, t):
        """
        Jacobian of constraint-function w.r.t. displacements u

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for three points just concatenated but only for 2 dimensions
            e.g. x and y [x1 y1 x2 y2 x3 y3] or y and z [y1 z1 y2 z2 y3 z3]
        u_local: numpy.array
            current displacements for three points just concatenated (c.f. X_local)
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """
        dofs_per_node = len(u_local) // 3

        x = X_local + u_local
        x1 = x[:dofs_per_node]
        x2 = x[dofs_per_node:2 * dofs_per_node]
        x3 = x[-dofs_per_node:]
        #        A = np.hstack((np.vstack((x1,x2,x3)), np.ones((3,1))))
        #        det_A = np.linalg.det(A)
        a1 = x1[0]
        b1 = x1[1]
        a2 = x2[0]
        b2 = x2[1]
        a3 = x3[0]
        b3 = x3[1]

        b = np.array([
            b2 - b3,  # 2*(b2 - b3)*(det_A),
            -a2 + a3,  # -2*(a2 - a3)*(det_A), #
            -b1 + b3,  # -2*(b1 - b3)*(det_A), #
            a1 - a3,  # 2*(a1 - a3)*(det_A), #
            b1 - b2,  # 2*(b1 - b2)*(det_A), #
            -a1 + a2,  # -2*(a1 - a2)*(det_A), #
        ])
        return b

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
        raise NotImplementedError('The a entity has not been implemented for the fixed distance to line'
                                  'constraint yet')


class EqualDisplacementConstraint(HolonomicConstraintBase):
    """
    Class to define a fixed distance between two nodes.
    """
    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        super().__init__()
        return

    def g(self, X_local, u_local, t):
        """
        Return residual of constraint function for an equal displacement constraint between two nodes.

        Parameters
        ----------
        X_local: numpy.array
            not needed for this constraint
        u_local: numpy.array
            current displacements for both dofs
        t: float
            time

        Returns
        -------
        g : numpy.array
            Residual of constraint function
        """
        return np.array([u_local[1] - u_local[0]], dtype=float, ndmin=1)

    def B(self, X_local, u_local, t):
        """
        Return derivative for an equal displacement constraint

        Parameters
        ----------
        X_local: numpy.array
            not needed for this constraint
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """
        return np.array([-1.0, 1.0], dtype=float)

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
        return np.array([0.0], ndmin=1)


class FixedDistanceToPlaneConstraint(HolonomicConstraintBase):
    """
    Class to define a fixed distance to plane constraint where three nodes define the plane
    and one node has a fixed distance to it.
    """
    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        super().__init__()

        raise NotImplementedError('Theano is not compatible anymore, the constraint must be reimplemented')

    def g(self, X_local, u_local, t):
        """
        Return residual of constraint function for a fixed distance to plane constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            The first three points define the plane, the fourth point shall have fixed distance
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for all four points just concatenated (c.f. X_local)
        t: float
            time

        Returns
        -------
        g : numpy.array
            Residual of constraint function
        """

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2 * 3]
        x3 = x[2 * 3:3 * 3]
        x4 = x[-3:]

        plane_vector_1 = x2 - x1
        plane_vector_2 = x3 - x1
        plane_normal = np.cross(plane_vector_1, plane_vector_2)
        plane_normal = plane_normal / (np.linalg.norm(plane_normal))
        x4_vector = x4 - x1

        X1 = X_local[:3]
        X2 = X_local[3:2 * 3]
        X3 = X_local[2 * 3:3 * 3]
        X4 = X_local[-3:]

        initial_vector_1 = X2 - X1
        initial_vector_2 = X3 - X1
        ini_plane_normal = np.cross(initial_vector_1, initial_vector_2)
        ini_plane_normal = ini_plane_normal / (np.linalg.norm(ini_plane_normal))
        X4_vector = X4 - X1

        return np.array([np.dot(x4_vector, plane_normal) -
                         np.dot(X4_vector, ini_plane_normal) ], dtype=float, ndmin=1)

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
        raise NotImplementedError('Theano is not compatible to AMfe anymore. This constraint is not implemented'
                                  'for now')

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
        raise NotImplementedError('The a entity has not been implemented for the fixed distance to'
                                  'plane constraint yet.')


class NodesCoplanarConstraint(HolonomicConstraintBase):
    """
    Class to define a nodes coplanar constraint between four nodes.
    """
    NO_OF_CONSTRAINTS = 1

    def __init__(self):
        super().__init__()
        return

    def g(self, X_local, u_local, t):
        """
        Return residual of constraint function for a nodes coplanar constraint between four nodes.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for all four nodes just concatenated (c.f. X_local)
        t: float
            time

        Returns
        -------
        g : numpy.array
            Residual of constraint function
        """

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2 * 3]
        x3 = x[2 * 3:3 * 3]
        x4 = x[-3:]

        # x1, x2, x3 and x4 are coplanar if the determinant of A is 0
        A = np.hstack((x1.T - x4.T, x2.T - x4.T, x3.T - x4.T)).reshape((3, 3))
        return np.array([np.linalg.det(A)], ndmin=1)

    def B(self, X_local, u_local, t):
        """
        Return derivative of c_equation with respect to u for a Fixed Distance constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for four points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4]
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        B: ndarray
            Partial derivative of constraint function g w.r.t. displacements u
        """

        x = X_local + u_local
        x1 = x[:3]
        x2 = x[3:2*3]
        x3 = x[2*3:3*3]
        x4 = x[-3:]
        a1, b1, c1 = x1
        a2, b2, c2 = x2
        a3, b3, c3 = x3
        a4, b4, c4 = x4

        b = np.array([
                        b2*c3 - b2*c4 - b3*c2 + b3*c4 + b4*c2 - b4*c3,  # a1
                       -a2*c3 + a2*c4 + a3*c2 - a3*c4 - a4*c2 + a4*c3,  # b1
                        a2*b3 - a2*b4 - a3*b2 + a3*b4 + a4*b2 - a4*b3,  # c1
                       -b1*c3 + b1*c4 + b3*c1 - b3*c4 - b4*c1 + b4*c3,  # a2
                        a1*c3 - a1*c4 - a3*c1 + a3*c4 + a4*c1 - a4*c3,  # b2
                       -a1*b3 + a1*b4 + a3*b1 - a3*b4 - a4*b1 + a4*b3,  # c2
                        b1*c2 - b1*c4 - b2*c1 + b2*c4 + b4*c1 - b4*c2,  # a3
                       -a1*c2 + a1*c4 + a2*c1 - a2*c4 - a4*c1 + a4*c2,  # b3
                        a1*b2 - a1*b4 - a2*b1 + a2*b4 + a4*b1 - a4*b2,  # c3
                       -b1*c2 + b1*c3 + b2*c1 - b2*c3 - b3*c1 + b3*c2,  # a4
                        a1*c2 - a1*c3 - a2*c1 + a2*c3 + a3*c1 - a3*c2,  # b4
                       -a1*b2 + a1*b3 + a2*b1 - a2*b3 - a3*b1 + a3*b2,  # c4
                      ])
        # Checked that this is the same as in the thesis by Gruber. But only first two rows have been checked by Meyer
        return b

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
        raise NotImplementedError('The a entity has not been implemented for the coplanar constraint yet.')
