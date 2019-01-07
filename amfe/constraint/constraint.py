#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import abc

import numpy as np
from scipy.sparse import eye as eyes
from scipy.sparse import csr_matrix
from ..linalg.norms import vector_norm

"""
Super-class of all constraints
"""


class ConstraintBase:
    def __init__(self):
        self.no_of_constraints = 0

    def after_assignment(self, dofids):
        """
        Method that is called after assignment in Constraint Manager

        Parameters
        ----------
        dofids: list or numpy.array
            list or numpy.array containing the dofids of the dofs which are passed to the constraint

        Returns
        -------
        None
        """
        pass
    
    def constraint_func(self, X, u, du, ddu, t):
        """
        Residual of constraint-function
        
        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            primary variable of previous iteration
        du: ndarray
            time-derivation of primary variable of previous iteration
        ddu: ndarray
            doubled time-derivation of primary variable of previous iteration
        t: int
            current time-step
        
        Returns
        -------
        g: ndarray
            residual of constraint function
        """
        pass
    
    def jacobian(self, X, u, du, ddu, t, primary_type='u'):
        """
        Jacobian of constraint-function derived for primary variable
        
        Parameters
        ----------
        X: ndarray
            local node coordinates of dofs in reference domain
        u: ndarray
            primary variable of previous iteration
        du: ndarray
            time-derivation of primary variable of previous iteration
        ddu: ndarray
            doubled time-derivation of primary variable of previous iteration
        t: int
            current time-step
        primary_type : str {'u', 'du', 'ddu'}
            describes with respect to which entitity the conatraint function is derived
        
        Returns
        -------
        J: csr_matrix
        """
        if primary_type == 'u':
            J = self._J_u(X, u, du, ddu, t)
        elif primary_type == 'du':
            J = self._J_du(X, u, du, ddu, t)
        elif primary_type == 'ddu':
            J = self._J_ddu(X, u, du, ddu, t)
        else:
            raise NotImplementedError('Jacobian derived for time is not implemented')

        return J

    @abc.abstractmethod
    def _J_u(self, X, u, du, ddu, t):
        raise NotImplementedError('The Jacobian w.r.t u has not been implemented for this constraint')

    @abc.abstractmethod
    def _J_du(self, X, u, du, ddu, t):
        raise NotImplementedError('The Jacobian w.r.t u has not been implemented for this constraint')

    @abc.abstractmethod
    def _J_ddu(self, X, u, du, ddu, t):
        raise NotImplementedError('The Jacobian w.r.t u has not been implemented for this constraint')

"""
Dirichlet constraint.
"""


class DirichletConstraint(ConstraintBase):
    """
    Class to define a Dirichlet constraints on several dofs.
    """
    def __init__(self, U=(lambda t: 0.), dU=(lambda t: 0.), ddU=(lambda t: 0.)):
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
            list or numpy.array containing the dof-IDs of the dofs that are contrained by this Dirichlet Constraint

        Returns
        -------
        None
        """
        self.no_of_constraints = len(dofids)

    def constraint_func(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Constraint-function for a fixed dirichlet constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for degrees of freedom
            e.g. [x1 x2 y3 y4 z5] if x-direction of node 1 and 2, y-direction node 3 and 4 and z-direction of node 5 is
            constrained
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        du_local: numpy.array
            current velocity for the dofs that shall be constrained
        ddu_local: numpy.array
            current acceleration for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        g: ndarray
        """
        return u_local - self._U(t)
    
    def _J_u(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Jacobian of constraint-function derived for primary variable

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for degrees of freedom
            e.g. [x1 x2 y3 y4 z5] if x-direction of node 1 and 2, y-direction node 3 and 4 and z-direction of node 5 is
            constrained
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        du_local: numpy.array
            current velocity for the dofs that shall be constrained
        ddu_local: numpy.array
            current acceleration for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        J: csr_matrix
        """

        return eyes(len(u_local))

    def _J_du(self, X_local, u_local, du_local, ddu_local, t=0):
        no_of_dofs = len(u_local)
        return csr_matrix((no_of_dofs, no_of_dofs))
    
    def _J_ddu(self, X_local, u_local, du_local, ddu_local, t=0):
        no_of_dofs = len(u_local)
        return csr_matrix((no_of_dofs, no_of_dofs))

"""
Fixed distance constraint.
"""


class FixedDistanceConstraint(ConstraintBase):
    """
    Class to define a fixed distance between two nodes.
    """

    def __init__(self):
        super().__init__()
        self.no_of_constraints = 1
        return

    def constraint_func(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Return residual of c_equation for a fixed distance constraint between two nodes.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for both points just concatenated
            e.g. [x1 y1 z1 x2 y2 z2], [x1 y1 x2 y2] for 3D or 2D problems respectively
        u_local: numpy.array
            current displacements for both points just concatenated (c.f. X_local)
        du_local: numpy.array
            current velocity for both points just concatenated (c.f. X_local)
        ddu_local: numpy.array
            current acceleration for both points just concatenated (c.f. X_local)

        t: float
            time, default: 0, in this case not necessary because fixed distance does not
            change over time

        Returns
        -------
        res : numpy.array
            Residual of c(q) as vector
        """

        dofs_per_node = len(u_local) // 2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = vector_norm(X2 - X1)

        return (vector_norm(x2 - x1) - vector_norm(X2 - X1)) * 10. / scaling

    def _J_u(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Return derivative of c_equation with respect to u for a Dirichlet constraint.

        Parameters
        ----------
        X_local: numpy.array
            X-Coords in reference Domain for degrees of freedom
            e.g. [x1 x2 y3 y4 z5] if x-direction of node 1 and 2, y-direction node 3 and 4 and z-direction of node 5 is
            constrained
        u_local: numpy.array
            current displacements for the dofs that shall be constrained
        du_local: numpy.array
            current velocity for the dofs that shall be constrained
        ddu_local: numpy.array
            current acceleration for the dofs that shall be constrained
        t: float
            time

        Returns
        -------
        b : numpy.array
            vector or matrix b with b = dc/du
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

        return csr_matrix(10.* np.concatenate((-(x2 - x1) / (l_current * scaling), (x2 - x1) / (l_current * scaling))))
        #return csr_matrix(np.concatenate((-(x2 - x1) / l_current, (x2 - x1) / l_current)))
        
    def _J_du(self, X_local, u_local, du_local, ddu_local, t=0):
        return csr_matrix((1, len(u_local)))
    
    def _J_ddu(self, X_local, u_local, du_local, ddu_local, t=0):
        return csr_matrix((1, len(u_local)))
