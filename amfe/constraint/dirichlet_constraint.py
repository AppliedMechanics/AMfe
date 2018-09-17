#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Dirichlet constraint.
"""

import numpy as np
from scipy.sparse import eye as eyes

from .structural_constraint import StructuralConstraint


class DirichletConstraint(StructuralConstraint):
    """
    Class to define a Dirichlet constraints on several dofs.
    """

    def __init__(self, dofs, U=(lambda t: 0.), dU=(lambda t: 0.), ddU=(lambda t: 0.)):
        super().__init__()
        self._U = U
        self._dU = dU
        self._ddU = ddU
        self._no_of_dofs = len(dofs)
        self._slave_dofs = np.arange(self._no_of_dofs)
        return

    def c(self, X_local, u_local, du_local, ddu_local, t):
        """
        Return residual of c_equation for a fixed dirichlet constraint.

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
        res : numpy.array
            Residual of c(q) as vector
        """

        return u_local - self._U(t)

    def b(self, X_local, u_local, du_local, ddu_local, t):
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

        return eyes(len(u_local))

    def u_slave(self, X_local, u_local, du_local, ddu_local, t):
        """
        Return the displacements of the slave_dofs (needed for elimination).

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
        u_slave : numpy.array
            displacements of the slave_dofs
        """

        return np.ones(self._no_of_dofs) * self._U(t)

    def du_slave(self, X_local, u_local, du_local, ddu_local, t):
        """
        Return the velocities of the slave_dofs (needed for elimination).

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
        du_slave : numpy.array
            velocities of the slave_dofs
        """

        return np.ones(self._no_of_dofs) * self._dU(t)

    def ddu_slave(self, X_local, u_local, du_local, ddu_local, t):
        """
        Return the accelerations of the slave_dofs (needed for elimination).

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
        ddu_slave : numpy.array
            accelerations of the slave_dofs
        """

        return np.ones(self._no_of_dofs) * self._ddU(t)

    def slave_dofs(self, dofs):
        """
        Parameters
        ----------
        dofs : ndarray
            ndarray of integers that describe the global dof indices that must be passed to the constraint (dofs arg)

        Returns
        -------
        slave_dofs : ndarray
            returns ndarray of integers that are the global indices of the dofs that are slaves (eliminated if 'elim'
            strategy is chosen)
        """

        return dofs[self._slave_dofs]
