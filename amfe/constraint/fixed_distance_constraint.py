#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from .structural_constraint import StructuralConstraint


class FixedDistanceConstraint(StructuralConstraint):
    """
    Fixed Distance Constraint

    Class to define a fixed distance between two nodes
    """
    def __init__(self, dofs):
        super().__init__()
        self._no_of_dofs = len(dofs)
        self._slave_dofs = [0]

    def c(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Returns residual of c_equation for a fixed distance constraint between two nodes

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

        scaling = np.linalg.norm(X2 - X1)

        return 10 * (np.sqrt(np.sum(((x2 - x1) ** 2))) -
                     np.sqrt(np.sum(((X2 - X1) ** 2)))) / scaling

    def b(self, X_local, u_local, du_local, ddu_local, t=0):
        """
        Returns the b vector (for assembly in B matrix) for a fixed distance constraint between two nodes

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
        b : numpy.array
            b vector for assembly in B matrix [bx1 by1 bz1 bx2 by2 bz2]

        """
        dofs_per_node = len(u_local) // 2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        scaling = np.linalg.norm(X2 - X1)
        l_current = np.sqrt(np.sum(((x2 - x1) ** 2)))

        return 10 * np.concatenate((-(x2 - x1) / (l_current * scaling),
                                    (x2 - x1) / (l_current * scaling)))

    def u_slave(self, X_local, u_local, du_local, ddu_local, t):
        dofs_per_node = len(u_local) // 2
        u1 = u_local[:dofs_per_node]
        u2 = u_local[dofs_per_node:]
        X1 = X_local[:dofs_per_node]
        X2 = X_local[dofs_per_node:]

        x1 = X1 + u1
        x2 = X2 + u2

        # u_slave is u1_x
        return u2[0] + X2[0] - X1[0] - np.sqrt(sum((X2-X1)**2) + sum(x2[1:]-x1[1:]))

    def slave_dofs(self, dofs):
        return dofs[self._slave_dofs]
