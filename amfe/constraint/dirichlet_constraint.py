#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#


import numpy as np
from scipy.sparse import eye as speye

from .structural_constraint import StructuralConstraint


class DirichletConstraint(StructuralConstraint):
    """
    DirichletConstraint

    Class to define a Dirichlet Constraint on several dofs


    """
    def __init__(self, dofs, U = lambda t: 0, dU = lambda t: 0, ddU = lambda t: 0):
        super().__init__()
        self._U = U
        self._dU = dU
        self._ddU = ddU
        self._no_of_dofs = len(dofs)
        self._slave_dofs = np.arange(self._no_of_dofs)

    def c(self, X_local, u_local, du_local, ddu_local, t):
        return u_local - self._U(t)

    def b(self, X_local, u_local, du_local, ddu_local, t):
        return speye(len(u_local))

    def u_slave(self, X_local, u_local, du_local, ddu_local, t):
        return np.ones(self._no_of_dofs) * self._U(t)

    def du_slave(self, X_local, u_local, du_local, ddu_local, t):
        return np.ones(self._no_of_dofs) * self._U(t)

    def ddu_slave(self, X_local, u_local, du_local, ddu_local, t):
        return np.ones(self._no_of_dofs) * self._U(t)

    def slave_dofs(self, dofs):
        return dofs[self._slave_dofs]
