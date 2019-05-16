#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import numpy as np
from amfe.component.constants import ELEPROTOTYPEHELPERLIST


class NeumannBase:
    """
    Base class for Neumann Conditions
    """
    ELEMENTFACTORY = {element[0]: element[2] for element in ELEPROTOTYPEHELPERLIST}

    def __init__(self, *args, **kwargs):
        self._boundary_element = None
    
    def f_ext(self, X, u, t):
        """
        Returns the local external force vector of Neumann element

        Parameters
        ----------
        X : numpy.array
            node coordinates in reference domain as 1D array (reshape -1)
        u : numpy.array
            local displacements of element
        t : float
            time t

        Returns
        -------
        f_ext : numpy.array
            local external force vector
        """
        # no minus sign as force will be on the right hand side of eqn.
        return self._f_proj(self._boundary_element.f_mat(X, u)) * self._amp(u, t)

    def k_and_f_ext(self, X, u, t):
        """
        Returns the local external force vector of Neumann element

        Parameters
        ----------
        X : numpy.array
            node coordinates in reference domain as 1D array (reshape -1)
        u : numpy.array
            local displacements of element
        t : float
            time t

        Returns
        -------
        K_ext : numpy.array
            local stiffness matrix, i.e. the Jocobian of the local external force vector w.r.t local displacement u
        f_ext : numpy.array
            local external force vector
        """
        f_ext = self.f_ext(X, u, t)
        ndof = len(f_ext)
        K = np.zeros((ndof, ndof))
        return K, f_ext

    def _amp(self, u, t):
        raise NotImplementedError('_amp is not implemented')

    def _f_proj(self, f_mat):
        raise NotImplementedError('_f_proj is not implemented')

    def set_element(self, ele_shape):
        self._boundary_element = self.ELEMENTFACTORY[ele_shape]()

    def dofs(self):
        return self._boundary_element.dofs()

    def fields(self):
        return self._boundary_element.fields()
