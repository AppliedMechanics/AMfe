#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d tri3 element.
"""

__all__ = [
    'Tri3'
]

import numpy as np

from .element import Element
from .tools import compute_B_matrix, scatter_matrix

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Tri3(Element):
    """
    Element class for a plane triangle element in Total Lagrangian formulation.
    The displacements are given in x- and y-coordinates;

    Notes
    -----
    The Element assumes constant strain and stress over the whole element.
    Thus the approximation quality is very moderate.


    References
    ----------
    Basis for this implementation is the Monograph of Ted Belytschko:
    Nonlinear Finite Elements for Continua and Structures.
    pp. 201 and 207.

    """
    plane_stress = True
    name = 'Tri3'

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        material : class HyperelasticMaterial
            Material class representing the material
        """
        super().__init__(*args, **kwargs)
        self.K = np.zeros((6,6))
        self.f = np.zeros(6)
        self.S = np.zeros((3,6))
        self.E = np.zeros((3,6))

    @staticmethod
    def fields():
        return ('ux', 'uy')

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'),
                ('N', 1, 'ux'), ('N', 1, 'uy'),
                ('N', 2, 'ux'), ('N', 2, 'uy'))

    def _compute_tensors(self, X, u, t):
        """
        Compute the tensors B0_tilde, B0, F, E and S at the Gauss Points.

        Variables
        ---------
            B0_tilde: ndarray
                Die Ableitung der Ansatzfunktionen nach den x- und
                y-Koordinaten (2x3-Matrix). In den Zeilein stehen die
                Koordinatenrichtungen, in den Spalten die Ansatzfunktionen
            B0: ndarray
                The mapping matrix of delta E = B0 * u^e
            F: ndarray
                Deformation gradient (2x2-Matrix)
            E: ndarray
                Der Green-Lagrange strain tensor (2x2-Matrix)
            S: ndarray
                2. Piola-Kirchhoff stress tensor, using Kirchhoff material
                (2x2-Matrix)
        """
        d = self.material.thickness
        X1, Y1, X2, Y2, X3, Y3 = X
        u_mat = u.reshape((-1,2))
        det = (X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2)
        A0       = 0.5*det
        dN_dX = 1/det*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]])
        H = u_mat.T @ dN_dX
        F = H + np.eye(2)
        E = 1/2*(H + H.T + H.T @ H)
        S, S_v, C_SE = self.material.S_Sv_and_C_2d(E)
        B0 = compute_B_matrix(dN_dX, F)
        K_geo_small = dN_dX @ S @ dN_dX.T * det/2 * d
        K_geo = scatter_matrix(K_geo_small, 2)
        K_mat = B0.T @ C_SE @ B0 * det/2 * d
        self.K = (K_geo + K_mat)
        self.f = B0.T @ S_v * det/2 * d
        self.E = np.ones((3,1)) @ np.array([[E[0,0], E[0,1], 0, E[1,1], 0, 0]])
        self.S = np.ones((3,1)) @ np.array([[S[0,0], S[0,1], 0, S[1,1], 0, 0]])
        return

    def _m_int(self, X, u, t=0):
        """
        Compute the mass matrix.

        Parameters
        ----------

        X : ndarray
            Position of the nodal coordinates in undeformed configuration
            using voigt notation X = (X1, Y1, X2, Y2, X3, Y3)
        u : ndarray
            Displacement of the element using same voigt notation as for X
        t : float
            Time

        Returns
        -------

        M : ndarray
            Mass matrix of the given element
        """
        t = self.material.thickness
        rho = self.material.rho
        X1, Y1, X2, Y2, X3, Y3 = X
        self.A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.M = np.array([[2, 0, 1, 0, 1, 0],
                           [0, 2, 0, 1, 0, 1],
                           [1, 0, 2, 0, 1, 0],
                           [0, 1, 0, 2, 0, 1],
                           [1, 0, 1, 0, 2, 0],
                           [0, 1, 0, 1, 0, 2]])*self.A0/12*t*rho
        return self.M


# overloading routines with fortran routines
if use_fortran:
    def compute_tri3_tensors(self, X, u, t):
        """
        Wrapping function for fortran function call.
        """
        self.K, self.f, self.S, self.E = amfe.f90_element.tri3_k_f_s_e(X, u, self.material.thickness,
                                                                       self.material.S_Sv_and_C_2d)

    Tri3._compute_tensors_python = Tri3._compute_tensors
    Tri3._compute_tensors = compute_tri3_tensors
