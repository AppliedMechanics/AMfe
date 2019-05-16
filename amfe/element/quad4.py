#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d quad4 element.
"""

__all__ = [
    'Quad4'
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


class Quad4(Element):
    """
    Quadrilateral 2D element with bilinear shape functions.

    .. code::

                ^ eta
        4       |       3
          o_____|_____o
          |     |     |
          |     |     |
        --------+---------->
          |     |     |   xi
          |     |     |
          o_____|_____o
        1       |       2

    """
    name = 'Quad4'

    def __init__(self, *args, **kwargs):
        """
        Definition of material properties and thickness as they are 2D-Elements.
        """
        super().__init__(*args, **kwargs)
        self.K = np.zeros((8,8))
        self.f = np.zeros(8)
        self.M_small = np.zeros((4,4))
        self.M = np.zeros((8,8))
        self.S = np.zeros((4,6))
        self.E = np.zeros((4,6))

        # Gauss-Point-Handling:
        g1 = 1/np.sqrt(3)

        # Tupel for enumerator (xi, eta, weight)
        self.gauss_points = ((-g1, -g1, 1.),
                             ( g1, -g1, 1.),
                             ( g1,  g1, 1.),
                             (-g1,  g1, 1.))

        self.extrapolation_points = np.array([
            [1+np.sqrt(3)/2, -1/2, 1-np.sqrt(3)/2, -1/2],
            [-1/2, 1+np.sqrt(3)/2, -1/2, 1-np.sqrt(3)/2],
            [1-np.sqrt(3)/2, -1/2, 1+np.sqrt(3)/2, -1/2],
            [-1/2, 1-np.sqrt(3)/2, -1/2, 1+np.sqrt(3)/2]]).T

    @staticmethod
    def fields():
        return ('ux', 'uy')

    def dofs(self):
        return (('N', 0, 'ux'),
                ('N', 0, 'uy'),
                ('N', 1, 'ux'),
                ('N', 1, 'uy'),
                ('N', 2, 'ux'),
                ('N', 2, 'uy'),
                ('N', 3, 'ux'),
                ('N', 3, 'uy'))

    def _compute_tensors(self, X, u, t):
        """
        Compute the tensors.
        """
        X_mat = X.reshape(-1, 2)
        u_e = u.reshape(-1, 2)
        t = self.material.thickness

        # Empty former values because they are properties and a new calculation
        # for another element or displacement than before is called
        self.K *= 0
        self.f *= 0
        self.S *= 0
        self.E *= 0

        for n_gauss, (xi, eta, w) in enumerate(self.gauss_points):

            dN_dxi = np.array([ [ eta/4 - 1/4,  xi/4 - 1/4],
                                [-eta/4 + 1/4, -xi/4 - 1/4],
                                [ eta/4 + 1/4,  xi/4 + 1/4],
                                [-eta/4 - 1/4, -xi/4 + 1/4]])

            dX_dxi = X_mat.T @ dN_dxi
            det = dX_dxi[0,0]*dX_dxi[1,1] - dX_dxi[1,0]*dX_dxi[0,1]
            dxi_dX = 1/det * np.array([[ dX_dxi[1,1], -dX_dxi[0,1]],
                                       [-dX_dxi[1,0],  dX_dxi[0,0]]])

            B0_tilde = dN_dxi @ dxi_dX
            H = u_e.T @ B0_tilde
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C_2d(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det*t
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T @ C_SE @ B0 *det*t
            self.K += (K_geo + K_mat)*w
            self.f += B0.T @ S_v*det*t*w
            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], 0, S[1,1], 0, 0]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], 0, E[1,1], 0, 0]])
        return

    def _m_int(self, X, u, t=0):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X
        self.M_small *= 0
        t = self.material.thickness
        rho = self.material.rho

        for xi, eta, w in self.gauss_points:
            det = 1/8*(- X1*Y2*eta + X1*Y2 + X1*Y3*eta - X1*Y3*xi + X1*Y4*xi
                       - X1*Y4 + X2*Y1*eta - X2*Y1 + X2*Y3*xi + X2*Y3
                       - X2*Y4*eta - X2*Y4*xi - X3*Y1*eta + X3*Y1*xi
                       - X3*Y2*xi - X3*Y2 + X3*Y4*eta + X3*Y4 - X4*Y1*xi
                       + X4*Y1 + X4*Y2*eta + X4*Y2*xi - X4*Y3*eta - X4*Y3)
            N = np.array([  [(-eta + 1)*(-xi + 1)/4],
                            [ (-eta + 1)*(xi + 1)/4],
                            [  (eta + 1)*(xi + 1)/4],
                            [ (eta + 1)*(-xi + 1)/4]])
            self.M_small += N.dot(N.T) * det * rho * t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M
