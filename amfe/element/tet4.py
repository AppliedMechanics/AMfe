#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d tet4 element.
"""

__all__ = [
    'Tet4'
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


class Tet4(Element):
    """
    Tetraeder-Element with 4 nodes
    """
    name = 'Tet4'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.zeros((12,12))
        self.f = np.zeros(12)
        self.M = np.zeros((12,12))
        self.S = np.zeros((4,6))
        self.E = np.zeros((4,6))

    @staticmethod
    def fields():
        return ('ux', 'uy', 'uz')

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'uz'),
                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'uz'),
                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'uz'),
                ('N', 3, 'ux'), ('N', 3, 'uy'), ('N', 3, 'uz'))

    def _compute_tensors(self, X, u, t):
        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4 = X
        u_mat = u.reshape(-1, 3)
        X_mat = X.reshape(-1, 3)

        det = -X1*Y2*Z3 + X1*Y2*Z4 + X1*Y3*Z2 - X1*Y3*Z4 - X1*Y4*Z2 + X1*Y4*Z3 \
             + X2*Y1*Z3 - X2*Y1*Z4 - X2*Y3*Z1 + X2*Y3*Z4 + X2*Y4*Z1 - X2*Y4*Z3 \
             - X3*Y1*Z2 + X3*Y1*Z4 + X3*Y2*Z1 - X3*Y2*Z4 - X3*Y4*Z1 + X3*Y4*Z2 \
             + X4*Y1*Z2 - X4*Y1*Z3 - X4*Y2*Z1 + X4*Y2*Z3 + X4*Y3*Z1 - X4*Y3*Z2

        B0_tilde = 1/det*np.array([
            [-Y2*Z3 + Y2*Z4 + Y3*Z2 - Y3*Z4 - Y4*Z2 + Y4*Z3,
              X2*Z3 - X2*Z4 - X3*Z2 + X3*Z4 + X4*Z2 - X4*Z3,
             -X2*Y3 + X2*Y4 + X3*Y2 - X3*Y4 - X4*Y2 + X4*Y3],
            [ Y1*Z3 - Y1*Z4 - Y3*Z1 + Y3*Z4 + Y4*Z1 - Y4*Z3,
             -X1*Z3 + X1*Z4 + X3*Z1 - X3*Z4 - X4*Z1 + X4*Z3,
              X1*Y3 - X1*Y4 - X3*Y1 + X3*Y4 + X4*Y1 - X4*Y3],
            [-Y1*Z2 + Y1*Z4 + Y2*Z1 - Y2*Z4 - Y4*Z1 + Y4*Z2,
              X1*Z2 - X1*Z4 - X2*Z1 + X2*Z4 + X4*Z1 - X4*Z2,
             -X1*Y2 + X1*Y4 + X2*Y1 - X2*Y4 - X4*Y1 + X4*Y2],
            [ Y1*Z2 - Y1*Z3 - Y2*Z1 + Y2*Z3 + Y3*Z1 - Y3*Z2,
             -X1*Z2 + X1*Z3 + X2*Z1 - X2*Z3 - X3*Z1 + X3*Z2,
              X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2]])

        H = u_mat.T @ B0_tilde
        F = H + np.eye(3)
        E = 1/2*(H + H.T + H.T @ H)
        S, S_v, C_SE = self.material.S_Sv_and_C(E)
        B0 = compute_B_matrix(B0_tilde, F)
        K_geo_small = B0_tilde @ S @ B0_tilde.T * det/6
        K_geo = scatter_matrix(K_geo_small, 3)
        K_mat = B0.T @ C_SE @ B0 * det/6
        self.K = K_geo + K_mat
        self.f = B0.T @ S_v*det/6
        self.E = np.ones((4,1)) @ np.array([[E[0,0], E[0,1], E[0,2],
                                             E[1,1], E[1,2], E[2,2]]])
        self.S = np.ones((4,1)) @ np.array([[S[0,0], S[0,1], S[0,2],
                                             S[1,1], S[1,2], S[2,2]]])

    def _m_int(self, X, u, t=0):
        """
        Mass matrix using CAS-System
        """
        rho = self.material.rho

        X1, Y1, Z1, X2, Y2, Z2, X3, Y3, Z3, X4, Y4, Z4 = X
        det =   X1*Y2*Z3 - X1*Y2*Z4 - X1*Y3*Z2 + X1*Y3*Z4 + X1*Y4*Z2 - X1*Y4*Z3 \
              - X2*Y1*Z3 + X2*Y1*Z4 + X2*Y3*Z1 - X2*Y3*Z4 - X2*Y4*Z1 + X2*Y4*Z3 \
              + X3*Y1*Z2 - X3*Y1*Z4 - X3*Y2*Z1 + X3*Y2*Z4 + X3*Y4*Z1 - X3*Y4*Z2 \
              - X4*Y1*Z2 + X4*Y1*Z3 + X4*Y2*Z1 - X4*Y2*Z3 - X4*Y3*Z1 + X4*Y3*Z2

        # same thing as above - it's not clear yet how the node numbering is done.
        det *= -1
        V = det/6
        self.M = V / 20 * rho * np.array([
            [ 2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  1.],
            [ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.],
            [ 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  0.],
            [ 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.]])
        return self.M


# overloading routines with fortran routines
if use_fortran:
    def compute_tet4_tensors(self, X, u, t):
        """
        Wrapping function for fortran function call.
        """
        self.K, self.f, self.S, self.E = amfe.f90_element.tet4_k_f_s_e(X, u, self.material.S_Sv_and_C)

    Tet4._compute_tensors_python = Tet4._compute_tensors
    Tet4._compute_tensors = compute_tet4_tensors
