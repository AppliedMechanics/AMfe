#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d tri6 element.
"""

__all__ = [
    'Tri6'
]

import numpy as np

from .element import Element
from .tri3 import compute_B_matrix, scatter_matrix

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Tri6(Element):
    """
    6 node second order triangle
    Triangle Element with 6 dofs; 3 dofs at the corner, 3 dofs in the
    intermediate point of every face.
    """
    plane_stress = True
    name = 'Tri6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.K = np.zeros((12,12))
        self.f = np.zeros(12)
        self.M_small = np.zeros((6,6))
        self.M = np.zeros((12,12))
        self.S = np.zeros((6,6))
        self.E = np.zeros((6,6))

        self.gauss_points2 = ((2/3, 1/6, 1/6, 1/3),
                              (1/6, 2/3, 1/6, 1/3),
                              (1/6, 1/6, 2/3, 1/3))

        self.extrapolation_points = np.array([
            [5/3, -1/3, -1/3, 2/3, -1/3, 2/3],
            [-1/3, 5/3, -1/3, 2/3, 2/3, -1/3],
            [-1/3, -1/3, 5/3, -1/3, 2/3, 2/3]]).T

        # self.gauss_points3 = ((1/3, 1/3, 1/3, -27/48),
        #                      (0.6, 0.2, 0.2, 25/48),
        #                      (0.2, 0.6, 0.2, 25/48),
        #                      (0.2, 0.2, 0.6, 25/48))
        #
        # alpha1 = 0.0597158717
        # beta1 = 0.4701420641 # 1/(np.sqrt(15)-6)
        # w1 = 0.1323941527
        #
        # alpha2 = 0.7974269853 #
        # beta2 = 0.1012865073 # 1/(np.sqrt(15)+6)
        # w2 = 0.1259391805
        #
        # self.gauss_points5 = ((1/3, 1/3, 1/3, 0.225),
        #                      (alpha1, beta1, beta1, w1),
        #                      (beta1, alpha1, beta1, w1),
        #                      (beta1, beta1, alpha1, w1),
        #                      (alpha2, beta2, beta2, w2),
        #                      (beta2, alpha2, beta2, w2),
        #                      (beta2, beta2, alpha2, w2))

        self.gauss_points = self.gauss_points2

    @staticmethod
    def fields():
        return ('ux', 'uy')

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'),
                ('N', 1, 'ux'), ('N', 1, 'uy'),
                ('N', 2, 'ux'), ('N', 2, 'uy'),
                ('N', 3, 'ux'), ('N', 3, 'uy'),
                ('N', 4, 'ux'), ('N', 4, 'uy'),
                ('N', 5, 'ux'), ('N', 5, 'uy'))

    def _compute_tensors(self, X, u, t):
        """
        Tensor computation the same way as in the Tri3 element
        """
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        u_mat = u.reshape((-1,2))
        # X_mat = X.reshape((-1,2))
        d = self.material.thickness

        self.K *= 0
        self.f *= 0
        self.E *= 0
        self.S *= 0
        for n_gauss, (L1, L2, L3, w) in enumerate(self.gauss_points):

            dN_dL = np.array([  [4*L1 - 1,        0,        0],
                                [       0, 4*L2 - 1,        0],
                                [       0,        0, 4*L3 - 1],
                                [    4*L2,     4*L1,        0],
                                [       0,     4*L3,     4*L2],
                                [    4*L3,        0,     4*L1]])

            # the entries in the jacobian dX_dL
            Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
            Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
            Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
            Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

            det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2


            dL_dX = 1/det*np.array([[ Jy2 - Jy3, -Jx2 + Jx3],
                                    [-Jy1 + Jy3,  Jx1 - Jx3],
                                    [ Jy1 - Jy2, -Jx1 + Jx2]])

            B0_tilde = dN_dL @ dL_dX

            H = u_mat.T @ B0_tilde
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C_2d(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det / 2 * d
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T @ C_SE @ B0 * det / 2 * d
            self.K += (K_geo + K_mat) * w
            self.f += B0.T @ S_v * det / 2*d*w
            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], 0, S[1,1], 0, 0]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], 0, E[1,1], 0, 0]])
        return

    def _m_int(self, X, u, t=0):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        t = self.material.thickness
        rho = self.material.rho
        self.M_small *= 0
        for L1, L2, L3, w in self.gauss_points:

            # the entries in the jacobian dX_dL
            Jx1 = 4*L2*X4 + 4*L3*X6 + X1*(4*L1 - 1)
            Jx2 = 4*L1*X4 + 4*L3*X5 + X2*(4*L2 - 1)
            Jx3 = 4*L1*X6 + 4*L2*X5 + X3*(4*L3 - 1)
            Jy1 = 4*L2*Y4 + 4*L3*Y6 + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y4 + 4*L3*Y5 + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y6 + 4*L2*Y5 + Y3*(4*L3 - 1)

            det = Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2

            N = np.array([  [L1*(2*L1 - 1)],
                            [L2*(2*L2 - 1)],
                            [L3*(2*L3 - 1)],
                            [      4*L1*L2],
                            [      4*L2*L3],
                            [      4*L1*L3]])

            self.M_small += N.dot(N.T) * det/2 * rho * t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M


# overloading routines with fortran routines
if use_fortran:
    def compute_tri6_tensors(self, X, u, t):
        """
        Wrapping function for fortran function call.
        """
        self.K, self.f, self.S, self.E = amfe.f90_element.tri6_k_f_s_e(X, u, self.material.thickness,
                                                                       self.material.S_Sv_and_C_2d)

    def compute_tri6_mass(self, X, u, t=0):
        """
        Wrapping function for fortran function call.
        """
        self.M = amfe.f90_element.tri6_m(X, self.material.rho, self.material.thickness)
        return self.M

    Tri6._compute_tensors_python = Tri6._compute_tensors
    Tri6._m_int_python = Tri6._m_int
    Tri6._compute_tensors = compute_tri6_tensors
    Tri6._m_int = compute_tri6_mass
