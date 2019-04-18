#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d tet10 element.
"""

__all__ = [
    'Tet10'
]

import numpy as np
from numpy import sqrt

from .element import Element
from .tools import compute_B_matrix, scatter_matrix

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Tet10(Element):
    r"""
    Tet10 solid element; Node numbering is done like in ParaView and in [1]_

    The node numbering is as follows:

    .. code::
                 3
               ,/|`\
             ,/  |  `\
           ,8    '.   `7
         ,/       9     `\
       ,/         |       `\
      1--------4--'.--------0
       `\.         |      ,/
          `\.      |    ,6
             `5.   '. ,/
                `\. |/
                   `2

    References
    ----------
    .. [1] Felippa, Carlos: Advanced Finite Element Methods (ASEN 6367),
        Spring 2013. `Online Source`__

    __ http://www.colorado.edu/engineering/CAS/courses.d/AFEM.d/AFEM.Ch10.d/
        AFEM.Ch10.index.html

    """
    name = 'Tet10'

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

        self.K = np.zeros((30,30))
        self.f = np.zeros(30)
        self.M = np.zeros((30,30))
        self.S = np.zeros((10,6))
        self.E = np.zeros((10,6))


        a = (5 - np.sqrt(5)) / 20
        b = (5 + 3*np.sqrt(5)) / 20
        w = 1/4
        gauss_points_4 = ((b,a,a,a,w),
                          (a,b,a,a,w),
                          (a,a,b,a,w),
                          (a,a,a,b,w),)
        self.gauss_points = gauss_points_4

        c1 = 1/4 + 3*sqrt(5)/4 # close corner node
        c2 = -sqrt(5)/4 + 1/4  # far corner node
        m1 = 1/4 + sqrt(5)/4   # close mid-node
        m2 = -sqrt(5)/4 + 1/4  # far mid node

        self.extrapolation_points = np.array(
            [[c1, c2, c2, c2, m1, m2, m1, m1, m2, m2],
             [c2, c1, c2, c2, m1, m1, m2, m2, m1, m2],
             [c2, c2, c1, c2, m2, m1, m1, m2, m2, m1],
             [c2, c2, c2, c1, m2, m2, m2, m1, m1, m1]]).T

    @staticmethod
    def fields():
        return ('ux', 'uy', 'uz')

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'uz'),
                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'uz'),
                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'uz'),
                ('N', 3, 'ux'), ('N', 3, 'uy'), ('N', 3, 'uz'),
                ('N', 4, 'ux'), ('N', 4, 'uy'), ('N', 4, 'uz'),
                ('N', 5, 'ux'), ('N', 5, 'uy'), ('N', 5, 'uz'),
                ('N', 6, 'ux'), ('N', 6, 'uy'), ('N', 6, 'uz'),
                ('N', 7, 'ux'), ('N', 7, 'uy'), ('N', 7, 'uz'),
                ('N', 8, 'ux'), ('N', 8, 'uy'), ('N', 8, 'uz'),
                ('N', 9, 'ux'), ('N', 9, 'uy'), ('N', 9, 'uz'))

    def _compute_tensors(self, X, u, t):

        X1, Y1, Z1, \
        X2, Y2, Z2, \
        X3, Y3, Z3, \
        X4, Y4, Z4, \
        X5, Y5, Z5, \
        X6, Y6, Z6, \
        X7, Y7, Z7, \
        X8, Y8, Z8, \
        X9, Y9, Z9, \
        X10, Y10, Z10 = X

        u_mat = u.reshape((10,3))
        self.K *= 0
        self.f *= 0
        self.S *= 0
        self.E *= 0

        for n_gauss, (L1, L2, L3, L4, w) in enumerate(self.gauss_points):

            Jx1 = 4*L2*X5 + 4*L3*X7 + 4*L4*X8  + X1*(4*L1 - 1)
            Jx2 = 4*L1*X5 + 4*L3*X6 + 4*L4*X9  + X2*(4*L2 - 1)
            Jx3 = 4*L1*X7 + 4*L2*X6 + 4*L4*X10 + X3*(4*L3 - 1)
            Jx4 = 4*L1*X8 + 4*L2*X9 + 4*L3*X10 + X4*(4*L4 - 1)
            Jy1 = 4*L2*Y5 + 4*L3*Y7 + 4*L4*Y8  + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y5 + 4*L3*Y6 + 4*L4*Y9  + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y7 + 4*L2*Y6 + 4*L4*Y10 + Y3*(4*L3 - 1)
            Jy4 = 4*L1*Y8 + 4*L2*Y9 + 4*L3*Y10 + Y4*(4*L4 - 1)
            Jz1 = 4*L2*Z5 + 4*L3*Z7 + 4*L4*Z8  + Z1*(4*L1 - 1)
            Jz2 = 4*L1*Z5 + 4*L3*Z6 + 4*L4*Z9  + Z2*(4*L2 - 1)
            Jz3 = 4*L1*Z7 + 4*L2*Z6 + 4*L4*Z10 + Z3*(4*L3 - 1)
            Jz4 = 4*L1*Z8 + 4*L2*Z9 + 4*L3*Z10 + Z4*(4*L4 - 1)

            det = -Jx1*Jy2*Jz3 + Jx1*Jy2*Jz4 + Jx1*Jy3*Jz2 - Jx1*Jy3*Jz4 \
                 - Jx1*Jy4*Jz2 + Jx1*Jy4*Jz3 + Jx2*Jy1*Jz3 - Jx2*Jy1*Jz4 \
                 - Jx2*Jy3*Jz1 + Jx2*Jy3*Jz4 + Jx2*Jy4*Jz1 - Jx2*Jy4*Jz3 \
                 - Jx3*Jy1*Jz2 + Jx3*Jy1*Jz4 + Jx3*Jy2*Jz1 - Jx3*Jy2*Jz4 \
                 - Jx3*Jy4*Jz1 + Jx3*Jy4*Jz2 + Jx4*Jy1*Jz2 - Jx4*Jy1*Jz3 \
                 - Jx4*Jy2*Jz1 + Jx4*Jy2*Jz3 + Jx4*Jy3*Jz1 - Jx4*Jy3*Jz2

            a1 = -Jy2*Jz3 + Jy2*Jz4 + Jy3*Jz2 - Jy3*Jz4 - Jy4*Jz2 + Jy4*Jz3
            a2 =  Jy1*Jz3 - Jy1*Jz4 - Jy3*Jz1 + Jy3*Jz4 + Jy4*Jz1 - Jy4*Jz3
            a3 = -Jy1*Jz2 + Jy1*Jz4 + Jy2*Jz1 - Jy2*Jz4 - Jy4*Jz1 + Jy4*Jz2
            a4 =  Jy1*Jz2 - Jy1*Jz3 - Jy2*Jz1 + Jy2*Jz3 + Jy3*Jz1 - Jy3*Jz2
            b1 =  Jx2*Jz3 - Jx2*Jz4 - Jx3*Jz2 + Jx3*Jz4 + Jx4*Jz2 - Jx4*Jz3
            b2 = -Jx1*Jz3 + Jx1*Jz4 + Jx3*Jz1 - Jx3*Jz4 - Jx4*Jz1 + Jx4*Jz3
            b3 =  Jx1*Jz2 - Jx1*Jz4 - Jx2*Jz1 + Jx2*Jz4 + Jx4*Jz1 - Jx4*Jz2
            b4 = -Jx1*Jz2 + Jx1*Jz3 + Jx2*Jz1 - Jx2*Jz3 - Jx3*Jz1 + Jx3*Jz2
            c1 = -Jx2*Jy3 + Jx2*Jy4 + Jx3*Jy2 - Jx3*Jy4 - Jx4*Jy2 + Jx4*Jy3
            c2 =  Jx1*Jy3 - Jx1*Jy4 - Jx3*Jy1 + Jx3*Jy4 + Jx4*Jy1 - Jx4*Jy3
            c3 = -Jx1*Jy2 + Jx1*Jy4 + Jx2*Jy1 - Jx2*Jy4 - Jx4*Jy1 + Jx4*Jy2
            c4 =  Jx1*Jy2 - Jx1*Jy3 - Jx2*Jy1 + Jx2*Jy3 + Jx3*Jy1 - Jx3*Jy2

            B0_tilde = 1/det*np.array([
                [    a1*(4*L1 - 1),     b1*(4*L1 - 1),     c1*(4*L1 - 1)],
                [    a2*(4*L2 - 1),     b2*(4*L2 - 1),     c2*(4*L2 - 1)],
                [    a3*(4*L3 - 1),     b3*(4*L3 - 1),     c3*(4*L3 - 1)],
                [    a4*(4*L4 - 1),     b4*(4*L4 - 1),     c4*(4*L4 - 1)],
                [4*L1*a2 + 4*L2*a1, 4*L1*b2 + 4*L2*b1, 4*L1*c2 + 4*L2*c1],
                [4*L2*a3 + 4*L3*a2, 4*L2*b3 + 4*L3*b2, 4*L2*c3 + 4*L3*c2],
                [4*L1*a3 + 4*L3*a1, 4*L1*b3 + 4*L3*b1, 4*L1*c3 + 4*L3*c1],
                [4*L1*a4 + 4*L4*a1, 4*L1*b4 + 4*L4*b1, 4*L1*c4 + 4*L4*c1],
                [4*L2*a4 + 4*L4*a2, 4*L2*b4 + 4*L4*b2, 4*L2*c4 + 4*L4*c2],
                [4*L3*a4 + 4*L4*a3, 4*L3*b4 + 4*L4*b3, 4*L3*c4 + 4*L4*c3]])

            H = u_mat.T @ B0_tilde
            F = H + np.eye(3)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det/6
            K_geo = scatter_matrix(K_geo_small, 3)
            K_mat = B0.T @ C_SE @ B0 * det/6

            self.K += (K_geo + K_mat) * w
            self.f += B0.T @ S_v * det/6 * w

            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], S[0,2],
                                            S[1,1], S[1,2], S[2,2]]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], E[0,2],
                                            E[1,1], E[1,2], E[2,2]]])
        return

    def _m_int(self, X, u, t=0):
        """
        Mass matrix using CAS-System
        """
        X1, Y1, Z1, \
        X2, Y2, Z2, \
        X3, Y3, Z3, \
        X4, Y4, Z4, \
        X5, Y5, Z5, \
        X6, Y6, Z6, \
        X7, Y7, Z7, \
        X8, Y8, Z8, \
        X9, Y9, Z9, \
        X10, Y10, Z10 = X
        X_mat = X.reshape((-1,3))

        self.M *= 0
        rho = self.material.rho

        for L1, L2, L3, L4, w in self.gauss_points:

            Jx1 = 4*L2*X5 + 4*L3*X7 + 4*L4*X8  + X1*(4*L1 - 1)
            Jx2 = 4*L1*X5 + 4*L3*X6 + 4*L4*X9  + X2*(4*L2 - 1)
            Jx3 = 4*L1*X7 + 4*L2*X6 + 4*L4*X10 + X3*(4*L3 - 1)
            Jx4 = 4*L1*X8 + 4*L2*X9 + 4*L3*X10 + X4*(4*L4 - 1)
            Jy1 = 4*L2*Y5 + 4*L3*Y7 + 4*L4*Y8  + Y1*(4*L1 - 1)
            Jy2 = 4*L1*Y5 + 4*L3*Y6 + 4*L4*Y9  + Y2*(4*L2 - 1)
            Jy3 = 4*L1*Y7 + 4*L2*Y6 + 4*L4*Y10 + Y3*(4*L3 - 1)
            Jy4 = 4*L1*Y8 + 4*L2*Y9 + 4*L3*Y10 + Y4*(4*L4 - 1)
            Jz1 = 4*L2*Z5 + 4*L3*Z7 + 4*L4*Z8  + Z1*(4*L1 - 1)
            Jz2 = 4*L1*Z5 + 4*L3*Z6 + 4*L4*Z9  + Z2*(4*L2 - 1)
            Jz3 = 4*L1*Z7 + 4*L2*Z6 + 4*L4*Z10 + Z3*(4*L3 - 1)
            Jz4 = 4*L1*Z8 + 4*L2*Z9 + 4*L3*Z10 + Z4*(4*L4 - 1)

            det = -Jx1*Jy2*Jz3 + Jx1*Jy2*Jz4 + Jx1*Jy3*Jz2 - Jx1*Jy3*Jz4 \
                 - Jx1*Jy4*Jz2 + Jx1*Jy4*Jz3 + Jx2*Jy1*Jz3 - Jx2*Jy1*Jz4 \
                 - Jx2*Jy3*Jz1 + Jx2*Jy3*Jz4 + Jx2*Jy4*Jz1 - Jx2*Jy4*Jz3 \
                 - Jx3*Jy1*Jz2 + Jx3*Jy1*Jz4 + Jx3*Jy2*Jz1 - Jx3*Jy2*Jz4 \
                 - Jx3*Jy4*Jz1 + Jx3*Jy4*Jz2 + Jx4*Jy1*Jz2 - Jx4*Jy1*Jz3 \
                 - Jx4*Jy2*Jz1 + Jx4*Jy2*Jz3 + Jx4*Jy3*Jz1 - Jx4*Jy3*Jz2

            N = np.array([  [L1*(2*L1 - 1)],
                            [L2*(2*L2 - 1)],
                            [L3*(2*L3 - 1)],
                            [L4*(2*L4 - 1)],
                            [      4*L1*L2],
                            [      4*L2*L3],
                            [      4*L1*L3],
                            [      4*L1*L4],
                            [      4*L2*L4],
                            [      4*L3*L4]])

            M_small = N.dot(N.T) * det/6 * rho * w
            self.M += scatter_matrix(M_small, 3)
        return self.M


# overloading routines with fortran routines
if use_fortran:
    def compute_tet10_tensors(self, X, u, t):
        """
        Wrapping function for fortran function call.
        """
        self.K, self.f, self.S, self.E = amfe.f90_element.tet10_k_f_s_e(X, u, self.material.S_Sv_and_C)

    Tet10._compute_tensors_python = Tet10._compute_tensors
    Tet10._compute_tensors = compute_tet10_tensors
