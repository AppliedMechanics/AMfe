#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d hexa20 element.
"""

__all__ = [
    'Hexa20'
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


class Hexa20(Element):
    """
    20-node brick element.

    .. code::

              v
        3----------2            3----10----2           3----13----2
        |\     ^   |\           |\         |\          |\         |\
        | \    |   | \          | 19       | 18        |15    24  | 14
        |  \   |   |  \        11  \       9  \        9  \ 20    11 \
        |   7------+---6        |   7----14+---6       |   7----19+---6
        |   |  +-- |-- | -> u   |   |      |   |       |22 |  26  | 23|
        0---+---\--1   |        0---+-8----1   |       0---+-8----1   |
        \  |    \  \  |         \  15      \  13       \ 17    25 \  18
         \ |     \  \ |         16 |        17|        10 |  21    12|
          \|      w  \|           \|         \|          \|         \|
           4----------5            4----12----5           4----16----5


    """
    name = 'Hexa20'

    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)

        self.K = np.zeros((60,60))
        self.f = np.zeros(60)
        self.M = np.zeros((60,60))
        self.S = np.zeros((20,6))
        self.E = np.zeros((20,6))

        a = np.sqrt(3/5)
        wa = 5/9
        w0 = 8/9
        self.gauss_points = ((-a, -a, -a, wa*wa*wa),
                             ( 0, -a, -a, w0*wa*wa),
                             ( a, -a, -a, wa*wa*wa),
                             (-a,  0, -a, wa*w0*wa),
                             ( 0,  0, -a, w0*w0*wa),
                             ( a,  0, -a, wa*w0*wa),
                             (-a,  a, -a, wa*wa*wa),
                             ( 0,  a, -a, w0*wa*wa),
                             ( a,  a, -a, wa*wa*wa),
                             (-a, -a,  0, wa*wa*w0),
                             ( 0, -a,  0, w0*wa*w0),
                             ( a, -a,  0, wa*wa*w0),
                             (-a,  0,  0, wa*w0*w0),
                             ( 0,  0,  0, w0*w0*w0),
                             ( a,  0,  0, wa*w0*w0),
                             (-a,  a,  0, wa*wa*w0),
                             ( 0,  a,  0, w0*wa*w0),
                             ( a,  a,  0, wa*wa*w0),
                             (-a, -a,  a, wa*wa*wa),
                             ( 0, -a,  a, w0*wa*wa),
                             ( a, -a,  a, wa*wa*wa),
                             (-a,  0,  a, wa*w0*wa),
                             ( 0,  0,  a, w0*w0*wa),
                             ( a,  0,  a, wa*w0*wa),
                             (-a,  a,  a, wa*wa*wa),
                             ( 0,  a,  a, w0*wa*wa),
                             ( a,  a,  a, wa*wa*wa),)

        b = 13*np.sqrt(15)/36 + 17/12
        c = (4 + np.sqrt(15))/9
        d = (1 + np.sqrt(15))/36
        e = (3 + np.sqrt(15))/27
        f = 1/9
        g = (1 - np.sqrt(15))/36
        h = -2/27
        i = (3 - np.sqrt(15))/27
        j = -13*np.sqrt(15)/36 + 17/12
        k = (-4 + np.sqrt(15))/9
        l = (3 + np.sqrt(15))/18
        m = np.sqrt(15)/6 + 2/3
        n = 3/18
        p = (- 3 + np.sqrt(15))/18
        q = (4 - np.sqrt(15))/6

        self.extrapolation_points = np.array([
            [b,-c,d,-c,e,f,d,f,g,-c,e,f,e,h,i,f,i,k,d,f,g,f,i,k,g,k,j],
            [d,-c,b,f,e,-c,g,f,d,f,e,-c,i,h,e,k,i,f,g,f,d,k,i,f,j,k,g],
            [g,f,d,f,e,-c,d,-c,b,k,i,f,i,h,e,f,e,-c,j,k,g,k,i,f,g,f,d],
            [d,f,g,-c,e,f,b,-c,d,f,i,k,e,h,i,-c,e,f,g,k,j,f,i,k,d,f,g],
            [d,f,g,f,i,k,g,k,j,-c,e,f,e,h,i,f,i,k,b,-c,d,-c,e,f,d,f,g],
            [g,f,d,k,i,f,j,k,g,f,e,-c,i,h,e,k,i,f,d,-c,b,f,e,-c,g,f,d],
            [j,k,g,k,i,f,g,f,d,k,i,f,i,h,e,f,e,-c,g,f,d,f,e,-c,d,-c,b],
            [g,k,j,f,i,k,d,f,g,f,i,k,e,h,i,-c,e,f,d,f,g,-c,e,f,b,-c,d],
            [l,m,l,-l,-l,-l,n,-n,n,-l,-l,-l,f,f,f,p,p,p,n,-n,n,p,p,p,-p,q,-p],
            [n,-l,l,-n,-l,m,n,-l,l,p,f,-l,p,f,-l,p,f,-l,-p,p,n,q,p,-n,-p,p,n],
            [n,-n,n,-l,-l,-l,l,m,l,p,p,p,f,f,f,-l,-l,-l,-p,q,-p,p,p,p,n,-n,n],
            [l,-l,n,m,-l,-n,l,-l,n,-l,f,p,-l,f,p,-l,f,p,n,p,-p,-n,p,q,n,p,-p],
            [n,-n,n,p,p,p,-p,q,-p,-l,-l,-l,f,f,f,p,p,p,l,m,l,-l,-l,-l,n,-n,n],
            [-p,p,n,q,p,-n,-p,p,n,p,f,-l,p,f,-l,p,f,-l,n,-l,l,-n,-l,m,n,-l,l],
            [-p,q,-p,p,p,p,n,-n,n,p,p,p,f,f,f,-l,-l,-l,n,-n,n,-l,-l,-l,l,m,l],
            [n,p,-p,-n,p,q,n,p,-p,-l,f,p,-l,f,p,-l,f,p,l,-l,n,m,-l,-n,l,-l,n],
            [l,-l,n,-l,f,p,n,p,-p,m,-l,-n,-l,f,p,-n,p,q,l,-l,n,-l,f,p,n,p,-p],
            [n,-l,l,p,f,-l,-p,p,n,-n,-l,m,p,f,-l,q,p,-n,n,-l,l,p,f,-l,-p,p,n],
            [-p,p,n,p,f,-l,n,-l,l,q,p,-n,p,f,-l,-n,-l,m,-p,p,n,p,f,-l,n,-l,l],
            [n,p,-p,-l,f,p,l,-l,n,-n,p,q,-l,f,p,m,-l,-n,n,p,-p,-l,f,p,l,-l,n]])

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
                ('N', 9, 'ux'), ('N', 9, 'uy'), ('N', 9, 'uz'),
                ('N', 10, 'ux'), ('N', 10, 'uy'), ('N', 10, 'uz'),
                ('N', 11, 'ux'), ('N', 11, 'uy'), ('N', 11, 'uz'),
                ('N', 12, 'ux'), ('N', 12, 'uy'), ('N', 12, 'uz'),
                ('N', 13, 'ux'), ('N', 13, 'uy'), ('N', 13, 'uz'),
                ('N', 14, 'ux'), ('N', 14, 'uy'), ('N', 14, 'uz'),
                ('N', 15, 'ux'), ('N', 15, 'uy'), ('N', 15, 'uz'),
                ('N', 16, 'ux'), ('N', 16, 'uy'), ('N', 16, 'uz'),
                ('N', 17, 'ux'), ('N', 17, 'uy'), ('N', 17, 'uz'),
                ('N', 18, 'ux'), ('N', 18, 'uy'), ('N', 18, 'uz'),
                ('N', 19, 'ux'), ('N', 19, 'uy'), ('N', 19, 'uz'))

    def _compute_tensors(self, X, u, t):
        X_mat = X.reshape(20, 3)
        u_mat = u.reshape(20, 3)

        self.K *= 0
        self.f *= 0
        self.S *= 0
        self.E *= 0

        for n_gauss, (xi, eta, zeta, w) in enumerate(self.gauss_points):

            dN_dxi = 1/8*np.array([
                                [ (eta-1)*(zeta-1)*(eta+2*xi+zeta+1),
                                 (xi-1)*(zeta-1)*(2*eta+xi+zeta+1),
                                 (eta-1)*(xi-1)*(eta+xi+2*zeta+1)],
                                [(eta-1)*(zeta-1)*(-eta+2*xi-zeta-1),
                                 (xi+1)*(zeta-1)*(-2*eta+xi-zeta-1),
                                 (eta-1)*(xi+1)*(-eta+xi-2*zeta-1)],
                                [(eta+1)*(zeta-1)*(-eta-2*xi+zeta+1),
                                 (xi+1)*(zeta-1)*(-2*eta-xi+zeta+1),
                                 (eta+1)*(xi+1)*(-eta-xi+2*zeta+1)],
                                [ (eta+1)*(zeta-1)*(eta-2*xi-zeta-1),
                                 (xi-1)*(zeta-1)*(2*eta-xi-zeta-1),
                                 (eta+1)*(xi-1)*(eta-xi-2*zeta-1)],
                                [(eta-1)*(zeta+1)*(-eta-2*xi+zeta-1),
                                 (xi-1)*(zeta+1)*(-2*eta-xi+zeta-1),
                                 (eta-1)*(xi-1)*(-eta-xi+2*zeta-1)],
                                [ (eta-1)*(zeta+1)*(eta-2*xi-zeta+1),
                                 (xi+1)*(zeta+1)*(2*eta-xi-zeta+1),
                                 (eta-1)*(xi+1)*(eta-xi-2*zeta+1)],
                                [ (eta+1)*(zeta+1)*(eta+2*xi+zeta-1),
                                 (xi+1)*(zeta+1)*(2*eta+xi+zeta-1),
                                 (eta+1)*(xi+1)*(eta+xi+2*zeta-1)],
                                [(eta+1)*(zeta+1)*(-eta+2*xi-zeta+1),
                                 (xi-1)*(zeta+1)*(-2*eta+xi-zeta+1),
                                 (eta+1)*(xi-1)*(-eta+xi-2*zeta+1)],
                                [-4*xi*(eta-1)*(zeta-1), -2*(xi**2-1)*(zeta-1),
                                 -2*(eta-1)*(xi**2-1)],
                                [ 2*(eta**2-1)*(zeta-1), 4*eta*(xi+1)*(zeta-1),
                                 2*(eta**2-1)*(xi+1)],
                                [ 4*xi*(eta+1)*(zeta-1),  2*(xi**2-1)*(zeta-1),
                                 2*(eta+1)*(xi**2-1)],
                                [-2*(eta**2-1)*(zeta-1),-4*eta*(xi-1)*(zeta-1),
                                 -2*(eta**2-1)*(xi-1)],
                                [ 4*xi*(eta-1)*(zeta+1),  2*(xi**2-1)*(zeta+1),
                                 2*(eta-1)*(xi**2-1)],
                                [-2*(eta**2-1)*(zeta+1),-4*eta*(xi+1)*(zeta+1),
                                 -2*(eta**2-1)*(xi+1)],
                                [-4*xi*(eta+1)*(zeta+1), -2*(xi**2-1)*(zeta+1),
                                 -2*(eta+1)*(xi**2-1)],
                                [ 2*(eta**2-1)*(zeta+1), 4*eta*(xi-1)*(zeta+1),
                                 2*(eta**2-1)*(xi-1)],
                                [-2*(eta-1)*(zeta**2-1), -2*(xi-1)*(zeta**2-1),
                                 -4*zeta*(eta-1)*(xi-1)],
                                [ 2*(eta-1)*(zeta**2-1),  2*(xi+1)*(zeta**2-1),
                                 4*zeta*(eta-1)*(xi+1)],
                                [-2*(eta+1)*(zeta**2-1), -2*(xi+1)*(zeta**2-1),
                                 -4*zeta*(eta+1)*(xi+1)],
                                [ 2*(eta+1)*(zeta**2-1),  2*(xi-1)*(zeta**2-1),
                                 4*zeta*(eta+1)*(xi-1)]])

            dX_dxi = X_mat.T @ dN_dxi
            dxi_dX = np.linalg.inv(dX_dxi)
            det = np.linalg.det(dX_dxi)
            B0_tilde = dN_dxi @ dxi_dX
            H = u_mat.T @ B0_tilde
            F = H + np.eye(3)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det
            K_geo = scatter_matrix(K_geo_small, 3)
            K_mat = B0.T @ C_SE @ B0 * det

            self.K += (K_geo + K_mat) * w
            self.f += B0.T @ S_v * det * w

            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], S[0,2],
                                            S[1,1], S[1,2], S[2,2]]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], E[0,2],
                                            E[1,1], E[1,2], E[2,2]]])
        return

    def _m_int(self, X, u, t=0):
        X_mat = X.reshape(20, 3)

        self.M *= 0
        rho = self.material.rho

        for n_gauss, (xi, eta, zeta, w) in enumerate(self.gauss_points):
            N = 1/8*np.array([[  (eta-1)*(xi-1)*(zeta-1)*(eta+xi+zeta+2)],
                              [ -(eta-1)*(xi+1)*(zeta-1)*(eta-xi+zeta+2)],
                              [ -(eta+1)*(xi+1)*(zeta-1)*(eta+xi-zeta-2)],
                              [-(eta+1)*(xi-1)*(zeta-1)*(-eta+xi+zeta+2)],
                              [ -(eta-1)*(xi-1)*(zeta+1)*(eta+xi-zeta+2)],
                              [  (eta-1)*(xi+1)*(zeta+1)*(eta-xi-zeta+2)],
                              [  (eta+1)*(xi+1)*(zeta+1)*(eta+xi+zeta-2)],
                              [ -(eta+1)*(xi-1)*(zeta+1)*(eta-xi+zeta-2)],
                              [            -2*(eta-1)*(xi**2-1)*(zeta-1)],
                              [             2*(eta**2-1)*(xi+1)*(zeta-1)],
                              [             2*(eta+1)*(xi**2-1)*(zeta-1)],
                              [            -2*(eta**2-1)*(xi-1)*(zeta-1)],
                              [             2*(eta-1)*(xi**2-1)*(zeta+1)],
                              [            -2*(eta**2-1)*(xi+1)*(zeta+1)],
                              [            -2*(eta+1)*(xi**2-1)*(zeta+1)],
                              [             2*(eta**2-1)*(xi-1)*(zeta+1)],
                              [            -2*(eta-1)*(xi-1)*(zeta**2-1)],
                              [             2*(eta-1)*(xi+1)*(zeta**2-1)],
                              [            -2*(eta+1)*(xi+1)*(zeta**2-1)],
                              [             2*(eta+1)*(xi-1)*(zeta**2-1)]])

            dN_dxi = 1/8*np.array([
                                [ (eta-1)*(zeta-1)*(eta+2*xi+zeta+1),
                                 (xi-1)*(zeta-1)*(2*eta+xi+zeta+1),
                                 (eta-1)*(xi-1)*(eta+xi+2*zeta+1)],
                                [(eta-1)*(zeta-1)*(-eta+2*xi-zeta-1),
                                 (xi+1)*(zeta-1)*(-2*eta+xi-zeta-1),
                                 (eta-1)*(xi+1)*(-eta+xi-2*zeta-1)],
                                [(eta+1)*(zeta-1)*(-eta-2*xi+zeta+1),
                                 (xi+1)*(zeta-1)*(-2*eta-xi+zeta+1),
                                 (eta+1)*(xi+1)*(-eta-xi+2*zeta+1)],
                                [ (eta+1)*(zeta-1)*(eta-2*xi-zeta-1),
                                 (xi-1)*(zeta-1)*(2*eta-xi-zeta-1),
                                 (eta+1)*(xi-1)*(eta-xi-2*zeta-1)],
                                [(eta-1)*(zeta+1)*(-eta-2*xi+zeta-1),
                                 (xi-1)*(zeta+1)*(-2*eta-xi+zeta-1),
                                 (eta-1)*(xi-1)*(-eta-xi+2*zeta-1)],
                                [ (eta-1)*(zeta+1)*(eta-2*xi-zeta+1),
                                 (xi+1)*(zeta+1)*(2*eta-xi-zeta+1),
                                 (eta-1)*(xi+1)*(eta-xi-2*zeta+1)],
                                [ (eta+1)*(zeta+1)*(eta+2*xi+zeta-1),
                                 (xi+1)*(zeta+1)*(2*eta+xi+zeta-1),
                                 (eta+1)*(xi+1)*(eta+xi+2*zeta-1)],
                                [(eta+1)*(zeta+1)*(-eta+2*xi-zeta+1),
                                 (xi-1)*(zeta+1)*(-2*eta+xi-zeta+1),
                                 (eta+1)*(xi-1)*(-eta+xi-2*zeta+1)],
                                [-4*xi*(eta-1)*(zeta-1), -2*(xi**2-1)*(zeta-1),
                                 -2*(eta-1)*(xi**2-1)],
                                [ 2*(eta**2-1)*(zeta-1), 4*eta*(xi+1)*(zeta-1),
                                 2*(eta**2-1)*(xi+1)],
                                [ 4*xi*(eta+1)*(zeta-1),  2*(xi**2-1)*(zeta-1),
                                 2*(eta+1)*(xi**2-1)],
                                [-2*(eta**2-1)*(zeta-1),-4*eta*(xi-1)*(zeta-1),
                                 -2*(eta**2-1)*(xi-1)],
                                [ 4*xi*(eta-1)*(zeta+1),  2*(xi**2-1)*(zeta+1),
                                 2*(eta-1)*(xi**2-1)],
                                [-2*(eta**2-1)*(zeta+1),-4*eta*(xi+1)*(zeta+1),
                                 -2*(eta**2-1)*(xi+1)],
                                [-4*xi*(eta+1)*(zeta+1), -2*(xi**2-1)*(zeta+1),
                                 -2*(eta+1)*(xi**2-1)],
                                [ 2*(eta**2-1)*(zeta+1), 4*eta*(xi-1)*(zeta+1),
                                 2*(eta**2-1)*(xi-1)],
                                [-2*(eta-1)*(zeta**2-1), -2*(xi-1)*(zeta**2-1),
                                 -4*zeta*(eta-1)*(xi-1)],
                                [ 2*(eta-1)*(zeta**2-1),  2*(xi+1)*(zeta**2-1),
                                 4*zeta*(eta-1)*(xi+1)],
                                [-2*(eta+1)*(zeta**2-1), -2*(xi+1)*(zeta**2-1),
                                 -4*zeta*(eta+1)*(xi+1)],
                                [ 2*(eta+1)*(zeta**2-1),  2*(xi-1)*(zeta**2-1),
                                 4*zeta*(eta+1)*(xi-1)]])

            dX_dxi = X_mat.T @ dN_dxi
            det = np.linalg.det(dX_dxi)

            M_small = N @ N.T * det * rho * w
            self.M += scatter_matrix(M_small, 3)

        return self.M


# overloading routines with fortran routines
if use_fortran:
    def compute_hexa20_tensors(self, X, u, t):
        """
        Wrapping function for fortran function call.
        """
        self.K, self.f, self.S, self.E = amfe.f90_element.hexa20_k_f_s_e(X, u, self.material.S_Sv_and_C)

    def compute_hexa20_mass(self, X, u, t=0):
        """
        Wrapping function for fortran function call.
        """
        self.M = amfe.f90_element.hexa20_m(X, self.material.rho)
        return self.M

    Hexa20._compute_tensors_python = Hexa20._compute_tensors
    Hexa20._m_int_python = Hexa20._m_int
    Hexa20._compute_tensors = compute_hexa20_tensors
    Hexa20._m_int = compute_hexa20_mass
