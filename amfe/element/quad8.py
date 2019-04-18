#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d quad8 element.
"""

__all__ = [
    'Quad8'
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


class Quad8(Element):
    """
    Plane Quadrangle with quadratic shape functions and 8 nodes. 4 nodes are
    at every corner, 4 nodes on every face.
    """
    name = 'Quad8'

    def __init__(self, *args, **kwargs):
        """
        Definition of material properties and thickness as they are 2D-Elements.
        """
        super().__init__(*args, **kwargs)
        self.K = np.zeros((16,16))
        self.f = np.zeros(16)
        self.M_small = np.zeros((8,8))
        self.M = np.zeros((16,16))
        self.S = np.zeros((8,6))
        self.E = np.zeros((8,6))

        # Quadrature like ANSYS or ABAQUS:
        g = np.sqrt(3/5)
        w = 5/9
        w0 = 8/9
        self.gauss_points = ((-g, -g,  w*w), ( g, -g,  w*w ), ( g,  g,   w*w),
                             (-g,  g,  w*w), ( 0, -g, w0*w ), ( g,  0,  w*w0),
                             ( 0,  g, w0*w), (-g,  0,  w*w0), ( 0,  0, w0*w0))

        # a little bit dirty but correct. Comes from sympy file.
        self.extrapolation_points = np.array(
        [[ 5*sqrt(15)/18 + 10/9, 5/18, -5*sqrt(15)/18 + 10/9,
            5/18, -5/9 - sqrt(15)/9, -5/9 + sqrt(15)/9,
            -5/9 + sqrt(15)/9, -5/9 - sqrt(15)/9,  4/9],
         [5/18,  5*sqrt(15)/18 + 10/9, 5/18, -5*sqrt(15)/18 + 10/9,
          -5/9 - sqrt(15)/9, -5/9 - sqrt(15)/9, -5/9 + sqrt(15)/9,
          -5/9 + sqrt(15)/9,  4/9],
         [-5*sqrt(15)/18 + 10/9, 5/18, 5*sqrt(15)/18 + 10/9, 5/18,
          -5/9 + sqrt(15)/9, -5/9 - sqrt(15)/9, -5/9 - sqrt(15)/9,
          -5/9 + sqrt(15)/9,  4/9],
         [ 5/18, -5*sqrt(15)/18 + 10/9, 5/18,  5*sqrt(15)/18 + 10/9,
          -5/9 + sqrt(15)/9, -5/9 + sqrt(15)/9, -5/9 - sqrt(15)/9,
          -5/9 - sqrt(15)/9,  4/9],
         [ 0,  0,  0,  0, sqrt(15)/6 + 5/6,  0, -sqrt(15)/6 + 5/6,  0, -2/3],
         [0, 0, 0, 0, 0, sqrt(15)/6 + 5/6,  0, -sqrt(15)/6 + 5/6, -2/3],
         [ 0, 0, 0, 0, -sqrt(15)/6 + 5/6, 0, sqrt(15)/6 + 5/6, 0, -2/3],
         [ 0, 0, 0, 0, 0, -sqrt(15)/6 + 5/6, 0, sqrt(15)/6 + 5/6, -2/3]])

    @staticmethod
    def fields():
        return ('ux', 'uy')

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'),
                ('N', 1, 'ux'), ('N', 1, 'uy'),
                ('N', 2, 'ux'), ('N', 2, 'uy'),
                ('N', 3, 'ux'), ('N', 3, 'uy'),
                ('N', 4, 'ux'), ('N', 4, 'uy'),
                ('N', 5, 'ux'), ('N', 5, 'uy'),
                ('N', 6, 'ux'), ('N', 6, 'uy'),
                ('N', 7, 'ux'), ('N', 7, 'uy'))

    def _compute_tensors(self, X, u, t):
        # X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6, X7, Y7, X8, Y8 = X
        X_mat = X.reshape(-1, 2)
        u_e = u.reshape(-1, 2)
        t = self.material.thickness


        self.K *= 0
        self.f *= 0
        self.S *= 0
        self.E *= 0

        for n_gauss, (xi, eta, w) in enumerate(self.gauss_points):
            # this is now the standard procedure for Total Lagrangian behavior
            dN_dxi = np.array([
                [-(eta - 1)*(eta + 2*xi)/4, -(2*eta + xi)*(xi - 1)/4],
                [ (eta - 1)*(eta - 2*xi)/4,  (2*eta - xi)*(xi + 1)/4],
                [ (eta + 1)*(eta + 2*xi)/4,  (2*eta + xi)*(xi + 1)/4],
                [-(eta + 1)*(eta - 2*xi)/4, -(2*eta - xi)*(xi - 1)/4],
                [             xi*(eta - 1),            xi**2/2 - 1/2],
                [          -eta**2/2 + 1/2,            -eta*(xi + 1)],
                [            -xi*(eta + 1),           -xi**2/2 + 1/2],
                [           eta**2/2 - 1/2,             eta*(xi - 1)]])
            dX_dxi = X_mat.T @ dN_dxi
            det = dX_dxi[0,0]*dX_dxi[1,1] - dX_dxi[1,0]*dX_dxi[0,1]
            dxi_dX = 1/det*np.array([[ dX_dxi[1,1], -dX_dxi[0,1]],
                                     [-dX_dxi[1,0],  dX_dxi[0,0]]])

            B0_tilde = dN_dxi @ dxi_dX
            H = u_e.T @ B0_tilde
            F = H + np.eye(2)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C_2d(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det*t
            K_geo = scatter_matrix(K_geo_small, 2)
            K_mat = B0.T @ C_SE @ B0 * det * t
            self.K += w*(K_geo + K_mat)
            self.f += B0.T @ S_v*det*t*w
            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], 0, S[1,1], 0, 0]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], 0, E[1,1], 0, 0]])
        return

    def _m_int(self, X, u, t=0):
        """
        Mass matrix using CAS-System
        """
        X_mat = X.reshape(-1, 2)
        t = self.material.thickness
        rho = self.material.rho

        self.M_small *= 0

        for xi, eta, w in self.gauss_points:
            N = np.array([  [(-eta + 1)*(-xi + 1)*(-eta - xi - 1)/4],
                            [ (-eta + 1)*(xi + 1)*(-eta + xi - 1)/4],
                            [   (eta + 1)*(xi + 1)*(eta + xi - 1)/4],
                            [  (eta + 1)*(-xi + 1)*(eta - xi - 1)/4],
                            [             (-eta + 1)*(-xi**2 + 1)/2],
                            [              (-eta**2 + 1)*(xi + 1)/2],
                            [              (eta + 1)*(-xi**2 + 1)/2],
                            [             (-eta**2 + 1)*(-xi + 1)/2]])

            dN_dxi = np.array([
                [-(eta - 1)*(eta + 2*xi)/4, -(2*eta + xi)*(xi - 1)/4],
                [ (eta - 1)*(eta - 2*xi)/4,  (2*eta - xi)*(xi + 1)/4],
                [ (eta + 1)*(eta + 2*xi)/4,  (2*eta + xi)*(xi + 1)/4],
                [-(eta + 1)*(eta - 2*xi)/4, -(2*eta - xi)*(xi - 1)/4],
                [             xi*(eta - 1),            xi**2/2 - 1/2],
                [          -eta**2/2 + 1/2,            -eta*(xi + 1)],
                [            -xi*(eta + 1),           -xi**2/2 + 1/2],
                [           eta**2/2 - 1/2,             eta*(xi - 1)]])
            dX_dxi = X_mat.T @ dN_dxi
            det = dX_dxi[0,0]*dX_dxi[1,1] - dX_dxi[1,0]*dX_dxi[0,1]
            self.M_small += N @ N.T * det * rho * t * w

        self.M = scatter_matrix(self.M_small, 2)
        return self.M
