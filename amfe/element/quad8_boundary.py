#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d quad8 boundary element.
"""

__all__ = [
    'Quad8Boundary'
]

import numpy as np

from .boundary_element import BoundaryElement

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Quad8Boundary(BoundaryElement):
    """
    Quad8 boundary element for 3D-Problems.
    """

    g = np.sqrt(3/5)
    w = 5/9
    w0 = 8/9
    gauss_points = ((-g, -g,  w*w), ( g, -g,  w*w ), ( g,  g,   w*w),
                    (-g,  g,  w*w), ( 0, -g, w0*w ), ( g,  0,  w*w0),
                    ( 0,  g, w0*w), (-g,  0,  w*w0), ( 0,  0, w0*w0))

    def __init__(self):
        super().__init__()
        return

    @staticmethod
    def fields():
        return ('ux', 'uy', 'uz')

    def f_mat(self, X, u):
        """
        Compute the full pressure contribution by performing gauss integration.
        """
        f_mat = np.zeros((8,3))
        x_vec = (X+u).reshape((8, 3))

        # gauss point evaluation of full pressure field
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

            dx_dxi = x_vec.T @ dN_dxi
            n = np.cross(dx_dxi[:,1], dx_dxi[:,0])
            f_mat += np.outer(N, n) * w
        return f_mat

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'uz'),
                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'uz'),
                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'uz'),
                ('N', 3, 'ux'), ('N', 3, 'uy'), ('N', 3, 'uz'),
                ('N', 4, 'ux'), ('N', 4, 'uy'), ('N', 4, 'uz'),
                ('N', 5, 'ux'), ('N', 5, 'uy'), ('N', 5, 'uz'),
                ('N', 6, 'ux'), ('N', 6, 'uy'), ('N', 6, 'uz'),
                ('N', 7, 'ux'), ('N', 7, 'uy'), ('N', 7, 'uz'))
