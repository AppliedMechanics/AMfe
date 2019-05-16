#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d quad4 boundary element.
"""

__all__ = [
    'Quad4Boundary'
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


class Quad4Boundary(BoundaryElement):
    """
    Quad4 boundary element for 3D-Problems.
    """

    g1 = 1/np.sqrt(3)

    gauss_points = ((-g1, -g1, 1.),
                    ( g1, -g1, 1.),
                    ( g1,  g1, 1.),
                    (-g1,  g1, 1.))

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
        f_mat = np.zeros((4,3))
        x_vec = (X+u).reshape((4, 3))

        # gauss point evaluation of full pressure field
        for xi, eta, w in self.gauss_points:

            N = np.array([  [(-eta + 1)*(-xi + 1)/4],
                            [ (-eta + 1)*(xi + 1)/4],
                            [  (eta + 1)*(xi + 1)/4],
                            [ (eta + 1)*(-xi + 1)/4]])

            dN_dxi = np.array([ [ eta/4 - 1/4,  xi/4 - 1/4],
                                [-eta/4 + 1/4, -xi/4 - 1/4],
                                [ eta/4 + 1/4,  xi/4 + 1/4],
                                [-eta/4 - 1/4, -xi/4 + 1/4]])

            dx_dxi = x_vec.T @ dN_dxi
            n = np.cross(dx_dxi[:,1], dx_dxi[:,0])
            f_mat += np.outer(N, n) * w
        return f_mat

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'uz'),
                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'uz'),
                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'uz'),
                ('N', 3, 'ux'), ('N', 3, 'uy'), ('N', 3, 'uz'))
