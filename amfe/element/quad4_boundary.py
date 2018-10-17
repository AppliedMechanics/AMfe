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

    def __init__(self, val, direct, time_func=None, shadow_area=False):
        super().__init__(val=val, direct=direct, time_func=time_func,
                         shadow_area=shadow_area, ndof=12)
        return

    def dofs(self):
        return ((('ux', 'uy', 'uz'), )*4 , ())

    def _compute_tensors(self, X, u, t):
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
        # no minus sign as force will be on the right hand side of eqn.
        self.f = self.f_proj(f_mat) * self.val * self.time_func(t)
