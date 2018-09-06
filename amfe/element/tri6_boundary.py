#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d tri6 boundary element.
"""

__all__ = [
    'Tri6Boundary'
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


class Tri6Boundary(BoundaryElement):
    """
    Boundary element with variatonally consistent boundary forces.

    Notes
    -----
    This function has been updated to give a variationally consistent
    integrated skin element.
    """

    # Gauss-Points like ABAQUS or ANSYS
    gauss_points = ((1/6, 1/6, 2/3, 1/3),
                    (1/6, 2/3, 1/6, 1/3),
                    (2/3, 1/6, 1/6, 1/3))

    def __init__(self, val, direct, time_func=None, shadow_area=False):
        super().__init__(val=val, direct=direct, time_func=time_func,
                         shadow_area=shadow_area, ndof=18)

    def _compute_tensors(self, X, u, t):
        """
        Compute the full pressure contribution by performing gauss integration.

        """
        # self.f *= 0
        f_mat = np.zeros((6,3))
        x_vec = (X+u).reshape((-1, 3))

        # gauss point evaluation of full pressure field
        for L1, L2, L3, w in self.gauss_points:
            N = np.array([L1*(2*L1 - 1), L2*(2*L2 - 1), L3*(2*L3 - 1),
                          4*L1*L2, 4*L2*L3, 4*L1*L3])

            dN_dL = np.array([  [4*L1 - 1,        0,        0],
                                [       0, 4*L2 - 1,        0],
                                [       0,        0, 4*L3 - 1],
                                [    4*L2,     4*L1,        0],
                                [       0,     4*L3,     4*L2],
                                [    4*L3,        0,     4*L1]])

            dx_dL = x_vec.T @ dN_dL
            v1 = dx_dL[:,2] - dx_dL[:,0]
            v2 = dx_dL[:,1] - dx_dL[:,0]
            n = np.cross(v1, v2)
            f_mat += np.outer(N, n) / 2 * w
        # no minus sign as force will be on the right hand side of eqn.
        self.f = self.f_proj(f_mat) * self.val * self.time_func(t)