#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d quadratic line boundary element.
"""

__all__ = [
    'LineQuadraticBoundary'
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


class LineQuadraticBoundary(BoundaryElement):
    """
    Quadratic line boundary element for 2D problems.
    """

    #rot_mat = np.array([[ 0, -1],
    #                    [ 1,  0]])
    # same as above:
    rot_mat = np.array([[0, 1],[-1, 0]])

    N = np.array([1, 1, 4])/6

    def __init__(self, val, direct, time_func=None, shadow_area=False):

        super().__init__(val=val, direct=direct, time_func=time_func,
                         shadow_area=shadow_area, ndof=6)

    def _compute_tensors(self, X, u, t):
        x_vec = (X+u).reshape((-1, 2)).T
        v = x_vec[:,1] - x_vec[:,0]
        n = self.rot_mat @ v
        f_mat = np.outer(self.N, n)
        self.f = self.f_proj(f_mat) * self.val * self.time_func(t)
