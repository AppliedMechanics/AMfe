#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d tri3 boundary element.
"""

__all__ = [
    'Tri3Boundary'
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


class Tri3Boundary(BoundaryElement):
    """
    Class for application of Neumann Boundary Conditions.
    """

    def __init__(self, val, direct, time_func=None, shadow_area=False):
        super().__init__(val=val, direct=direct, time_func=time_func,
                         shadow_area=shadow_area, ndof=9)

    def _compute_tensors(self, X, u, t):
        x_vec = (X+u).reshape((-1, 3)).T
        v1 = x_vec[:,2] - x_vec[:,0]
        v2 = x_vec[:,1] - x_vec[:,0]
        n = np.cross(v1, v2)/2
        N = np.array([1/3, 1/3, 1/3])
        f_mat = np.outer(N, n)
        # positive sign as it is external force on the right hand side of the
        # function
        self.f = self.f_proj(f_mat) * self.val * self.time_func(t)
