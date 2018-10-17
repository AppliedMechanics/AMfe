#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d linear line boundary element.
"""

__all__ = [
    'LineLinearBoundary'
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


class LineLinearBoundary(BoundaryElement):
    """
    Line Boundary element for 2D-Problems.
    """
    # Johannes Rutzmoser: rot_mat is a rotationmatrix, that turns +90deg
    # rot_mat = np.array([[0,-1], [1, 0]])
    # Christian Meyer: more intuitiv: boundary in math. positive direction equals outer vector => rotate -90deg:
    rot_mat = np.array([[0, 1], [-1, 0]])
    # weighting for the two nodes
    N = np.array([1/2, 1/2])

    def __init__(self, val, direct, time_func=None, shadow_area=False, ):
        super().__init__(val=val, direct=direct, time_func=time_func,
                         shadow_area=shadow_area, ndof=4)

    def dofs(self):
        return ((('ux', 'uy'), )*2 , ())

    def _compute_tensors(self, X, u, t):
        x_vec = (X+u).reshape((-1, 2)).T
        # Connection line between two nodes of the element
        v = x_vec[:,1] - x_vec[:,0]
        # Generate the orthogonal vector to connection line by rotation 90deg
        n = self.rot_mat @ v
        # Remember: n must not be normalized because forces are defined as force-values per area of forces per line
        # Thus the given forces have to be scaled with the area or line length respectively.
        #
        # f_mat: Generate the weights or in other words participations of each dof to generate force
        # Dimension of f_mat: (number of nodes, number of dofs per node)
        f_mat = np.outer(self.N, n)
        self.f = self.f_proj(f_mat) * self.val * self.time_func(t)
