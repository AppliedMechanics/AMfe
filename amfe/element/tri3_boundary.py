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

    def __init__(self):
        super().__init__()

    @staticmethod
    def fields():
        return ('ux', 'uy', 'uz')

    def f_mat(self, X, u):
        x_vec = (X+u).reshape((-1, 3)).T
        v1 = x_vec[:,2] - x_vec[:,0]
        v2 = x_vec[:,1] - x_vec[:,0]
        n = np.cross(v1, v2)/2
        N = np.array([1/3, 1/3, 1/3])
        f_mat = np.outer(N, n)
        return f_mat

    def dofs(self):
        return (('N', 0, 'ux'), ('N', 0, 'uy'), ('N', 0, 'uz'),
                ('N', 1, 'ux'), ('N', 1, 'uy'), ('N', 1, 'uz'),
                ('N', 2, 'ux'), ('N', 2, 'uy'), ('N', 2, 'uz'))
