#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d prism6 element.
"""

__all__ = [
    'Prism6'
]

import numpy as np

from .element import Element

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Prism6(Element):
    """
    Three dimensional Prism element.
    """
    name = 'Prism6'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.zeros((18,18))
        self.f = np.zeros(18)
        self.M = np.zeros((18,18))
        self.S = np.zeros((6,6))
        self.E = np.zeros((6,6))
        return

    @staticmethod
    def fields():
        return ('ux', 'uy', 'uz')

    def dofs(self):
        return ()

    def _compute_tensors(self, X, u, t):
        raise NotImplementedError('The Prism Element is not implemented')

    def _m_int(self, X, u, t=0):
        raise NotImplementedError('The Prism Element is not implemented')
