#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
2d lumped bar element.
"""

__all__ = [
    'Bar2Dlumped'
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


class Bar2Dlumped(Element):
    """
    Bar-Element with 2 nodes and lumped stiffness matrix.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.zeros((4,4))
        self.M = np.zeros((4,4))
        self.f = np.zeros(4)

    @staticmethod
    def fields():
        return ('ux', 'uy')

    def dofs(self):
        return (('N', 0, 'ux'),
                ('N', 0, 'uy'),
                ('N', 1, 'ux'),
                ('N', 1, 'uy'))

    def _compute_tensors(self, X, u, t):
        self._k_and_m_int(X, u, t)

    def _k_and_m_int(self, X, u, t):

        # X1, Y1, X2, Y2 = X
        X_mat = X.reshape(-1, 2)
        l = np.linalg.norm(X_mat[1,:]-X_mat[0,:])

        # Element stiffnes matrix
        k_el_loc = self.material.E_modulus*self.material.crosssec/l*np.array([[1, -1],
                                                          [-1, 1]])
        temp = (X_mat[1,:]-X_mat[0,:])/l
        A = np.array([[temp[0], temp[1], 0,       0],
                      [0,       0,       temp[0], temp[1]]])
        k_el = A.T.dot(k_el_loc.dot(A))

        # Element mass matrix
        m_el = self.material.rho*self.material.crosssec*l/6*np.array([[3, 0, 0, 0],
                                                    [0, 3, 0, 0],
                                                    [0, 0, 3, 0],
                                                    [0, 0, 0, 3]])

        # Make symmetric (because of round-off errors)
        self.K = 1/2*(k_el+k_el.T)
        self.M = 1/2*(m_el+m_el.T)
        return self.K, self.M

    def _k_int(self, X, u, t):
        k_el, m_el = self._k_and_m_int(X, u, t)
        return k_el

    def _m_int(self, X, u, t=0):
        k_el, m_el = self._k_and_m_int(X, u, t)
        return m_el
