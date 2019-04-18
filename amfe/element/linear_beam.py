#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3D linear Beam element
"""

__all__ = [
    'LinearBeam3D'
]

import numpy as np

from .element import Element

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except ModuleNotFoundError:
    print('Python was not able to load the fast fortran element routines.')


class LinearBeam3D(Element):
    """
    Beam-Element with Euler Bernoulli Theory
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = np.zeros((12, 12))
        self.M = np.zeros((12, 12))
        self.f = np.zeros(12)

    def fields(self):
        return ('ux', 'uy', 'uz', 'rotx', 'roty', 'rotz')

    def dofs(self):
        return (('N', 0, 'ux'),
                ('N', 0, 'uy'),
                ('N', 0, 'uz'),
                ('N', 0, 'rotx'),
                ('N', 0, 'roty'),
                ('N', 0, 'rotz'),
                ('N', 1, 'ux'),
                ('N', 1, 'uy'),
                ('N', 1, 'uz'),
                ('N', 1, 'rotx'),
                ('N', 1, 'roty'),
                ('N', 1, 'rotz'))

    def _compute_tensors(self, X, u, t):
        X_mat = X.reshape(-1, 3)
        l = np.linalg.norm(X_mat[1,:]-X_mat[0,:])

        # Element stiffness matrix
        k_el_loc = np.zeros((12, 12), dtype=float)

        # Tension part
        scale_tension = self.material.E_modulus*self.material.crosssec/l
        k_el_loc[0, 0] += scale_tension
        k_el_loc[6, 6] += scale_tension
        k_el_loc[6, 0] -= scale_tension
        k_el_loc[0, 6] -= scale_tension
        # Bending I_z
        scale_Iz = self.material.E_modulus * self.material.I_z/l
        k_el_loc[1, 1] += scale_Iz*12/l**2
        k_el_loc[1, 7] -= scale_Iz*12/l**2
        k_el_loc[7, 1] -= scale_Iz * 12 / l ** 2
        k_el_loc[5, 5] += scale_Iz*4
        k_el_loc[7, 7] += scale_Iz*12/l**2
        k_el_loc[11, 11] += scale_Iz*4
        k_el_loc[1, 5] += scale_Iz*6/l
        k_el_loc[5, 1] += scale_Iz*6/l
        k_el_loc[11, 1] += scale_Iz*6/l
        k_el_loc[1, 11] += scale_Iz*6/l
        k_el_loc[7, 5] -= scale_Iz*6/l
        k_el_loc[5, 7] -= scale_Iz * 6 / l
        k_el_loc[11, 5] += scale_Iz*2
        k_el_loc[5, 11] += scale_Iz * 2
        k_el_loc[11, 7] -= scale_Iz*6/l
        k_el_loc[7, 11] -= scale_Iz * 6 / l
        # Bending I_y
        scale_Iy = self.material.E_modulus * self.material.I_y/l
        k_el_loc[2, 2] += scale_Iy*12/l**2
        k_el_loc[4, 4] += scale_Iy*4
        k_el_loc[8, 8] += scale_Iy*12/l**2
        k_el_loc[10, 10] += scale_Iy*4
        k_el_loc[4, 2] -= scale_Iy*6/l
        k_el_loc[2, 4] -= scale_Iy*6/l
        k_el_loc[8, 2] -= scale_Iy*12/l**2
        k_el_loc[2, 8] -= scale_Iy * 12 / l ** 2
        k_el_loc[10, 2] -= scale_Iy*6/l
        k_el_loc[2, 10] -= scale_Iy * 6 / l
        k_el_loc[4, 8] += scale_Iy*6/l
        k_el_loc[8, 4] += scale_Iy * 6 / l
        k_el_loc[4, 10] += scale_Iy*2
        k_el_loc[10, 4] += scale_Iy*2
        k_el_loc[8, 10] += scale_Iy*6/l
        k_el_loc[10, 8] += scale_Iy * 6 / l
        # Torsion
        scale_Jx = self.material.G_modulus * self.material.J_x/l
        k_el_loc[3, 3] += scale_Jx
        k_el_loc[9, 9] += scale_Jx
        k_el_loc[3, 9] -= scale_Jx
        k_el_loc[9, 3] -= scale_Jx

        scale_m = self.material.rho*self.material.crosssec*l
        m_el_loc = np.zeros((12, 12))
        m_el_loc[0, 0] += 1.0/3.0
        m_el_loc[0, 6] += 1.0/6.0
        m_el_loc[6, 0] += 1.0/6.0
        m_el_loc[6, 6] += 1.0/3.0
        m_el_loc[1, 1] += 13.0/35.0
        m_el_loc[5, 1] += 11.0*l/210.0
        m_el_loc[1, 5] += 11.0 * l / 210.0
        m_el_loc[7, 1] += 9.0/70.0
        m_el_loc[1, 7] += 9.0/70.0
        m_el_loc[1, 11] -= 13.0*l/420.0
        m_el_loc[11, 1] -= 13.0*l/420.0
        m_el_loc[2, 2] += 13.0/35.0
        m_el_loc[4, 2] -= 11.0*l/210.0
        m_el_loc[2, 4] -= 11.0*l/210.0
        m_el_loc[8, 2] += 9.0/70.0
        m_el_loc[2, 8] += 9.0/70.0
        m_el_loc[10, 2] += 13.0*l/420.0
        m_el_loc[2, 10] += 13.0*l/420.0
        m_el_loc[3, 3] += self.material.I_p/3.0
        m_el_loc[9, 3] += self.material.I_p/6.0
        m_el_loc[3, 9] += self.material.I_p/6.0
        m_el_loc[9, 9] += self.material.I_p/3.0
        m_el_loc[4, 4] += l**2/105.0
        m_el_loc[4, 8] -= 13.0*l/420.0
        m_el_loc[8, 4] -= 13.0*l/420.0
        m_el_loc[4, 10] -= l**2/140.0
        m_el_loc[10, 4] -= l**2/140.0
        m_el_loc[5, 5] += l**2/105.0
        m_el_loc[5, 7] += 13.0*l/420.0
        m_el_loc[7, 5] += 13.0 * l / 420.0
        m_el_loc[11, 5] -= l**2/140.0
        m_el_loc[5, 11] -= l ** 2 / 140.0
        m_el_loc[7, 7] += 13.0/35.0
        m_el_loc[7, 11] -= 11.0*l/210.0
        m_el_loc[11, 7] -= 11.0 * l / 210.0
        m_el_loc[8, 8] += 13.0/35.0
        m_el_loc[8, 10] += 11.0*l/210.0
        m_el_loc[10, 8] += 11.0*l/210.0
        m_el_loc[10, 10] += l**2/105.0
        m_el_loc[11, 11] += l**2/105.0
        m_el_loc *= scale_m

        # Rotation
        X3 = self.material.X3

        d_2 = (X_mat[1, :] - X_mat[0, :])
        e_x = d_2/l
        d_3 = X3 - X_mat[0, :]
        e_y = np.cross(d_3, d_2)
        e_y = e_y/np.linalg.norm(e_y)
        e_z = np.cross(e_x, e_y)

        R = np.zeros((3, 3), dtype=float)

        R[0, :] = e_x
        R[1, :] = e_y
        R[2, :] = e_z

        Trot = np.zeros((12, 12))
        for i in range(4):
            Trot[3*i, 3*i:3*i+3] = R[0, :]
            Trot[3 * i+1, 3 * i:3 * i + 3] = R[1, :]
            Trot[3 * i+2, 3 * i:3 * i + 3] = R[2, :]

        k_el = Trot.T.dot(k_el_loc).dot(Trot)
        m_el = Trot.T.dot(m_el_loc).dot(Trot)
        # Make symmetric (because of round-off errors)
        self.K = 1/2*(k_el+k_el.T)
        self.M = 1/2*(m_el+m_el.T)
        self.f = self.K.dot(u)
        self.S = np.zeros((2, 6))
        self.E = np.zeros((2, 6))
        return

    def _m_int(self, X, u, t):
        self._compute_tensors(X, u, t)
        return self.M
