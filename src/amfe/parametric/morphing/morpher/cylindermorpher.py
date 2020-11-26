#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from .basemorpher import MeshMorpher
from ..implementer.ffdimplementer import FfdMorpherImplementer


__all__ = ['CylinderFfdMorpher']


class CylinderFfdMorpher(MeshMorpher):
    def __init__(self, origin=np.array([[0], [0], [0]]), csys=np.eye(3), mu_shape=(3, 3, 3)):
        super().__init__()
        self._implementer = FfdMorpherImplementer(origin, csys, mu_shape)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def _getshifts(self, max_value, n):
        return [max_value * i / (n - 1) for i in range(n)]

    def morph(self, nodes_reference, scaleZ, R, centerline):
        (nx, ny, nz) = self._implementer.mu_shape
        mu_x = np.zeros((nx, ny, nz))
        mu_y = np.zeros((nx, ny, nz))
        mu_z = np.zeros((nx, ny, nz))

        if R:
            for i in range(ny):
                for j in range(nz):
                    val = self._getshifts(R(j / (nz - 1))[0] - 1, nx)
                    mu_x[:, i, j] = mu_x[:, i, j] + val - (R(j / (nz - 1))[0] - 1) * 0.5 + centerline(j / (nz - 1))[0]
            for i in range(nx):
                for j in range(nz):
                    val = self._getshifts(R(j / (nz - 1))[0] - 1, ny)
                    mu_y[i, :, j] = mu_y[i, :, j] + val - (R(j / (nz - 1))[1] - 1) * 0.5 + centerline(j / (nz - 1))[1]
        if scaleZ and scaleZ != 1:
            val = self._getshifts(scaleZ - 1, nz)
            for i in range(nx):
                for j in range(ny):
                    mu_z[i, j, :] = mu_z[i, j, :] + val
        return self._implementer.morph(nodes_reference, mu_x, mu_y, mu_z)
