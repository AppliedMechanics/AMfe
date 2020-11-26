#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from .basemorpher import MeshMorpher
from ..implementer.ffdimplementer import FfdMorpherImplementer


__all__ = ['CuboidFfdMorpher']


class CuboidFfdMorpher(MeshMorpher):
    def __init__(self, origin=np.array([[0], [0], [0]]), csys=np.eye(3), mu_shape=(3, 3, 3)):
        super().__init__()
        self._implementer = FfdMorpherImplementer(origin, csys, mu_shape)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def _getshifts(self, max_value, n):
        return [max_value * i / (n - 1) for i in range(n)]

    def morph(self, nodes_reference, scale_x, scale_y, scale_z):
        (nx, ny, nz) = self._implementer.mu_shape
        mu_x = np.zeros((nx, ny, nz))
        mu_y = np.zeros((nx, ny, nz))
        mu_z = np.zeros((nx, ny, nz))

        # x-direction
        if scale_x and scale_x != 1:
            val = self._getshifts(scale_x - 1, nx)
            for i in range(ny):
                for j in range(nz):
                    mu_x[:, i, j] = mu_x[:, i, j] + val
        if scale_y and scale_y != 1:
            val = self._getshifts(scale_y - 1, ny)
            for i in range(nx):
                for j in range(nz):
                    mu_y[i, :, j] = mu_y[i, :, j] + val

        if scale_z and scale_z != 1:
            val = self._getshifts(scale_z - 1, nz)
            for i in range(nx):
                for j in range(ny):
                    mu_z[i, j, :] = mu_z[i, j, :] + val
        return self._implementer.morph(nodes_reference, mu_x, mu_y, mu_z)
