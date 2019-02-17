#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from .basemorpher import MeshMorpher
from ..implementer.ffdimplementer import FfdMorpherImplementer2D
from ..implementer.rbfimplementer import RbfMorpherImplementer


__all__ = ['RectangleFfdMorpher',
           'RectangleRbfMorpher'
           ]


class RectangleFfdMorpher(MeshMorpher):
    def __init__(self, origin=np.array([[0], [0]]), csys=np.eye(2), mu_shape=(2, 2)):
        super().__init__()
        self._implementer = FfdMorpherImplementer2D(origin, csys, mu_shape)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def _getshifts(self, max_value, n):
        return [max_value * i / (n - 1) for i in range(n)]

    def morph(self, nodes_reference, scaleX, scaleY):
        (nx, ny) = self._implementer.mu_shape
        mu_x = np.zeros((nx, ny))
        mu_y = np.zeros((nx, ny))

        # x-direction
        if scaleX and scaleX != 1:
            val = self._getshifts(scaleX - 1, nx)
            for i in range(ny):
                mu_x[:, i] = mu_x[:, i] + val
        # y-direction
        if scaleY and scaleY != 1:
            val = self._getshifts(scaleY - 1, ny)
            for i in range(nx):
                mu_y[i, :] = mu_y[i, :] + val

        return self._implementer.morph(nodes_reference, mu_x, mu_y)


class RectangleRbfMorpher(MeshMorpher):
    def __init__(self, basis='multi_quadratic_biharmonic_spline', radius=0.1):
        super().__init__()
        self._implementer = RbfMorpherImplementer(basis, radius)

    def _writeshifts(self, nodes_boundary_before, scale, direction, nodes_boundary_after):
        dirdict = {'x': 0, 'y': 1, 'z': 2}
        origin = np.min(nodes_boundary_before[:, dirdict[direction]])
        length = np.max(nodes_boundary_before[:, dirdict[direction]]) - origin
        nodes_boundary_after[:, :] = nodes_boundary_before
        nodes_boundary_after[:, dirdict[direction]] = origin + scale * (nodes_boundary_before[:, dirdict[direction]] - origin)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def get_shifts(self, nodes_boundary, scaleX, scaleY):

        nodes_boundary_after = np.copy(nodes_boundary)

        # x-direction
        if scaleX and scaleX != 1.0:
            self._writeshifts(nodes_boundary_after, scaleX, 'x', nodes_boundary_after)
        # y-direction
        if scaleY and scaleY != 1.0:
            self._writeshifts(nodes_boundary_after, scaleY, 'y', nodes_boundary_after)

        return nodes_boundary_after

    def morph(self, nodes_reference, nodes_boundary_before, nodes_boundary_after, n=1, callback=None):
        return self._implementer.morph(nodes_reference, nodes_boundary_before, nodes_boundary_after, n, callback)
