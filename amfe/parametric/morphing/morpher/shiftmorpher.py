#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from .basemorpher import MeshMorpher
from ..implementer.rbfimplementer import RbfMorpherImplementer


__all__ = ['ShiftRbfMorpher'
           ]


class ShiftRbfMorpher(MeshMorpher):
    def __init__(self, basis='multi_quadratic_biharmonic_spline', radius=0.1):
        super().__init__()
        self._implementer = RbfMorpherImplementer(basis, radius)

    def _writeshifts(self, nodes_boundary_before, shift, direction, nodes_boundary_after):
        dirdict = {'x': 0, 'y': 1, 'z': 2}
        nodes_boundary_after[:, :] = nodes_boundary_before
        nodes_boundary_after[:, dirdict[direction]] += shift

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def get_shifts(self, nodes_boundary, shiftX=0.0, shiftY=0.0):

        nodes_boundary_after = np.copy(nodes_boundary)

        # x-direction
        if shiftX and shiftX != 0.0:
            self._writeshifts(nodes_boundary_after, shiftX, 'x', nodes_boundary_after)
        # y-direction
        if shiftY and shiftY != 0.0:
            self._writeshifts(nodes_boundary_after, shiftY, 'y', nodes_boundary_after)

        return nodes_boundary_after

    def morph(self, nodes_reference, nodes_boundary_before, nodes_boundary_after):
        return self._implementer.morph(nodes_reference, nodes_boundary_before, nodes_boundary_after)
