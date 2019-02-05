#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from operator import iadd, isub

from .basemorpher import MeshMorpher
from ..implementer.rbfimplementer import RbfMorpherImplementer


__all__ = ['TaperRbfMorpher',
           'TaperRbfMorpherEnhanced',
           'TaperRbfMorpherPhase1'
           ]


class TaperRbfMorpher(MeshMorpher):
    def __init__(self, basis='multi_quadratic_biharmonic_spline', radius=0.1):
        super().__init__()
        self._implementer = RbfMorpherImplementer(basis, radius)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def get_shifts(self, nodes_boundary, x_0, r, direction='+'):

        nodes_boundary_after = np.copy(nodes_boundary)

        directiondict = {'+': iadd,
                         '-': isub
                         }

        x_min = x_0 - r
        x_max = x_0 + r

        for i, X in enumerate(nodes_boundary_after[:,0]):
            if x_min < X < x_max:
                phi = np.pi/2 * (X - x_0) / r
                shift_x = r * np.sin(phi) - (X - x_0)
                shift_y = r * np.cos(phi)
                nodes_boundary_after[i, 0] += shift_x
                nodes_boundary_after[i, 1] = directiondict[direction](nodes_boundary_after[i, 1], shift_y)

        return nodes_boundary_after

    def morph(self, nodes_reference, nodes_boundary_before, nodes_boundary_after, n=1, callback=None):
        return self._implementer.morph(nodes_reference, nodes_boundary_before, nodes_boundary_after, n, callback)


class TaperRbfMorpherEnhanced(MeshMorpher):
    def __init__(self, basis='multi_quadratic_biharmonic_spline', radius=0.1):
        super().__init__()
        self._implementer = RbfMorpherImplementer(basis, radius)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def get_shifts(self, nodes_boundary, x_0, r, direction='+'):

        nodes_boundary_after = np.copy(nodes_boundary)

        directiondict = {'+': iadd,
                         '-': isub
                         }

        r_star = 2.0*r
        r_inf = 6.0*r

        x_min = x_0 - r_star
        x_max = x_0 + r_star

        x_min_inf = x_0 - r_inf
        x_max_inf = x_0 + r_inf

        B = r_inf - r_star
        H = r_inf - r

        for i, X in enumerate(nodes_boundary_after[:,0]):
            if x_min < X < x_max:
                phi = np.pi/2 * (X - x_0) / r_star

                nodes_boundary_after[i, 0] = r * np.sin(phi) + x_0
                nodes_boundary_after[i, 1] = directiondict[direction](nodes_boundary_after[i, 1], r * np.cos(phi))
            elif x_min_inf < X < x_0:
                s = X - (x_0 - r_inf)
                nodes_boundary_after[i, 0] = x_0 - r_inf + s / B * H
            elif x_0 < X < x_max_inf:
                s = x_0 + r_inf - X
                nodes_boundary_after[i, 0] = x_0 + r_inf - s/B * H

        return nodes_boundary_after

    def morph(self, nodes_reference, nodes_boundary_before, nodes_boundary_after, n=1, callback=None):
        return self._implementer.morph(nodes_reference, nodes_boundary_before, nodes_boundary_after, n, callback)


class TaperRbfMorpherPhase1(MeshMorpher):
    def __init__(self, basis='multi_quadratic_biharmonic_spline', radius=0.1):
        super().__init__()
        self._implementer = RbfMorpherImplementer(basis, radius)

    def offline(self, nodes_reference):
        self._implementer.offline(nodes_reference)

    def get_shifts(self, nodes_boundary, x_0, r, direction='+'):

        nodes_boundary_after = np.copy(nodes_boundary)

        r_star = 2.0*r
        r_inf = 6.0*r

        x_min = x_0 - r_star
        x_max = x_0 + r_star

        x_min_inf = x_0 - r_inf
        x_max_inf = x_0 + r_inf

        B = r_inf - r_star
        H = r_inf - r

        for i, X in enumerate(nodes_boundary_after[:,0]):
            if x_min < X < x_max:
                # phi = np.pi/2 * (X - x_0) / r_star

                # nodes_boundary_after[i, 0] = r * np.sin(phi) + x_0
                nodes_boundary_after[i]
            elif x_min_inf < X < x_0:
                s = X - (x_0 - r_inf)
                nodes_boundary_after[i, 0] = x_0 - r_inf + s / B * H
            elif x_0 < X < x_max_inf:
                s = x_0 + r_inf - X
                nodes_boundary_after[i, 0] = x_0 + r_inf - s/B * H

        return nodes_boundary_after

    def morph(self, nodes_reference, nodes_boundary_before, nodes_boundary_after, n=1, callback=None):
        return self._implementer.morph(nodes_reference, nodes_boundary_before, nodes_boundary_after, n, callback)
