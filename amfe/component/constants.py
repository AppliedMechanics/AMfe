#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from amfe.element import *

__all__ = ['ELEPROTOTYPEHELPERLIST',
           'SHELLELEPROTOTYPEHELPERLIST']


ELEPROTOTYPEHELPERLIST = (('Tri3', Tri3, Tri3Boundary),
                          ('Tri6', Tri6, Tri6Boundary),
                          ('Quad4', Quad4, Quad4Boundary),
                          ('Quad8', Quad8, Quad8Boundary),
                          ('Tet4', Tet4, None),
                          ('Tet10', Tet10, None),
                          ('Hexa8', Hexa8, None),
                          ('Hexa20', Hexa20, None),
                          ('Prism6', None, None),
                          ('straight_line', Bar2Dlumped, LineLinearBoundary),
                          ('quadratic_line', None, LineQuadraticBoundary))

SHELLELEPROTOTYPEHELPERLIST = (('straight_line', LinearBeam3D, None),)
