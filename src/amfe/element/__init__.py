#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Element module.

Element module in which the finite elements are described on element level.

This module is arbitrarily extensible. The idea is to use the basis class element which provides the functionality for
an efficient solution of a time integration by only once calling the internal tensor computation and then extracting
the tangential stiffness matrix and the internal force vector in one run.

Some remarks resulting in the observations of the profiler:
Most of the time is spent with python-functions, when they are used. For instance the kron-function in order to build
the scattered geometric stiffness matrix or the trace function are very inefficient. They can be done better when using
direct functions.

If some element things are really time critical, it is recommended to port the heavy computation to FORTRAN. This can
be achieved by using the provided f2py routines and reprogram the stuff for the own use.
"""

# --- TOOLS ---
from .tools import *


# --- ELEMENTS ---
#     --- volume ---
# from .element import *  # super class
from .tri3 import *
from .tri6 import *
from .quad4 import *
from .quad8 import *
from .tet4 import *
from .tet10 import *
from .hexa8 import *
from .hexa20 import *
from .bar_2d_lumped import *
from .linear_beam import *

#     --- boundary ---
# from .boundary_element import *  # super class
from .tri3_boundary import *
from .tri6_boundary import *
from .quad4_boundary import *
from .quad8_boundary import *
from .line_linear_boundary import *
from .line_quadratic_boundary import *
