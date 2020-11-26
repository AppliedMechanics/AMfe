#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solver module.

Module for solving systems in AMfe.
"""

# --- CENTRAL SOLVER MODULE ---
from .solver import *

# --- INTEGRATOR MODULE ---
from .integrator import *

# --- NONLINEAR SOLVERS  ---
from .nonlinear_solver import *

# --- TRANSLATOR MODULE BETWEEN DIFFERENT SYSTEM TYPES AND SOLVERS ---
from .translators import *

# --- SOLUTION CLASSES ---
from .solution import *

# --- DOMAIN DECOMPOSITION SOLVERS ---
from .domain_decomposition_solver import *

# --- INITIALIZERS ---
from .initializer import *

# --- TOOLS ---
from .tools import *
