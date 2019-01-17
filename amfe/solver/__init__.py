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

# --- ABSTRACT SUPER CLASS ---
# from .solver import *  # abstract super class


# --- STATICS SOLVERS ---
from .nonlinear_statics_solver import *
from .linear_statics_solver import *


# --- DYNAMICS SOLVERS ---
#     --- nonlinear ---
# from .nonlinear_dynamics_solver import *  # super class
from .generalized_alpha_nonlinear_dynamics_solver import *
from .jwh_alpha_nonlinear_dynamics_solver import *
#     --- linear ---
# from .linear_dynamics_solver import *  # super class
from .generalized_alpha_linear_dynamics_solver import *
from .jwh_alpha_linear_dynamics_solver import *


# --- STATE-SPACE DYNAMICS SOLVERS ---
#     --- nonlinear ---
# from .nonlinear_dynamics_solver_state_space import *  # super class
from .jwh_alpha_nonlinear_dynamics_solver_state_space import *
#     --- linear ---
# from .linear_dynamics_solver_state_space import *  # super class
from .jwh_alpha_linear_dynamics_solver_state_space import *

# --- SOLUTION CLASSES ---
from .solution import *
