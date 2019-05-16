#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Tools for mechanical systems.
"""


import copy

from .reduced_system import ReducedSystem
from .mechanical_system_state_space import MechanicalSystemStateSpace
from .reduced_system_state_space import ReducedSystemStateSpace

__all__ = [
    'reduce_mechanical_system',
    'convert_mechanical_system_to_state_space',
    'reduce_mechanical_system_state_space'
]


def reduce_mechanical_system(mechanical_system, V, overwrite=False, assembly='indirect'):
    '''
    Reduce the given mechanical system with the linear basis V.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray
        Reduction Basis for the reduced system
    overwrite : bool, optional
        Switch, if mechanical system should be overwritten (is less memory intensive for large systems) or not.
    assembly : str {'direct', 'indirect'}
        flag setting, if direct or indirect assembly is done. For larger reduction bases, the indirect method is much
        faster.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and reduction basis V.
    '''

    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
    reduced_sys.__class__ = ReducedSystem
    reduced_sys.V = V.copy()
    reduced_sys.V_unconstr = reduced_sys.dirichlet_class.unconstrain_vec(V)
    reduced_sys.u_red_output = []
    reduced_sys.assembly_type = assembly
    reduced_sys.M(force_update=True)
    reduced_sys.D(force_update=True)
    return reduced_sys


def convert_mechanical_system_to_state_space(mechanical_system, regular_matrix=None, overwrite=False):
    if overwrite:
        sys = mechanical_system
    else:
        sys = copy.deepcopy(mechanical_system)
    sys.__class__ = MechanicalSystemStateSpace
    sys.x_output = []
    if regular_matrix is None:
        sys.R_constr = sys.K()
    else:
        sys.R_constr = regular_matrix
    sys.E_constr = None
    sys.E(force_update=True)
    return sys


def reduce_mechanical_system_state_space(mechanical_system_state_space, right_basis, left_basis=None, overwrite=False):
    if overwrite:
        red_sys = mechanical_system_state_space
    else:
        red_sys = copy.deepcopy(mechanical_system_state_space)
    red_sys.__class__ = ReducedSystemStateSpace
    red_sys.V = right_basis.copy()
    if left_basis is None:
        red_sys.W = right_basis.copy()
    else:
        red_sys.W = left_basis.copy()
    red_sys.x_red_output = []
    red_sys.E_constr = None
    red_sys.E(force_update=True)
    return red_sys
