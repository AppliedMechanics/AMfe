#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Tools for all elements.
"""

__all__ = [
    'f_proj_a',
    'f_proj_a_shadow'
]

import numpy as np


def f_proj_a(f_mat, direction):
    """
    Compute the force traction proportional to the area of the element
    in any-direction.

    Parameters
    ----------
    f_mat : ndarray
        normal force vector of one element in matrix notation. The shape of
        `f_mat`  is (no_of_nodes, dofs_per_node)
        It weights the participations of the nodes to the defined force
        e.g. for line element: a half for each node times length of the element
    direction : ndarray
        normalized vector describing the direction, in which the force should
        act.

    Returns
    -------
    f : ndarray
        force vector of traction vector in voigt notation (1d-array)

    """
    n_nodes, dofs_per_node = f_mat.shape
    f_out = np.zeros(n_nodes * dofs_per_node)
    for i, f_vec in enumerate(f_mat):
        f_out[i*dofs_per_node:(i+1)*dofs_per_node] = direction * np.sqrt(f_vec @ f_vec)
    return f_out


def f_proj_a_shadow(f_mat, direction):
    """
    Compute the force projection in any direction proportional to the projected
    area, i.e. the shadow-area, the area throws in the given direction.

    Parameters
    ----------
    f_mat : ndarray
        normal force vector of one element in matrix notation. The shape of
        `f_mat`  is (no_of_nodes, dofs_per_node)
    direction : ndarray
        normalized vector describing the direction, in which the force should
        act.

    Returns
    -------
    f : ndarray
        force vector of traction vector in voigt notation (1d-array)

    """
    n_nodes, dofs_per_node = f_mat.shape
    f_out = np.zeros(n_nodes * dofs_per_node)
    for i, f_vec in enumerate(f_mat):
        # by Johannes Rutzmoser:
        # f_out[i*dofs_per_node:(i+1)*dofs_per_node] = direction * (direction @ f_vec)
        # by Christian Meyer: I think this has to be divided by || direction || because of projection
        f_out[i * dofs_per_node:(i + 1) * dofs_per_node] = direction * ((direction @ f_vec) / np.linalg.norm(direction))

    return f_out
