#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from .tools import f_proj_a, f_proj_a_shadow
from .neumann_base import *


class FixedDirectionNeumann(NeumannBase):
    """
    Class for a Neumann condition that has a fixed direction and a constant force/area ratio (fixed traction)
    """
    def __init__(self, direction, time_func=lambda t: 1):
        """

        Parameters
        ----------
        direction : ndarray, dtype=float
            direction of the force
        time_func : function
             pointer to function with signature  float func(float: t)
        """
        super().__init__(direct=direction, time_func=time_func)
        self._direction = direction
        self._time_func = time_func

    def _amp(self, u, t):
        return self._time_func(t)

    def _f_proj(self, f_mat):
        # non-projected solution
        return f_proj_a(f_mat, self._direction)


class NormalFollowingNeumann(NeumannBase):
    """
    Class for a Neumann condition that follows the normal on a surface
    """
    def __init__(self, time_func=lambda t: 1):
        """

        Parameters
        ----------
        time_func : function
             pointer to function with signature  float func(float: t)
        """
        super().__init__(time_func=time_func)
        self._time_func = time_func

    def _amp(self, u, t):
        return self._time_func(t)

    def _f_proj(self, f_mat):
        return f_mat.flatten()


class ProjectedAreaNeumann(NeumannBase):
    """
    Neumann Condition with fixed direction, but forces proportional to the shadow area in that direction,

    i.e. the area of the surface projected on the direction
    """
    def __init__(self, direction, time_func=lambda t: 1):
        """

        Parameters
        ----------
        direction : ndarray, dtype=float
            direction of the force
        time_func : function
             pointer to function with signature  float func(float: t)
        """
        super().__init__(direction=direction, time_func=time_func)
        self._direction = direction
        self._time_func = time_func

    def _amp(self, u, t):
        return self._time_func(t)

    def _f_proj(self, f_mat):
        return f_proj_a_shadow(f_mat, self._direction)
