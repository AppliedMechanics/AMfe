#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class for all boundary elements.
"""

__all__ = [
    'BoundaryElement'
]

import numpy as np

from .element import Element
from .tools import f_proj_a, f_proj_a_shadow

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class BoundaryElement(Element):
    """
    Class for the application of Neumann Boundary Conditions.

    Attributes
    ----------
    time_func : func
        function returning a value between {-1, 1} which is time dependent
        storing the time dependency of the Neumann Boundary condition.
        Example for constant function:

        >>> def func(t):
        >>>    return 1

    f_proj : func
        function producing the nodal force vector from the given nodal force
        vector in normal direction.

    val : float
    
    direct : {'normal', numpy.array}
        'normal' means that force acts normal to the current surface
        numpy.array is a vector in which the force should act (with global coordinate system as reference)
    
    f : numpy.array
        local external force vector of the element
    
    K : numpy.array
        tangent stiffness matrix of the element (for boundary elements typically zero)
    
    M : numpy.array
        mass matrix of the element (for boundary elements typically zero)
    
        """

    def __init__(self, val, ndof, direct='normal', time_func=None,
                 shadow_area=False):
        """
        Parameters
        ----------
        val : float
            scale value for the pressure/traction onto the element
        ndof : int
            number of dofs of the boundary element
        direct : str 'normal' or ndarray, optional
            array giving the direction, in which the traction force should act.
            alternatively, the keyword 'normal' may be given.
            Normal means that the force will follow the normal of the surface
            Default value: 'normal'.
        time_func : function object
            Function object returning a value between -1 and 1 given the
            input t:

            >>> f_applied = val * time_func(t)

        shadow_area : bool, optional
            Flat setting, if force should be proportional to the shadow area,
            i.e. the area of the surface projected on the direction. Default
            value: 'False'.

        Returns
        -------
        None
        """
        self.val = val
        self.f = np.zeros(ndof)
        self.K = np.zeros((ndof, ndof))
        self.M = np.zeros((ndof, ndof))
        self.direct = direct

        # select the correct f_proj function in order to fulfill the direct
        # and shadow area specification
        if direct is 'normal':
            def f_proj(f_mat):
                return f_mat.flatten()
        else: # direct has to be a vector
            # save direct to be an array
            self.direct = np.array(direct)
            if shadow_area: # projected solution
                def f_proj(f_mat):
                    return f_proj_a_shadow(f_mat, self.direct)
            else: # non-projected solution
                def f_proj(f_mat):
                    return f_proj_a(f_mat, self.direct)

        self.f_proj = f_proj
        # time function...
        def const_func(t):
            return 1
        if time_func is None:
            self.time_func = const_func
        else:
            self.time_func = time_func

    def _m_int(self, X, u, t=0):
        return self.M
