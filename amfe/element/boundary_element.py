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


class BoundaryElement:
    """
    Class for the application of Neumann Boundary Conditions.
    """
    def __init__(self):
        """
        Parameters
        ----------
        ndof : int
            number of dofs of the boundary element

        Returns
        -------
        None
        """

    @staticmethod
    def fields():
        """
        Returns the unique physical fields that are local dofs of the element

        Returns
        -------
        fields: tuple[str]
            unique fields
        """
        return ()

    def f_mat(self, X, u):
        raise NotImplementedError('The f_mat is not implemented')

    def dofs(self):
        return ()
