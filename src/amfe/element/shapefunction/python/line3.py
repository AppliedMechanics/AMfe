#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING,
# CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license.
# See LICENSE file for more information.
#
"""
This module implements the Line3 shape-function
"""

from amfe.element.shapefunction.base import ShapeFunction1DBase


class PythonLine3ShapeFunction1D(ShapeFunction1DBase):
    r"""
    Quadratic Line3 Shape Function with three nodes

    ..code::
                 ^
                 |
        (-1.0)   |(0.0)   (1.0)
        -o-------o-------o--------->
          1      |3       2        xi
                 |

    """

    def evaluate(self, xi, out):
        """
        Evaluates the shape-function at xi

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        out : array_like
            Array object in which the evaluated shape-function
            coordinates are written.
        """
        out[0] = -0.5 * xi * (1.0 - xi)
        out[1] = 0.5 * xi * (1.0 + xi)
        out[2] = (1.0 - xi ** 2)

    def jacobian(self, xi, out):
        """
        Evaluates the jacobian of the shape-function at xi with respect
        to xi.

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        out : array_like
            Array object in which the jacobian is written.
        """
        out[0, 0] = 1.0 * xi - 0.5
        out[1, 0] = 1.0 * xi + 0.5
        out[2, 0] = -2.0 * xi

    @property
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        return 'Line3'

    @property
    def no_of_nodes(self):
        """
        Returns the number of nodes of the shape function.

        This is typically the length of the array that must be passed
        to the evaluate function.

        Returns
        -------
        no_of_nodes : int
            Number of nodes.
        """
        return 3

    @staticmethod
    def xi_lower():
        """
        Returns the lowest value of coordinate xi for integration over
        the shape-function's domain.

        Returns
        -------
        xi_lower : float
            Lowest value of coordinate xi for integration over the
            shape-function's domain.
        """
        return -1.0

    @staticmethod
    def xi_upper():
        """
        Returns the upper value of coordinate xi for integration over
        the shape-function's domain.

        Returns
        -------
        xi_upper : float
            Upper value of coordinate xi for integration over the
            shape-function's domain.
        """
        return 1.0
