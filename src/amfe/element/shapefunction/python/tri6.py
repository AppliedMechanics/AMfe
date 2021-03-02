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
This module implements the Tri6 shape-function
"""

from amfe.element.shapefunction.base import ShapeFunction2DBase


class PythonTri6ShapeFunction2D(ShapeFunction2DBase):
    r"""
    Tri6 Shape Function with six nodes

    ..code::

                ^ eta
                |
                o 3
                |\
                | .
                |  \
              6 o   o 5
                |    \
                |     .
                |      \
        --------o---o---o--------->
                | 1  4   2        xi
                |

    """

    @staticmethod
    def eta_lower(xi):
        """
        Returns the lowest value of coordinate eta for integration over the
        shape-function's domain.

        Parameters
        ----------
        xi : float
            value of coordinate xi at which eta_lower is evaluated.

        Returns
        -------
        eta_lower : float
            Lowest value of coordinate eta for integration over the
            shape-function's domain.
        """
        return 0.0

    @staticmethod
    def eta_upper(xi):
        """
        Returns the upper value of coordinate eta for integration over the
        shape-function's domain.

        Parameters
        ----------
        xi : float
            Value of coordinate xi at which eta_lower is evaluated.

        Returns
        -------
        eta_upper : float
            Lowest value of coordinate xi for integration over the
            shape-function's domain.
        """
        return 1.0 - xi

    def evaluate(self, xi, eta, out):
        """
        Evaluates the shape-function at xi and eta.

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        eta : float
            Value of local coordinate eta.
        out : array_like
            Array object in which the evaluated shape-function coordinates
            are written.
        """
        out[0] = (1.0 - xi - eta) * (1.0 - 2.0 * (xi + eta))
        out[1] = - 2.0 * xi * (0.5 - xi)
        out[2] = - 2.0 * eta * (0.5 - eta)
        out[3] = 4.0 * xi * (1.0 - xi - eta)
        out[4] = 4.0 * xi * eta
        out[5] = 4.0 * eta * (1.0 - xi - eta)

    def jacobian(self, xi, eta, out):
        """
        Evaluates the jacobian of the shape-function at (xi, eta)
        with respect to (xi, eta).

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        eta : float
            Value of local coordinate eta.
        out : array_like
            Array object in which the jacobian is written.
        """
        out[0, 0] = 4.0 * eta + 4.0 * xi - 3.0
        out[0, 1] = 4.0 * eta + 4.0 * xi - 3.0
        out[1, 0] = 4.0 * xi - 1.0
        out[1, 1] = 0.0
        out[2, 0] = 0.0
        out[2, 1] = 4.0 * eta - 1.0
        out[3, 0] = -4.0 * eta - 8.0 * xi + 4.0
        out[3, 1] = -4.0 * xi
        out[4, 0] = 4.0 * eta
        out[4, 1] = 4.0 * xi
        out[5, 0] = -4.0 * eta
        out[5, 1] = -8.0 * eta - 4.0 * xi + 4.0

    @property
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        return 'Tri6'

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
        return 6

    @staticmethod
    def xi_lower():
        """
        Returns the lowest value of coordinate xi for integration over the
        shape-function's domain.

        Returns
        -------
        xi_lower : float
            Lowest value of coordinate xi for integration over the
            shape-function's domain.
        """
        return 0.0

    @staticmethod
    def xi_upper():
        """
        Returns the upper value of coordinate xi for integration over the
        shape-function's domain.

        Returns
        -------
        xi_upper : float
            Upper value of coordinate xi for integration over the
            shape-function's domain.
        """
        return 1.0
