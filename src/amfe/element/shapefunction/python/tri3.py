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
This module implements the Tri3 shape-function
"""

import numpy as np

from amfe.element.shapefunction.base import ShapeFunction2DBase


class PythonTri3ShapeFunction2D(ShapeFunction2DBase):
    r"""
    Tri3 Shape Function with three nodes

    ..code::

                ^ eta
                |
                o 3
                |\
                | .
                |  \
                |   .
                |    \
                |     .
                |      \
        --------o-------o--------->
                | 1       2        xi
                |

    """
    _dndxi = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.double)

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
        out[0] = 1.0 - xi - eta
        out[1] = xi
        out[2] = eta

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
        out[:, :] = self._dndxi[:, :]

    @property
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        return 'Tri3'

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
