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
This module implements the Tet10 shape-function
"""

from amfe.element.shapefunction.base import ShapeFunction3DBase


class PythonTet10ShapeFunction3D(ShapeFunction3DBase):
    r"""
    Tet10 Shape Function with ten nodes

    .. code::

              ^ zeta
              | 4
              o_
             /|  \_   9
            | o10   \o
            / |        \_
          8 o |1     7     \3   eta
           /  o-----o------o--->
          |  /      _____ /
          |5o _____o 6
          // /
        2 o--
         /
        v xi

    """

    @staticmethod
    def eta_lower(xi):
        """
        Returns the lowest value of coordinate eta for integration over the
        shape-function's domain.

        Parameters
        ----------
        xi : float
            Value of coordinate xi at which eta_lower is evaluated.

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
            Value of coordinate xi at which eta_upper is evaluated.

        Returns
        -------
        eta_upper : float
            Upper value of coordinate eta for integration over the
            shape-function's domain.
        """
        return 1.0 - xi

    def evaluate(self, xi, eta, zeta, out):
        """
        Evaluates the shape-function at xi and eta.

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        eta : float
            Value of local coordinate eta.
        zeta : float
            Value of local coordinate zeta.
        out : array_like
            Array object in which the evaluated shape-function
            coordinates are written.
        """
        out[0] = 2.0 * eta * (eta + xi + zeta - 1.0) - eta + 2.0 * xi * (
                    eta + xi + zeta - 1.0) - xi + 2.0 * zeta * (
                         eta + xi + zeta - 1.0) - zeta + 1.0
        out[1] = xi * (2.0 * xi - 1.0)
        out[2] = eta * (2.0 * eta - 1.0)
        out[3] = zeta * (2.0 * zeta - 1.0)
        out[4] = 4.0 * xi * (-eta - xi - zeta + 1.0)
        out[5] = 4.0 * eta * xi
        out[6] = 4.0 * eta * (-eta - xi - zeta + 1.0)
        out[7] = 4.0 * xi * zeta
        out[8] = 4.0 * eta * zeta
        out[9] = 4.0 * zeta * (-eta - xi - zeta + 1.0)

    def jacobian(self, xi, eta, zeta, out):
        """
        Evaluates the jacobian of the shape-function at (xi, eta, zeta)
        with respect to (xi, eta, zeta).

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        eta : float
            Value of local coordinate eta.
        zeta : float
            Value of local coordinate zeta.
        out : array_like
            array object in which the jacobian is written.
        """
        out[0, 0] = 4.0 * eta + 4.0 * xi + 4.0 * zeta - 3.0
        out[0, 1] = 4.0 * eta + 4.0 * xi + 4.0 * zeta - 3.0
        out[0, 2] = 4.0 * eta + 4.0 * xi + 4.0 * zeta - 3.0
        out[1, 0] = 4.0 * xi - 1.0
        out[1, 1] = 0.0
        out[1, 2] = 0.0
        out[2, 0] = 0.0
        out[2, 1] = 4.0 * eta - 1.0
        out[2, 2] = 0.0
        out[3, 0] = 0.0
        out[3, 1] = 0.0
        out[3, 2] = 4.0 * zeta - 1.0
        out[4, 0] = -4.0 * eta - 8.0 * xi - 4.0 * zeta + 4.0
        out[4, 1] = -4.0 * xi
        out[4, 2] = -4.0 * xi
        out[5, 0] = 4.0 * eta
        out[5, 1] = 4.0 * xi
        out[5, 2] = 0.0
        out[6, 0] = -4.0 * eta
        out[6, 1] = -8.0 * eta - 4.0 * xi - 4.0 * zeta + 4.0
        out[6, 2] = -4.0 * eta
        out[7, 0] = 4.0 * zeta
        out[7, 1] = 0.0
        out[7, 2] = 4.0 * xi
        out[8, 0] = 0.0
        out[8, 1] = 4.0 * zeta
        out[8, 2] = 4.0 * eta
        out[9, 0] = -4.0 * zeta
        out[9, 1] = -4.0 * zeta
        out[9, 2] = -4.0 * eta - 4.0 * xi - 8.0 * zeta + 4.0

    @property
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        return 'Tet10'

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
        return 10

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

    @staticmethod
    def zeta_lower(xi, eta):
        """
        Returns the lowest value of coordinate zeta for integration over the
        shape-function's domain.

        Parameters
        ----------
        xi : float
            Value of coordinate xi at which zeta_lower is evaluated.
        eta : float
            Value of coordinate eta at which zeta_lower is evaluated.

        Returns
        -------
        zeta_lower : float
            Lowest value of coordinate zeta for integration over the
            shape-function's domain.
        """
        return 0.0

    @staticmethod
    def zeta_upper(xi, eta):
        """
        Returns the upper value of coordinate zeta for integration over the
        shape-function's domain.

        Parameters
        ----------
        xi : float
            Value of coordinate xi at which zeta_upper is evaluated.
        eta : float
            Value of coordinate eta at which zeta_upper is evaluated.

        Returns
        -------
        zeta_upper : float
            Upper value of coordinate zeta for integration over the
            shape-function's domain.
        """
        return 1.0 - xi - eta
