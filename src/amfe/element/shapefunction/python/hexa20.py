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
This module implements the Hexa8 shape-function
"""

from amfe.element.shapefunction.base import ShapeFunction3DBase


class PythonHexa20ShapeFunction3D(ShapeFunction3DBase):
    r"""
    Hexa20 Shape Function with twenty nodes

    .. code::

         eta        3----10----2
        ^           |\         |\
        |           | 19       | 18
        |           11  \       9  \
        ---> xi     |   7----14+---6
        \           |   |      |   |
         \          0---+-8----1   |
         v zeta     \  15      \  13
                     16 |        17|
                       \|         \|
                        4----12----5


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
        return -1.0

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
        return 1.0

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
        out[0] = (1.0 - eta) * (1.0 - xi) * (1.0 - zeta) * (
                    -eta - xi - zeta - 2.0) / 8.0
        out[1] = (1.0 - eta) * (1.0 - zeta) * (xi + 1.0) * (
                    -eta + xi - zeta - 2.0) / 8.0
        out[2] = (1.0 - zeta) * (eta + 1.0) * (xi + 1.0) * (
                    eta + xi - zeta - 2.0) / 8.0
        out[3] = (1.0 - xi) * (1.0 - zeta) * (eta + 1.0) * (
                    eta - xi - zeta - 2.0) / 8.0
        out[4] = (1.0 - eta) * (1.0 - xi) * (zeta + 1.0) * (
                    -eta - xi + zeta - 2.0) / 8.0
        out[5] = (1.0 - eta) * (xi + 1.0) * (zeta + 1.0) * (
                    -eta + xi + zeta - 2.0) / 8.0
        out[6] = (eta + 1.0) * (xi + 1.0) * (zeta + 1.0) * (
                    eta + xi + zeta - 2.0) / 8.0
        out[7] = (1.0 - xi) * (eta + 1.0) * (zeta + 1.0) * (
                    eta - xi + zeta - 2.0) / 8.0
        out[8] = (0.25 - 0.25 * xi ** 2) * (1.0 - eta) * (1.0 - zeta)
        out[9] = (1.0 - eta ** 2) * (1.0 - zeta) * (0.25 * xi + 0.25)
        out[10] = (0.25 - 0.25 * xi ** 2) * (1.0 - zeta) * (eta + 1.0)
        out[11] = (0.25 - 0.25 * xi) * (1.0 - eta ** 2) * (1.0 - zeta)
        out[12] = (0.25 - 0.25 * xi ** 2) * (1.0 - eta) * (zeta + 1.0)
        out[13] = (1.0 - eta ** 2) * (0.25 * xi + 0.25) * (zeta + 1.0)
        out[14] = (0.25 - 0.25 * xi ** 2) * (eta + 1.0) * (zeta + 1.0)
        out[15] = (0.25 - 0.25 * xi) * (1.0 - eta ** 2) * (zeta + 1.0)
        out[16] = (0.25 - 0.25 * xi) * (1.0 - eta) * (1.0 - zeta ** 2)
        out[17] = (1.0 - eta) * (1.0 - zeta ** 2) * (0.25 * xi + 0.25)
        out[18] = (1.0 - zeta ** 2) * (eta + 1.0) * (0.25 * xi + 0.25)
        out[19] = (0.25 - 0.25 * xi) * (1.0 - zeta ** 2) * (eta + 1.0)

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
        out[0, 0] = -(1.0 / 8.0 - eta / 8.0) * (1.0 - xi) * (1.0 - zeta) + (
                    1.0 - zeta) * (eta / 8.0 - 1.0 / 8.0) * (
                                -eta - xi - zeta - 2.0)
        out[0, 1] = -(1.0 / 8.0 - eta / 8.0) * (1.0 - xi) * (1.0 - zeta) + (
                    1.0 - zeta) * (xi / 8.0 - 1.0 / 8.0) * (
                                -eta - xi - zeta - 2.0)
        out[0, 2] = -(1.0 / 8.0 - eta / 8.0) * (1.0 - xi) * (1.0 - zeta) - (
                    1.0 / 8.0 - eta / 8.0) * (1.0 - xi) * (
                                -eta - xi - zeta - 2.0)
        out[1, 0] = (1.0 / 8.0 - eta / 8.0) * (1.0 - zeta) * (
                    -eta + xi - zeta - 2.0) + (1.0 - eta) * (1.0 - zeta) * (
                                xi / 8.0 + 1.0 / 8.0)
        out[1, 1] = -(1.0 - eta) * (1.0 - zeta) * (xi / 8.0 + 1.0 / 8.0) + (
                    1.0 - zeta) * (-xi / 8.0 - 1.0 / 8.0) * (
                                -eta + xi - zeta - 2.0)
        out[1, 2] = -(1.0 - eta) * (1.0 - zeta) * (xi / 8.0 + 1.0 / 8.0) - (
                    1.0 - eta) * (xi / 8.0 + 1.0 / 8.0) * (
                                -eta + xi - zeta - 2.0)
        out[2, 0] = (1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) * (xi + 1.0) + (
                    1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) * (
                                eta + xi - zeta - 2.0)
        out[2, 1] = (1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) * (xi + 1.0) + (
                    1.0 - zeta) * (xi / 8.0 + 1.0 / 8.0) * (
                                eta + xi - zeta - 2.0)
        out[2, 2] = -(1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) * (xi + 1.0) - (
                    eta / 8.0 + 1.0 / 8.0) * (xi + 1) * (eta + xi - zeta - 2.0)
        out[3, 0] = -(1.0 - xi) * (1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) + (
                    1.0 - zeta) * (-eta / 8.0 - 1.0 / 8.0) * (
                                eta - xi - zeta - 2.0)
        out[3, 1] = (1.0 / 8.0 - xi / 8) * (1.0 - zeta) * (
                    eta - xi - zeta - 2.0) + (1.0 - xi) * (1.0 - zeta) * (
                                eta / 8.0 + 1.0 / 8.0)
        out[3, 2] = -(1.0 - xi) * (1.0 - zeta) * (eta / 8.0 + 1.0 / 8.0) - (
                    1.0 - xi) * (eta / 8.0 + 1.0 / 8.0) * (
                                eta - xi - zeta - 2.0)
        out[4, 0] = -(1.0 - eta) * (1.0 - xi) * (zeta / 8.0 + 1.0 / 8.0) - (
                    1.0 - eta) * (zeta / 8.0 + 1.0 / 8.0) * (
                                -eta - xi + zeta - 2.0)
        out[4, 1] = -(1.0 - eta) * (1.0 - xi) * (zeta / 8.0 + 1.0 / 8.0) + (
                    1.0 - xi) * (-zeta / 8.0 - 1.0 / 8.0) * (
                                -eta - xi + zeta - 2.0)
        out[4, 2] = (1.0 / 8.0 - eta / 8.0) * (1.0 - xi) * (
                    -eta - xi + zeta - 2.0) + (1.0 - eta) * (1.0 - xi) * (
                                zeta / 8.0 + 1.0 / 8.0)
        out[5, 0] = (1.0 - eta) * (xi / 8.0 + 1.0 / 8.0) * (zeta + 1.0) + (
                    1.0 - eta) * (zeta / 8.0 + 1.0 / 8.0) * (
                                -eta + xi + zeta - 2.0)
        out[5, 1] = -(1.0 - eta) * (xi / 8.0 + 1.0 / 8.0) * (zeta + 1.0) - (
                    xi / 8.0 + 1.0 / 8.0) * (zeta + 1.0) * (
                                -eta + xi + zeta - 2.0)
        out[5, 2] = (1.0 - eta) * (xi / 8.0 + 1.0 / 8.0) * (zeta + 1.0) + (
                    1.0 - eta) * (xi / 8.0 + 1.0 / 8.0) * (
                                -eta + xi + zeta - 2.0)
        out[6, 0] = (eta / 8.0 + 1.0 / 8.0) * (xi + 1) * (zeta + 1.0) + (
                    eta / 8.0 + 1.0 / 8.0) * (zeta + 1.0) * (
                                eta + xi + zeta - 2.0)
        out[6, 1] = (eta / 8.0 + 1.0 / 8.0) * (xi + 1) * (zeta + 1.0) + (
                    xi / 8.0 + 1.0 / 8.0) * (zeta + 1.0) * (
                                eta + xi + zeta - 2.0)
        out[6, 2] = (eta / 8.0 + 1.0 / 8.0) * (xi + 1) * (zeta + 1.0) + (
                    eta / 8.0 + 1.0 / 8.0) * (xi + 1.0) * (
                                eta + xi + zeta - 2.0)
        out[7, 0] = -(1.0 - xi) * (eta / 8.0 + 1.0 / 8.0) * (zeta + 1.0) - (
                    eta / 8.0 + 1.0 / 8.0) * (zeta + 1.0) * (
                                eta - xi + zeta - 2.0)
        out[7, 1] = (1.0 - xi) * (eta / 8.0 + 1.0 / 8.0) * (zeta + 1.0) + (
                    1.0 - xi) * (zeta / 8.0 + 1.0 / 8.0) * (
                                eta - xi + zeta - 2.0)
        out[7, 2] = (1.0 - xi) * (eta / 8.0 + 1.0 / 8.0) * (zeta + 1.0) + (
                    1.0 - xi) * (eta / 8.0 + 1.0 / 8.0) * (
                                eta - xi + zeta - 2.0)
        out[8, 0] = -xi * (1.0 - eta) * (1.0 - zeta) / 2.0
        out[8, 1] = (0.25 - xi ** 2 / 4.0) * (zeta - 1.0)
        out[8, 2] = (0.25 - xi ** 2 / 4.0) * (eta - 1.0)
        out[9, 0] = (1.0 - eta ** 2) * (1.0 - zeta) / 4.0
        out[9, 1] = -2.0 * eta * (1.0 - zeta) * (xi / 4.0 + 0.25)
        out[9, 2] = (eta ** 2 - 1.0) * (xi / 4.0 + 0.25)
        out[10, 0] = -xi * (1.0 - zeta) * (eta + 1) / 2.0
        out[10, 1] = (0.25 - xi ** 2 / 4.0) * (1.0 - zeta)
        out[10, 2] = (0.25 - xi ** 2 / 4.0) * (-eta - 1)
        out[11, 0] = -(1.0 - eta ** 2) * (1.0 - zeta) / 4.0
        out[11, 1] = -2.0 * eta * (0.25 - xi / 4) * (1.0 - zeta)
        out[11, 2] = (0.25 - xi / 4.0) * (eta ** 2 - 1.0)
        out[12, 0] = -xi * (1.0 - eta) * (zeta + 1.0) / 2.0
        out[12, 1] = (0.25 - xi ** 2 / 4.0) * (-zeta - 1.0)
        out[12, 2] = (0.25 - xi ** 2 / 4.0) * (1.0 - eta)
        out[13, 0] = (1.0 - eta ** 2) * (zeta + 1.0) / 4.0
        out[13, 1] = -2.0 * eta * (xi / 4 + 0.25) * (zeta + 1.0)
        out[13, 2] = (1.0 - eta ** 2) * (xi / 4.0 + 0.25)
        out[14, 0] = -xi * (eta + 1) * (zeta + 1) / 2.0
        out[14, 1] = (0.25 - xi ** 2 / 4.0) * (zeta + 1.0)
        out[14, 2] = (0.25 - xi ** 2 / 4.0) * (eta + 1.0)
        out[15, 0] = -(1.0 - eta ** 2) * (zeta + 1.0) / 4.0
        out[15, 1] = -2.0 * eta * (0.25 - xi / 4.0) * (zeta + 1.0)
        out[15, 2] = (0.25 - xi / 4.0) * (1.0 - eta ** 2)
        out[16, 0] = -(1.0 - eta) * (1.0 - zeta ** 2) / 4.0
        out[16, 1] = (0.25 - xi / 4.0) * (zeta ** 2 - 1.0)
        out[16, 2] = -2.0 * zeta * (0.25 - xi / 4.0) * (1.0 - eta)
        out[17, 0] = (1.0 - eta) * (1.0 - zeta ** 2) / 4.0
        out[17, 1] = (xi / 4.0 + 0.25) * (zeta ** 2 - 1.0)
        out[17, 2] = -2 * zeta * (1.0 - eta) * (xi / 4.0 + 0.25)
        out[18, 0] = (1.0 - zeta ** 2) * (eta + 1.0) / 4.0
        out[18, 1] = (1.0 - zeta ** 2) * (xi / 4.0 + 0.25)
        out[18, 2] = -2.0 * zeta * (eta + 1.0) * (xi / 4.0 + 0.25)
        out[19, 0] = -(1.0 - zeta ** 2) * (eta + 1.0) / 4.0
        out[19, 1] = (0.25 - xi / 4) * (1.0 - zeta ** 2)
        out[19, 2] = -2.0 * zeta * (0.25 - xi / 4.0) * (eta + 1.0)

    @property
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        return 'Hexa20'

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
        return 20

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
        return -1.0

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
        return -1.0

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
        return 1.0
