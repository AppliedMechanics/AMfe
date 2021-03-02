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
This module implements the Tet4 shape-function
"""

import numpy as np

from amfe.element.shapefunction.base import ShapeFunction3DBase


class PythonTet4ShapeFunction3D(ShapeFunction3DBase):
    r"""
    Tet4 Shape Function with four nodes

    .. code::

              ^ zeta
              | 4
              o_
             /|  \_
            | |     \_
            / |        \_
           |  | 1         \ 3
           /  o------------o---> eta
          |  /      _____ /
          | / _____/
          // /
        2 o--
         /
        v xi

    """
    _dndxi = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.double)

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
        out[0] = 1.0 - xi - eta - zeta
        out[1] = xi
        out[2] = eta
        out[3] = zeta

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
        return 'Tet4'

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
        return 4

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
