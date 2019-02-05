#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
from scipy.spatial.distance import cdist

from amfe.linalg.tools import coordinate_transform
from amfe.parametric.morphing.implementer import MorpherImplementer

__all__ = ['RbfMorpherImplementer'
           ]


class RbfMorpherImplementer(MorpherImplementer):
    """
    Implements the RBF morphing technique
    """

    @staticmethod
    def gaussian_spline(X, r):
        """
        It implements the following formula:
        .. math::
            \\varphi(\\boldsymbol{x}) = e^{-\\frac{\\boldsymbol{x}^2}{r^2}}
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """
        result = np.exp(-(X * X) / (r * r))
        return result

    @staticmethod
    def multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:
        .. math::
            \\varphi(\\boldsymbol{x}) = \\sqrt{\\boldsymbol{x}^2 + r^2}
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """
        result = np.sqrt((X * X) + (r * r))
        return result

    @staticmethod
    def inv_multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:
        .. math::
            \\varphi(\\boldsymbol{x}) =
            (\\boldsymbol{x}^2 + r^2 )^{-\\frac{1}{2}}
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """
        result = 1.0 / (np.sqrt((X * X) + (r * r)))
        return result

    @staticmethod
    def thin_plate_spline(X, r):
        """
        It implements the following formula:
        .. math::
            \\varphi(\\boldsymbol{x}) =
            \\left(\\frac{\\boldsymbol{x}}{r}\\right)^2
            \\ln\\frac{\\boldsymbol{x}}{r}
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        result = arg * arg
        result = np.where(arg > 0, result * np.log(arg), result)
        return result

    @staticmethod
    def beckert_wendland_c2_basis(X, r):
        """
        It implements the following formula:
        .. math::
            \\varphi(\\boldsymbol{x}) =
            \\left( 1 - \\frac{\\boldsymbol{x}}{r}\\right)^4 +
            \\left( 4 \\frac{ \\boldsymbol{x} }{r} + 1 \\right)
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        first = np.where((1 - arg) > 0, np.power((1 - arg), 4), 0)
        second = (4 * arg) + 1
        result = first * second
        return result

    def polyharmonic_spline(self, X, r):
        """
        It implements the following formula:
        .. math::

            \\varphi(\\boldsymbol{x}) =
                \\begin{cases}
                \\frac{\\boldsymbol{x}}{r}^k
                    \\quad & \\text{if}~k = 1,3,5,...\\\\
                \\frac{\\boldsymbol{x}}{r}^{k-1}
                \\ln(\\frac{\\boldsymbol{x}}{r}^
                {\\frac{\\boldsymbol{x}}{r}})
                    \\quad & \\text{if}~\\frac{\\boldsymbol{x}}{r} < 1,
                    ~k = 2,4,6,...\\\\
                \\frac{\\boldsymbol{x}}{r}^k
                \\ln(\\frac{\\boldsymbol{x}}{r})
                    \\quad & \\text{if}~\\frac{\\boldsymbol{x}}{r} \\ge 1,
                    ~k = 2,4,6,...\\\\
                \\end{cases}
        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.
        :return: result: the result of the formula above.
        :rtype: float
        """

        k = self.power
        r_sc = X / r

        # k odd
        if k & 1:
            return np.power(r_sc, k)

        print(r_sc)
        # k even
        result = np.where(r_sc < 1,
                          np.power(r_sc, k - 1) * np.log(np.power(r_sc, r_sc)),
                          np.power(r_sc, k) * np.log(r_sc))
        return result

    def __init__(self, basis='multi_quadratic_biharmonic_spline',
                 radius=0.1):
        super().__init__()
        self.power = 3
        self.radius = radius
        self.weights = None

        self.BASES = {
            'gaussian_spline':
                self.gaussian_spline,
            'multi_quadratic_biharmonic_spline':
                self.multi_quadratic_biharmonic_spline,
            'inv_multi_quadratic_biharmonic_spline':
                self.inv_multi_quadratic_biharmonic_spline,
            'thin_plate_spline':
                self.thin_plate_spline,
            'beckert_wendland_c2_basis':
                self.beckert_wendland_c2_basis,
            'polyharmonic_spline':
                self.polyharmonic_spline
        }

        # to make the str callable we have to use a dictionary with all the
        # implemented radial basis functions
        if basis in self.BASES:
            self.basis = self.BASES[basis]
        else:
            raise NameError(
                """The name of the basis function in the parameters file is not
                correct or not implemented. Check the documentation for
                all the available functions.""")

    def offline(self, nodes_reference):
        """
        This method can be called once to initialize the morpher and let it calculate variables that must be calculated
        once

        Parameters
        ----------
        nodes_reference : ndarray
            ndarray with node coordinates (rows = nodes, columns = x,y,z coordinate)

        Returns
        -------
        None
        """
        pass

    def morph(self, nodes_original, control_points_before, control_points_after, n, callback=None):
        """

        Parameters
        ----------
        args[0] : ndarray
            node coordinates
        args[1:] : unknown
            morpher specific parameters

        Returns
        -------
        morphed nodes : ndarray
            node coordinates of the morphed nodes
        """
        delta_control_points = control_points_after - control_points_before
        n_mesh_points, dim = nodes_original.shape
        n_control_points = control_points_before.shape[0]
        H = np.zeros((n_mesh_points, n_control_points + dim + 1))

        for i in range(n):
            control_points_after_current_iteration = control_points_before + 1/n*delta_control_points
            self.weights = self._get_weights(
                control_points_before,
                control_points_after_current_iteration)

            H[:, :n_control_points] = self.basis(
                cdist(nodes_original,
                      control_points_before),
                self.radius)
            H[:, n_control_points] = 1.0
            H[:, -dim:] = nodes_original
            nodes_original = np.dot(H, self.weights)
            control_points_before = control_points_after_current_iteration
            if callback is not None:
                callback(i+1, nodes_original)
        return nodes_original

    def _get_weights(self, X, Y):
        """
        This private method, given the original control points and the deformed
        ones, returns the matrix with the weights and the polynomial terms, that
        is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
        (n_control_points+1+3)-by-3.
        :param numpy.ndarray X: it is an n_control_points-by-3 array with the
            coordinates of the original interpolation control points before the
            deformation.
        :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
            coordinates of the interpolation control points after the
            deformation.
        :return: weights: the matrix with the weights and the polynomial terms.
        :rtype: numpy.matrix
        """
        n_points, dim = X.shape
        H = np.zeros((n_points + dim + 1, n_points + dim + 1))
        H[:n_points, :n_points] = self.basis(
            cdist(X, X), self.radius)
        H[n_points, :n_points] = 1.0
        H[:n_points, n_points] = 1.0
        H[:n_points, -dim:] = X
        H[-dim:, :n_points] = X.T

        rhs = np.zeros((n_points + dim + 1, dim))
        rhs[:n_points, :] = Y
        weights = np.linalg.solve(H, rhs)
        return weights
