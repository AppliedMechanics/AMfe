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
This module tests shapefunction implementations
"""


import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import sympy

from amfe.element.shapefunction.python import (ShapeFunction1DFactory,
                                               ShapeFunction2DFactory,
                                               ShapeFunction3DFactory)
from amfe.element.shapefunction import Shape


def jacobian(func, x):
    """
    Compute the jacobian of func with respect to x using a finite differences
    scheme.

    """
    h = np.sqrt(np.finfo(float).eps)
    fx = func(x).copy()
    outdim = len(fx)
    indim = len(x)
    jac = np.zeros((outdim, indim), dtype=float)
    for i in range(indim):
        x_temp = np.zeros(indim)
        x_temp[i] = 1.0
        fplus = func(x+h*x_temp).copy()
        fminus = func(x-h*x_temp).copy()
        jac[:, i] = (fplus - fminus)/(2.0*h)
    return jac


class ShapefunctionT:
    """
    Base class for Shapefunction Tests
    """
    factory = None
    order = 'C'

    @staticmethod
    def _create_element(factory, shape):
        """
        creates a shapefunction

        Parameters
        ----------
        factory : ShapeFunctionFactory
        shape : Shape

        Returns
        -------
        shapefunction : ShapeFunctionBase
            A shapefunction object
        """
        factory.set_shape(shape)
        return factory.create()

    def check_shapefunction(self, element, testcoordinates, nodepositions):
        """
        Method to check if computation of shapefunction is correct

        Parameters
        ----------
        element : ShapeFunction2DBase, ShapeFunction3DBase, ShapeFunction1DBase
            Shapefunction object to test
        testcoordinates : tuple
            tuple with local coordinates to test
        nodepositions : tuple
            tuple with reference positions of the nodes of the element in local
            coordinate system

        Returns
        -------
        """
        out = np.zeros((element.no_of_nodes,), order=self.order)
        id_before = id(out)
        args = [0.0]*element.no_of_local_coordinates
        element.evaluate(*args, out)

        # check if ids of returned and preallocated arrays are the same
        assert id(out) == id_before

        # all shape functions in sum must return 1
        for coord in testcoordinates:
            element.evaluate(*coord, out)
            all_added = np.sum(out)
            assert_allclose(all_added, 1.0)

        # Check if nodal displacements are 1.0 when element
        # testcoordinates are matched with node testcoordinates

        for i, coord in enumerate(nodepositions):
            desired = np.zeros(element.no_of_nodes)
            desired[i] = 1.0
            actual = np.zeros((element.no_of_nodes,))
            element.evaluate(*coord, actual)
            assert_array_equal(actual, desired)

    def check_jacobian(self, element, testcoordinates):
        """
        Checks if the jacobian of the shapefunction is computed correctly

        Parameters
        ----------
        element : Shapefunction
        testcoordinates : list

        Returns
        -------
        """
        outjacobian = np.zeros((element.no_of_nodes,
                                element.no_of_local_coordinates),
                               order=self.order)
        outshapefunc = np.zeros(element.no_of_nodes, order=self.order)

        def wrapped_n(x):
            element.evaluate(*x, outshapefunc)
            return outshapefunc.copy()

        for coord in testcoordinates:
            idoutjacobian_before = id(outjacobian)
            element.jacobian(*coord, outjacobian)

            # check if ids of returned and preallocated arrays are the same
            assert id(outjacobian) == idoutjacobian_before

            # check if jacobian is correctly calculated
            assert_allclose(outjacobian, jacobian(wrapped_n, coord),
                            rtol=5e-6, atol=1e-9)

    @staticmethod
    def check_with_symbols(element):
        """
        Test if shapefunction can also be used with symbols (e.g. with sympy
        symbols )

        Parameters
        ----------
        element

        Returns
        -------

        """
        no_of_coordinates = element.no_of_local_coordinates
        xi = sympy.symbols('xi{}:{}'.format(0, no_of_coordinates))
        outN = sympy.Matrix.zeros(element.no_of_nodes, 1)
        element.evaluate(*xi, outN)
        outdNdxi = sympy.Matrix.zeros(element.no_of_nodes,
                                      element.no_of_local_coordinates)
        element.jacobian(*xi, outdNdxi)

    @staticmethod
    def check_print_info(element):
        """
        Check if object shapefunction is printable

        Parameters
        ----------
        element : Shapefunction

        Returns
        -------

        """
        print(element)


class TestLine2Shapefunction(ShapefunctionT):
    """
    Tests for Line2 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a line2 element

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """
        testcoordinates = np.random.rand(5, 1) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_shapefunction_line2(self, factory):
        """
        Tests Line2 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction1DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(-1.0, ), (1.0, )]

        element = self._create_element(factory, Shape.LINE2)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_jacobian_line2(self, factory):
        """
        Test correct computation of jacobian of a Line2 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.LINE2)

        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_print_info_line2(self, factory):
        """
        Tests if Line2 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE2)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_bounds_line2(self, factory):
        """
        Test bounds of a Line2 Element

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE2)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_with_symbols_line2(self, factory):
        """
        Tests if Line2 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE2)
        self.check_with_symbols(element)


class TestLine3Shapefunction(ShapefunctionT):
    """
    Tests for Line3 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a line3 element

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """
        testcoordinates = np.random.rand(5, 1) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_shapefunction_line3(self, factory):
        """
        Tests Line3 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction1DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(-1.0, ), (1.0, ), (0.0, )]

        element = self._create_element(factory, Shape.LINE3)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_jacobian_line3(self, factory):
        """
        Test correct computation of jacobian of a Line3 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()

        element = self._create_element(factory, Shape.LINE3)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_print_info_line3(self, factory):
        """
        Tests if Line3 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE3)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_bounds_line3(self, factory):
        """
        Test bounds of a Line3 Element

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE3)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction1DFactory()])
    def test_with_symbols_line3(self, factory):
        """
        Tests if Line3 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction1DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.LINE3)
        self.check_with_symbols(element)


class TestQuad4Shapefunction(ShapefunctionT):
    """
    Tests for Quad4 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a quad4 element

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """
        testcoordinates = np.random.rand(5, 2) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_shapefunction_quad4(self, factory):
        """
        Tests Quad4 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction2DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]

        element = self._create_element(factory, Shape.QUAD4)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_jacobian_quad4(self, factory):
        """
        Test correct computation of jacobian of a Quad4 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.QUAD4)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_print_info_quad4(self, factory):
        """
        Tests if Quad4 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD4)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_bounds_quad4(self, factory):
        """
        Test bounds of a Quad4 Element

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD4)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta0 = element.eta_lower(xi0)
        assert eta0 == -1.0
        eta1 = element.eta_upper(xi1)
        assert eta1 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_with_symbols_quad4(self, factory):
        """
        Tests if Quad4 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD4)
        self.check_with_symbols(element)


class TestTri3Shapefunction(ShapefunctionT):
    """
    Tests for Tri3 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a tri3 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """
        testcoordinates = np.random.rand(5, 2) * 1.0
        testcoordinates = np.clip(testcoordinates, 0.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_shapefunction_tri3(self, factory):
        """
        Tests Tri3 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction2DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        element = self._create_element(factory, Shape.TRI3)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_jacobian_tri3(self, factory):
        """
        Test correct computation of jacobian of a Tri3 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.TRI3)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_print_info_tri3(self, factory):
        """
        Tests if Tri3 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI3)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_bounds_tri3(self, factory):
        """
        Test bounds of a Tri3 Element

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI3)
        xi0 = element.xi_lower()
        assert xi0 == 0.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xi0)
        assert eta00 == 0.0
        eta10 = element.eta_upper(xi0)
        assert eta10 == 1.0
        eta01 = element.eta_lower(xi1)
        assert eta01 == 0.0
        eta11 = element.eta_upper(xi1)
        assert eta11 == 0.0

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_with_symbols_tri3(self, factory):
        """
        Tests if Tri3 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI3)
        self.check_with_symbols(element)


class TestQuad8Shapefunction(ShapefunctionT):
    """
    Tests for Qaud4 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a Quad8 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """
        testcoordinates = np.random.rand(5, 2) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_shapefunction_quad8(self, factory):
        """
        Tests Quad8 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction2DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0),
                         (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)]
        element = self._create_element(factory, Shape.QUAD8)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_jacobian_quad8(self, factory):
        """
        Test correct computation of jacobian of a Quad8 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.QUAD8)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_print_info_quad8(self, factory):
        """
        Tests if Quad8 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD8)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_bounds_quad8(self, factory):
        """
        Test bounds of a Quad8 Element

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD8)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xi0)
        assert eta00 == -1.0
        eta10 = element.eta_upper(xi0)
        assert eta10 == 1.0
        eta01 = element.eta_lower(xi1)
        assert eta01 == -1.0
        eta11 = element.eta_upper(xi1)
        assert eta11 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_with_symbols_quad8(self, factory):
        """
        Tests if Quad8 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.QUAD8)
        self.check_with_symbols(element)


class TestTri6Shapefunction(ShapefunctionT):
    """
    Tests for Tri6 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a Tri6 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """

        testcoordinates = np.random.rand(5, 2) * 1.0
        testcoordinates = np.clip(testcoordinates, 0.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_shapefunction_tri6(self, factory):
        """
        Tests Tri6 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction2DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0),
                         (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)]
        element = self._create_element(factory, Shape.TRI6)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_jacobian_tri6(self, factory):
        """
        Test correct computation of jacobian of a Tri6 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.TRI6)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_print_info_tri6(self, factory):
        """
        Tests if Tri6 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI6)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_bounds_tri6(self, factory):
        """
        Test bounds of a Tri6 Element

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI6)
        xi0 = element.xi_lower()
        assert xi0 == 0.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xi0)
        assert eta00 == 0.0
        eta10 = element.eta_upper(xi0)
        assert eta10 == 1.0
        eta01 = element.eta_lower(xi1)
        assert eta01 == 0.0
        eta11 = element.eta_upper(xi1)
        assert eta11 == 0.0

    @pytest.mark.parametrize("factory", [ShapeFunction2DFactory()])
    def test_with_symbols_tri6(self, factory):
        """
        Tests if Tri6 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction2DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TRI6)
        self.check_with_symbols(element)


class TestHexa8Shapefunction(ShapefunctionT):
    """
    Tests for Hexa8 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a Hexa8 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """

        testcoordinates = np.random.rand(5, 3) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_shapefunction_hexa8(self, factory):
        """
        Tests Hexa8 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction3DFactory

        Returns
        -------

        """

        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(-1.0, -1.0, -1.0),
                         (1.0, -1.0, -1.0),
                         (1.0, 1.0, -1.0),
                         (-1.0, 1.0, -1.0),
                         (-1.0, -1.0, 1.0),
                         (1.0, -1.0, 1.0),
                         (1.0, 1.0, 1.0),
                         (-1.0, 1.0, 1.0)]
        element = self._create_element(factory, Shape.HEXA8)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_jacobian_hexa8(self, factory):
        """
        Test correct computation of jacobian of a Hexa8 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.HEXA8)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_print_info_hexa8(self, factory):
        """
        Tests if Hexa8 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.HEXA8)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_bounds_hexa8(self, factory):
        """
        Test bounds of a Hexa8 Element

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.HEXA8)
        xirand = np.clip(np.random.rand(2) * 2.0 - 1.0, -1.0, 1.0)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xirand[0])
        assert eta00 == -1.0
        eta10 = element.eta_upper(xirand[0])
        assert eta10 == 1.0
        eta01 = element.eta_lower(xirand[0])
        assert eta01 == -1.0
        eta11 = element.eta_upper(xirand[0])
        assert eta11 == 1.0
        zeta0 = element.zeta_lower(*xirand)
        assert zeta0 == -1.0
        zeta1 = element.zeta_upper(*xirand)
        assert zeta1 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_with_symbols_hexa8(self, factory):
        """
        Tests if Hexa8 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.HEXA8)
        self.check_with_symbols(element)


class TestTet4Shapefunction(ShapefunctionT):
    """
    Tests for Tet4 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a Tet4 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """

        testcoordinates = np.random.rand(5, 3) * 1.0
        testcoordinates = np.clip(testcoordinates, 0.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_shapefunction_tet4(self, factory):
        """
        Tests Tet4 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction3DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        element = self._create_element(factory, Shape.TET4)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_jacobian_tet4(self, factory):
        """
        Test correct computation of jacobian of a Tet4 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.TET4)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_print_info_tet4(self, factory):
        """
        Tests if Tet4 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET4)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_bounds_tet4(self, factory):
        """
        Test bounds of a Tet4 Element

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET4)
        xi0 = element.xi_lower()
        assert xi0 == 0.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xi0)
        assert eta00 == 0.0
        eta10 = element.eta_upper(xi0)
        assert eta10 == 1.0
        eta01 = element.eta_lower(xi1)
        assert eta01 == 0.0
        eta11 = element.eta_upper(xi1)
        assert eta11 == 0.0
        zeta000 = element.zeta_lower(0.0, 0.0)
        assert zeta000 == 0.0
        zeta100 = element.zeta_upper(0.0, 0.0)
        assert zeta100 == 1.0
        zeta010 = element.zeta_lower(1.0, 0.0)
        assert zeta010 == 0.0
        zeta110 = element.zeta_upper(1.0, 0.0)
        assert zeta110 == 0.0
        zeta010 = element.zeta_lower(0.0, 1.0)
        assert zeta010 == 0.0
        zeta110 = element.zeta_upper(0.0, 1.0)
        assert zeta110 == 0.0
        zeta = element.zeta_upper(0.5, 0.0)
        assert zeta == 0.5
        zeta = element.zeta_upper(0.0, 0.5)
        assert zeta == 0.5

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_with_symbols_tet4(self, factory):
        """
        Tests if Tet4 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET4)
        self.check_with_symbols(element)


class TestHexa20Shapefunction(ShapefunctionT):
    """
    Tests for Hexa20 Shapefunctions.
    """
    @staticmethod
    def _create_testcoordinates():
        """
        Creates random testcoordinates for a Hexa20 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """

        testcoordinates = np.random.rand(5, 3) * 2.0 - 1.0
        testcoordinates = np.clip(testcoordinates, -1.0, 1.0)
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_shapefunction_hexa20(self, factory):
        """
        Tests Hexa20 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction3DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates()
        nodepositions = [(-1, -1, -1),  # 0
                         (1, -1, -1),  # 1
                         (1,  1, -1),  # 2
                         (-1,  1, -1),  # 3
                         (-1, -1,  1),  # 4
                         (1, -1,  1),  # 5
                         (1,  1,  1),  # 6
                         (-1,  1,  1),  # 7
                         (0, -1, -1),  # 8
                         (1,  0, -1),  # 9
                         (0,  1, -1),  # 10
                         (-1,  0, -1),  # 11
                         (0, -1,  1),  # 12
                         (1,  0,  1),  # 13
                         (0,  1,  1),  # 14
                         (-1,  0,  1),  # 15
                         (-1, -1,  0),  # 16
                         (1, -1,  0),  # 17
                         (1,  1,  0),  # 18
                         (-1,  1,  0)]  # 19
        element = self._create_element(factory, Shape.HEXA20)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_jacobian_hexa20(self, factory):
        """
        Test correct computation of jacobian of a Hexa20 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates()
        element = self._create_element(factory, Shape.HEXA20)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_print_info_hexa20(self, factory):
        """
        Tests if Hexa20 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.HEXA20)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_bounds_hexa20(self, factory):
        """
        Test bounds of a Hexa20 Element

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        xirand = np.clip(np.random.rand(2) * 2.0 - 1.0, -1.0, 1.0)
        element = self._create_element(factory, Shape.HEXA20)
        xi0 = element.xi_lower()
        assert xi0 == -1.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xirand[0])
        assert eta00 == -1.0
        eta10 = element.eta_upper(xirand[0])
        assert eta10 == 1.0
        eta01 = element.eta_lower(xirand[0])
        assert eta01 == -1.0
        eta11 = element.eta_upper(xirand[0])
        assert eta11 == 1.0
        zeta0 = element.zeta_lower(*xirand)
        assert zeta0 == -1.0
        zeta1 = element.zeta_upper(*xirand)
        assert zeta1 == 1.0

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_with_symbols_hexa20(self, factory):
        """
        Tests if Hexa20 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.HEXA20)
        self.check_with_symbols(element)


class TestTet10Shapefunction(ShapefunctionT):
    """
    Tests for Tet10 Shapefunctions.
    """
    def _create_testcoordinates(self, factory):
        """
        Creates random testcoordinates for a Tet10 shapefunction

        Returns
        -------
        testcoordinates : ndarray
            testcoordinates (rows -> different sets, cols -> spatial direction)
        """

        testcoordinates = np.zeros((5, 3))
        element = self._create_element(factory, Shape.TET10)
        for i in range(testcoordinates.shape[0]):
            xi = np.clip(np.random.rand(), 0.0, 1.0)
            eta = np.clip((np.random.rand() * (element.eta_upper(xi)
                                               - element.eta_lower(xi))
                           + element.eta_lower(xi)),
                          element.eta_lower(xi), element.eta_upper(xi))
            zeta = np.clip((np.random.rand() * (
                        element.zeta_upper(xi, eta) -
                        element.zeta_lower(xi, eta))
                            + element.zeta_lower(xi, eta)),
                element.zeta_lower(xi, eta), element.zeta_upper(xi, eta))
            testcoordinates[i, :] = [xi, eta, zeta]
        return testcoordinates

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_shapefunction_tet10(self, factory):
        """
        Tests Tet10 shapefunction evaluation

        Parameters
        ----------
        factory : Shapefunction3DFactory

        Returns
        -------

        """
        # create 5 random test coordinate tuples:
        testcoordinates = self._create_testcoordinates(factory)
        nodepositions = [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0),
                         (0.0, 1.0, 0.0),
                         (0.0, 0.0, 1.0),
                         (0.5, 0.0, 0.0),
                         (0.5, 0.5, 0.0),
                         (0.0, 0.5, 0.0),
                         (0.5, 0.0, 0.5),
                         (0.0, 0.5, 0.5),
                         (0.0, 0.0, 0.5)]
        element = self._create_element(factory, Shape.TET10)
        self.check_shapefunction(element, testcoordinates, nodepositions)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_jacobian_tet10(self, factory):
        """
        Test correct computation of jacobian of a Tet10 Shapefunction

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        testcoordinates = self._create_testcoordinates(factory)
        element = self._create_element(factory, Shape.TET10)
        self.check_jacobian(element, testcoordinates)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_print_info_tet10(self, factory):
        """
        Tests if Tet10 Shapefunction is printable

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET10)
        self.check_print_info(element)

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_bounds_tet10(self, factory):
        """
        Test bounds of a Tet10 Element

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET10)
        xi0 = element.xi_lower()
        assert xi0 == 0.0
        xi1 = element.xi_upper()
        assert xi1 == 1.0
        eta00 = element.eta_lower(xi0)
        assert eta00 == 0.0
        eta10 = element.eta_upper(xi0)
        assert eta10 == 1.0
        eta01 = element.eta_lower(xi1)
        assert eta01 == 0.0
        eta11 = element.eta_upper(xi1)
        assert eta11 == 0.0
        zeta000 = element.zeta_lower(0.0, 0.0)
        assert zeta000 == 0.0
        zeta100 = element.zeta_upper(0.0, 0.0)
        assert zeta100 == 1.0
        zeta010 = element.zeta_lower(1.0, 0.0)
        assert zeta010 == 0.0
        zeta110 = element.zeta_upper(1.0, 0.0)
        assert zeta110 == 0.0
        zeta010 = element.zeta_lower(0.0, 1.0)
        assert zeta010 == 0.0
        zeta110 = element.zeta_upper(0.0, 1.0)
        assert zeta110 == 0.0
        zeta = element.zeta_upper(0.5, 0.0)
        assert zeta == 0.5
        zeta = element.zeta_upper(0.0, 0.5)
        assert zeta == 0.5

    @pytest.mark.parametrize("factory", [ShapeFunction3DFactory()])
    def test_with_symbols_tet10(self, factory):
        """
        Tests if Tet10 Shapefunction works with symbols (e.g. sympy symbols)

        Parameters
        ----------
        factory : ShapeFunction3DFactory

        Returns
        -------

        """
        element = self._create_element(factory, Shape.TET10)
        self.check_with_symbols(element)
