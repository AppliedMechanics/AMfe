#
# Copyright (c) 2020 TECHNICAL UNIVERSITY OF MUNICH,
# DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license.
# See LICENSE file for more information.
#
# AUTHOR: Christian Meyer
r"""
.. versionadded:: 1.2.0

This module implements abstract base classes for shape-functions.
Each shape-function must implement the interface of the abstract base class to
interact with other objects of amfe's
element module.

Notes about methods returning the boundaries for integration
------------------------------------------------------------

The order of integration over a shapefunction is assumed as follows:

.. math::
    \int\limits_{\xi_{\mathrm{lower}}}^{\xi_{\mathrm{upper}}}
    f(\xi)\ \mathrm{d}\xi

for one dimensional shape-functions.

.. math::
    \int\limits_{\xi_{\mathrm{lower}}}^{\xi_{\mathrm{upper}}}
    \int\limits_{\eta_{\mathrm{lower}}(\xi)}^{\eta_{\mathrm{upper}}(\xi)}
    f(\xi, \eta)\ \mathrm{d}\eta\ \mathrm{d}\xi

for two dimensional shape-functions.

.. math::
    \int\limits_{\xi_{\mathrm{lower}}}^{\xi_{\mathrm{upper}}}
    \int\limits_{\eta_{\mathrm{lower}}(\xi)}^{\eta_{\mathrm{upper}}(\xi)}
    \int\limits_{\zeta_{\mathrm{lower}}(\xi, \eta)}^{\zeta_{\mathrm{upper}}
    (\xi, \eta)}
    f(\xi, \eta, \zeta) \ \mathrm{d}\zeta \ \mathrm{d}\eta \ \mathrm{d}\xi

for three dimensional shape-functions.

"""

__all__ = ['Shape',
           'ShapeFunction1DBase',
           'ShapeFunction2DBase',
           'ShapeFunction3DBase',
           'ShapeFunction1DFactoryBase',
           'ShapeFunction2DFactoryBase',
           'ShapeFunction3DFactoryBase',
           ]

from abc import ABC, abstractmethod
from enum import Enum


class Shape(Enum):
    """
    This class defines constants to define shapes
    """
    LINE2 = 12
    LINE3 = 13
    TRI3 = 23
    QUAD4 = 24
    TRI6 = 26
    QUAD8 = 28
    TET4 = 34
    HEXA8 = 38
    TET10 = 310
    HEXA20 = 320


class ShapeFunction1DBase(ABC):
    """
    This class provides an interface for one-dimensional Shape-Functions
    """
    @abstractmethod
    def evaluate(self, xi, out):
        """
        Evaluates the shape-function at xi

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        out : array_like
            Array object in which the evaluated shape-function coordinates are
            written.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def jacobian(self, xi, out):
        """
        Evaluates the jacobian of the shape-function at xi with respect to xi.

        Parameters
        ----------
        xi : float
            Value of local coordinate xi.
        out : array_like
            Array object in which the jacobian is written.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        raise NotImplementedError

    @property
    def no_of_local_coordinates(self):
        """
        Returns the number of local coordinates of the shape function.

        Returns
        -------
        no_of_local_coordinates : int
            Number of local coordinates.
        """
        return 1

    @property
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def xi_lower():
        """
        Returns the lowest value of coordinate xi for integration over the
        shape-function's domain.

        Returns
        -------
        xi_lower : float
            Lowest value of coordinate xi for integration over
            the shape-function's domain.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


class ShapeFunction2DBase(ABC):
    """
    This class provides an interface for two-dimensional Shape-Functions
    """
    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
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

        Returns
        -------
        None
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        raise NotImplementedError

    @property
    def no_of_local_coordinates(self):
        """
        Returns the number of local coordinates of the shape-function.

        Returns
        -------
        no_of_local_coordinates : int
            Number of local coordinates.
        """
        return 2

    @property
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


class ShapeFunction3DBase(ABC):
    """
    This class provides an interface for three-dimensional Shape-Functions
    """
    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
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

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
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

        Returns
        -------
        None
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        """
        Returns a name characterizing the shape function.

        Returns
        -------
        name : str
            The name of the shape-function.
        """
        raise NotImplementedError

    @property
    def no_of_local_coordinates(self):
        """
        Returns the number of local coordinates of the shape function.

        Returns
        -------
        no_of_local_coordinates : int
            Number of local coordinates.
        """
        return 3

    @property
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError

    @staticmethod
    @abstractmethod
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
        raise NotImplementedError


class ShapeFunction1DFactoryBase(ABC):
    """
    This class provides an interface for factory classes that create
    Shape-Function objects for one dimensional shape functions
    """
    @abstractmethod
    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shapefunction that is to be
            returned when invoking create().

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction1DBase
            Shape-function object with interface of ShapeFunction1DBase.
        """
        raise NotImplementedError


class ShapeFunction2DFactoryBase(ABC):
    """
    This class provides an interface for factory classes that create
    Shape-Function objects for two dimensional shape functions
    """
    @abstractmethod
    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shapefunction that is to be
            returned when invoking create().

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction2DBase
            Shape-function object with interface of ShapeFunction2DBase.
        """
        raise NotImplementedError


class ShapeFunction3DFactoryBase(ABC):
    """
    This class provides an interface for factory classes that create
    Shape-Function objects for three dimensional shape functions
    """
    @abstractmethod
    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shapefunction that is to be
            returned when invoking create().

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction3DBase
            Shape-function object with interface of ShapeFunction1DBase.
        """
        raise NotImplementedError
