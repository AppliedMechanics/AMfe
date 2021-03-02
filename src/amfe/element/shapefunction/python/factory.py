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
.. versionadded:: 1.2.0

This module implements a Factory for creating NumpyShapeFunction Objects
"""

from amfe.element.shapefunction.base import (Shape,
                                             ShapeFunction1DFactoryBase,
                                             ShapeFunction2DFactoryBase,
                                             ShapeFunction3DFactoryBase)
from amfe.element.shapefunction.python.hexa8 import PythonHexa8ShapeFunction3D
from amfe.element.shapefunction.python.hexa20 import PythonHexa20ShapeFunction3D
from amfe.element.shapefunction.python.line2 import PythonLine2ShapeFunction1D
from amfe.element.shapefunction.python.line3 import PythonLine3ShapeFunction1D
from amfe.element.shapefunction.python.quad4 import PythonQuad4ShapeFunction2D
from amfe.element.shapefunction.python.quad8 import PythonQuad8ShapeFunction2D
from amfe.element.shapefunction.python.tet4 import PythonTet4ShapeFunction3D
from amfe.element.shapefunction.python.tet10 import PythonTet10ShapeFunction3D
from amfe.element.shapefunction.python.tri3 import PythonTri3ShapeFunction2D
from amfe.element.shapefunction.python.tri6 import PythonTri6ShapeFunction2D


__all__ = ['ShapeFunction1DFactory',
           'ShapeFunction2DFactory',
           'ShapeFunction3DFactory',
           ]


class ShapeFunction1DFactory(ShapeFunction1DFactoryBase):
    """
    This class implements a factory to create NumpyShapeFunction1D objects
    """
    _shape2shapefunctions = {Shape.LINE2: PythonLine2ShapeFunction1D,
                             Shape.LINE3: PythonLine3ShapeFunction1D
                             }

    def __init__(self):
        """
        Constructor of NumpyShapeFunctionFactory.
        """
        self._shape = None

    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction1DBase
            Shape-function object with interface of ShapeFunction1DBase
        """
        obj = self._shape2shapefunctions[self._shape]()
        return obj

    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shape-function that is to be
            returned when invoking create().

        Raises
        ------
        NotImplementedError
            If there is no Numpy implementation for the passed Shape object.
        TypeError
            If passed shape parameter has wrong type
        """
        if shape in self._shape2shapefunctions:
            self._shape = shape
        elif isinstance(shape, Shape):
            raise ValueError('This shape is not implemented as Numpy'
                             'ShapeFunction or you have used the wrong'
                             'ShapeFunction factory (wrong dimension.')
        else:
            raise TypeError('Shape argument must be of Type Shape.'
                            'import amfe.element.shapefunction.Shape')


class ShapeFunction2DFactory(ShapeFunction2DFactoryBase):
    """
    This class implements a factory to create NumpyShapeFunction2D objects
    """
    _shape2shapefunctions = {Shape.QUAD4: PythonQuad4ShapeFunction2D,
                             Shape.QUAD8: PythonQuad8ShapeFunction2D,
                             Shape.TRI3: PythonTri3ShapeFunction2D,
                             Shape.TRI6: PythonTri6ShapeFunction2D,
                             }

    def __init__(self):
        """
        Constructor of NumpyShapeFunctionFactory.
        """
        self._shape = None

    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction2DBase
            Shape-function object with interface of ShapeFunction2DBase
        """
        obj = self._shape2shapefunctions[self._shape]()
        return obj

    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shape-function that is to be
            returned when invoking create().

        Raises
        ------
        NotImplementedError
            If there is no Numpy implementation for the passed Shape object.
        TypeError
            If passed shape parameter has wrong type
        """
        if shape in self._shape2shapefunctions:
            self._shape = shape
        elif isinstance(shape, Shape):
            raise ValueError('This shape is not implemented as Numpy'
                             'ShapeFunction or you have used the wrong'
                             'ShapeFunction factory.')
        else:
            raise TypeError('Shape argument must be of Type Shape.'
                            'import amfe.element.shapefunction.Shape')


class ShapeFunction3DFactory(ShapeFunction3DFactoryBase):
    """
    This class implements a factory to create NumpyShapeFunction3D objects
    """
    _shape2shapefunctions = {Shape.HEXA20: PythonHexa20ShapeFunction3D,
                             Shape.HEXA8: PythonHexa8ShapeFunction3D,
                             Shape.TET10: PythonTet10ShapeFunction3D,
                             Shape.TET4: PythonTet4ShapeFunction3D,
                             }

    def __init__(self):
        """
        Constructor of NumpyShapeFunctionFactory.
        """
        self._shape = None

    def create(self):
        """
        Returns a shape-function object

        Returns
        -------
        obj : ShapeFunction3DBase
            Shape-function object with interface of ShapeFunction3DBase
        """
        obj = self._shape2shapefunctions[self._shape]()
        return obj

    def set_shape(self, shape):
        """
        Set the shape of the shape-function to build.

        Parameters
        ----------
        shape : Shape
            Shape constant of type Shape
            indicating the shape of the shape-function that is to be
            returned when invoking create().

        Raises
        ------
        NotImplementedError
            If there is no Numpy implementation for the passed Shape object.
        TypeError
            If passed shape parameter has wrong type
        """
        if shape in self._shape2shapefunctions:
            self._shape = shape
        elif isinstance(shape, Shape):
            raise ValueError('This shape is not implemented as Numpy'
                             'ShapeFunction or you have used the wrong'
                             'ShapeFunction factory.')
        else:
            raise TypeError('Shape argument must be of Type Shape.'
                            'import amfe.element.shapefunction.Shape')
