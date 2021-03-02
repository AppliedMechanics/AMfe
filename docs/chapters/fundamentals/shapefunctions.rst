Shapefunctions
==============


The Shapefunctions module includes shapefunctions that are implemented in AMfe.

The base module provides for each spatial dimension (1D, 2D, 3D) base classes for the shapefunctions
that describe their interface.
Special submodules provide different implementations of these interfaces.
For example, there is a Numpy implementation of the shapefunctions in the :py:module:`amfe.element.shapefunction.numpy`.

The creation of shapefunction objects is done by factories. The factory has two methods:

- set_shape(shape)
- create()

The shape argument is of type Shape and sets the shape of the Shapefunction.
Afterwards, a call to create() returns a Shapefunction object of the shape that was set before.

Example with Numpy implementation and Quad8 shapefunction::

    >>> from amfe.element.shapefunction import Shape, NumpyShapeFunction2DFactory
    >>> factory = NumpyShapeFunction2DFactory()
    >>> factory.set_shape(Shape.QUAD8)
    >>> shapefunction = factory.create()

The shapefunction can then be used to evaluate or to compute the Jacobian::

    >>> import numpy as np
    >>> # Allocate array to write result into
    >>> value = np.zeros(shapefunction.no_of_nodes)
    >>> value_jacobian = np.zeros((shapefunction.no_of_nodes, shapefunction.no_of_local_coordinates))
    >>> # Choose some local coordinates
    >>> xi = 0.3
    >>> eta = 0.5
    >>> # Call evaluation functions
    >>> shapefunction.evaluate(xi, eta, value)
    >>> shapefunction.jacobian(xi, eta, value_jacobian)


Furthermore one can get bounds for integration over the shapefunction domain

    >>> xi_lower = shapefunction.xi_lower()
    >>> print(xi_lower)
    -1.0

See

* :py:class:`ShapeFunction1DBase<amfe.element.shapefunction.base.ShapeFunction1DBase>`
* :py:class:`ShapeFunction2DBase<amfe.element.shapefunction.base.ShapeFunction2DBase>`
* :py:class:`ShapeFunction3DBase<amfe.element.shapefunction.base.ShapeFunction3DBase>`

class documentations for more details.
