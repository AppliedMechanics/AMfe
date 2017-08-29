Elements
========


The Element module includes all element types that are implemented in AMfe.

It consists of two main classes:

* :py:class:`Element<amfe.element.Element>`

* :py:class:`BoundaryElement<amfe.element.BoundaryElement>`

The first class is the main class for all Elements in the problem domain.
The second class is needed for so calles boundary elements. These elements help
to apply neumann boundary conditions.

All elements are derived from these two classes.


The Element class
-----------------


Getter-Methods
^^^^^^^^^^^^^^

There are six methods that return local matrices and vectors for the assembly.
They only differ in the entity/entities they return for the element dependent
on coordinates in reference configuration (X), displacement(u) and time(t).

.. note::

    Time t is not used yet. It is inteded for future implementations that can
    also consider time-dependent materials.


The getter functions are listed below:
    
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| Method                                                                        | Return value(s)                                                                    |
+===============================================================================+====================================================================================+
| :py:meth:`k_and_f_int(self, X, u, t=0)<amfe.element.Element.k_and_f_int>`     | local tangential stiffness matrix and internal force vector                        |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| :py:meth:`k_int(self, X, u, t=0)<amfe.element.Element.k_int>`                 | local tangential stiffness matrix                                                  |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| :py:meth:`f_int(self, X, u, t=0)<amfe.element.Element.f_int>`                 | local internal force vector                                                        |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| :py:meth:`m_and_vec_int(self, X, u, t=0)<amfe.element.Element.m_and_vec_int>` | local mass matrix and local internal force vector                                  |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| :py:meth:`m_int(self, X, u, t=0)<amfe.element.Element.m_int>`                 | local mass matrix                                                                  |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+
| :py:meth:`k_f_S_E_int(self, X, u, t=0)<amfe.element.Element.k_f_S_E_int>`     | local stiffess matrix, internal force vector, stress and strain in voigt notation  |
+-------------------------------------------------------------------------------+------------------------------------------------------------------------------------+

.. note::

    **For developers:** In fact each of those methods call the method _compute_tensors() or _m_int()
    depending on which return values are asked for. Those methods calculate the
    return values and store them in the properties of the element.
    Afterwards those properties are returned by the getter methods.
    Thus, for each element one only has to implement the internal methods
    :py:meth:`_compute_tensors()<amfe.element.Element._compute_tensors>`
    and :py:meth:`_m_int()<amfe.element.Element._m_int>`
    because those methods are specific for each element type.


Implemented element-types
^^^^^^^^^^^^^^^^^^^^^^^^^

Following elements are implemented with their own internal methods for
calculating the element entities:

* Tri3
* Tri6
* Quad4
* Quad8
* Tet4
* Tet10
* Hexa8
* Hexa20
* Bar2Dlumped



Boundary Elements
-----------------

 
Following boundary elements are implemented:

* Tri3Boundary
* Tri6Boundary
* Quad4Boundary
* Quad8Boundary
* LineLinearBoundary
* LineQuadraticBoundary
           



Helper functions
----------------

There are two helper functions:

* :py:func:`scatter_matrix<amfe.element.scatter_matrix>`

* :py:func:`compute_B_matrix<amfe.element.compute_B_matrix>`

* :py:func:`f_proj_a<amfe.element.f_proj_a>`

* :py:func:`f_proj_a_shadow<amfe.element.f_proj_a_shadow>`


.. todo::
    
    Explain helper functions.
