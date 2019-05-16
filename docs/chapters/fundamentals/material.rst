Materials
=========

The material module has the following tasks:

- Define materials with certain constitutive laws and their parameters
- Calculate stresses and constitutive matrices C for a given strain tensor E


HyperelasticMaterial base class
-------------------------------

Currently only hyperelastic materials are implemented. There is a base class
called :py:class:`HyperelasticMaterial <amfe.material.HyperelasticMaterial>` which provides two methods:

- :func:`S_Sv_and_C(E) <amfe.material.HyperelasticMaterial.S_Sv_and_C>`
- :func:`S_Sv_and_C_2d(E) <amfe.material.HyperelasticMaterial.S_Sv_and_C_2d>`

The first method is for 3d problems, the second is for 2d problems.
However, the methods have the same tasks:

Both methods return three numpy-arrays.
The first array is the 2nd Piola Kirchhoff stress tensor in matrix notation.
The second array is the 2nd Piola Kirchhoff stress tensor in voigt (or vector)
notation.
The third array is the constitutive matrix C.


The Green-Lagrange strain tensor is needed as input parameter, because stresses
depend on the strains.
This parameter should be passed as matrix (numpy-array).

Example for using the methods::

    >>> import amfe
    >>> my_material = amfe.KirchhoffMaterial()
    >>> import numpy as np
    >>> E = np.array([(0.5,0.2),(0.2,0.8)])
    >>> (S,S_v,C) = my_material.S_Sv_and_C_2d(E)
    >>> print(S)
    [[  1.70769231e+11   3.23076923e+10]
     [  3.23076923e+10   2.19230769e+11]]
    >>> print(S_v)
    [  1.70769231e+11   2.19230769e+11   3.23076923e+10]
    >>> print(C)
    [[  2.30769231e+11   6.92307692e+10   0.00000000e+00]
     [  6.92307692e+10   2.30769231e+11   0.00000000e+00]
     [  0.00000000e+00   0.00000000e+00   8.07692308e+10]]


.. note::

   The returned constitutive matrix :math:`C` is a 6x6 matrix (in 3d case),
   such that :math:`S_{\mathrm{voigt}} = C \cdot E_{\mathrm{voigt}}`
   where :math:`S_{\mathrm{voigt}}` and :math:`E_{\mathrm{voigt}}` are written
   in Voigt notation.
   The returned matrix is 3x3 for 2d case.

Derived classes for different constitutive laws
-----------------------------------------------

There are different hyperelastic material models implemented. Each implemented
material model is implemented as a class which is derived from the
:py:class:`HyperelasticMaterial() <amfe.material.HyperelasticMaterial>` base class.
Therefore each material has the above mentioned methods for returning stresses
and the constitutive matrix C for a given green lagrange strain tensor.
Currently the following material-models are implemented:

- St. Venant Kirchhoff Material :py:class:`KirchhoffMaterial() <amfe.material.KirchhoffMaterial>`, :py:class:`LinearMaterial() <amfe.material.LinearMaterial>`
- Neo Hookean Material :py:class:`NeoHookean() <amfe.material.NeoHookean>`
- Mooney Rivlin Material :py:class:`MooneyRivlin() <amfe.material.MooneyRivlin>`

LinearMaterial is just an alias for :py:class:`KirchhoffMaterial() <amfe.material.KirchhoffMaterial>`

The constitutive laws for different materials can be studied in the
module-documentation.



To instantiate a material-class, one has to know which parameters are needed
to describe the contitutive relation.
You can lookup these parameters in the documentation for __init__()-methods:

- :func:`KirchhoffMaterial() <amfe.material.KirchhoffMaterial.__init__>`
- :func:`NeoHookean() <amfe.material.NeoHookean.__init__>`
- :func:`MooneyRivlin() <amfe.material.MooneyRivlin.__init__>`


As example we want to instantiate a linear Kirchhoff material. The constitutive
parameters are

* Young's modulus E
* Poisson's ratio nu
* Mass density rho
* Optional for 2d Problems: Plane stress or plane strain assumption
* For 2d problems: thickness

Example::

  import amfe
  my_material = KirchhoffMaterial(E=210e9, nu=0.3, rho=7.8e3, plane_stress=false, thickness=0.001)


The parameters are saved as attributes in the material-class.

Example::

  print(my_material.E_modulus)
  print(my_material.rho)


.. warning::

  If one wants to change a material-parameter after instantiation it is highly
  recommended to reinstantiate the material.
  Furthermore one has to update the mesh.

  Otherwise one has to change the attributes of the material-object first, e.g.::

    my_material.E_modulus = 70e9

  and afterwards update the internal variables by running the method::

    my_material._update_variables()

  However this is not recommended.
