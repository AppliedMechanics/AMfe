Reduced Basis
=============

Compute reduced bases
---------------------

The first step to reduce models via projection is to choose or compute a reduction basis V.

This can be done by using the :py:mod:`reduced_basis` module.

.. _tab_reduced_basis_methods:

.. table:: Methods for computing reduction bases

    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Method                                                                                                                                                        | Description                                                                                                                                                    |
    +===============================================================================================================================================================+================================================================================================================================================================+
    | :py:func:`krylov_subspace(M, K, b, omega=0, no_of_moments=3, mass_orth=True)<amfe.reduced_basis.krylov_subspace>`                                             | Computes a Krylov Basis :math:`\{ (K-\omega^2 M)^{-1} b, {((K-\omega^2 M)^{-1})}^{2} b, \ldots {((K-\omega^2 M)^{-1})}^{n} b \}`                               |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:func:`vibration_modes(mechanical_system, n=10, save=False)<amfe.reduced_basis.vibration_modes>`                                                           | Computes n vibration modes (frequency omega and mode shape) with given stiffness and mass matrix K and M                                                       |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:func:`craig_bampton(M, K, b, no_of_modes=5, one_basis=True)<amfe.reduced_basis.craig_bampton>`                                                            | Computes the reduction basis for craig bampton substructuring method with fixed interface modes. The interface dofs are selected via passed Boolean b vector   |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:func:`pod(mechanical_system, n=None)<amfe.reduced_basis.pod>`                                                                                             | Returns the n most important POD vectors of the constrained system based on u\_output of the mechanical\_system object                                         |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:func:`modal_derivatives(V, omega, K_func, M, h=1.0, verbose=True, symmetric=True, finite_diff='central')<amfe.reduced_basis.modal_derivatives>`           | Computes modal derivatives i.e. solution to the perturbed eigenvalue problem of the linearized system around zero and                                          |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:func:`static_derivatives(V, K_func, M=None, omega=0, h=1.0, verbose=True, symmetric=True, finite_diff='central')<amfe.reduced_basis.static_derivatives>`  | Computes static derivatives i.e. solution to the perturbed eigenvalue problem with neglected inertia terms                                                     |
    +---------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+


Comments:
^^^^^^^^^

**Krylov Subspace:** If mass\_orth is set to true, the krylov subspaced is M-orthogonalized otherwise the krylov space is just
orthogonalized.

**Vibration Modes:** If save is set to true, the vibration modes will also be saved as displacement sets in
:py:attr:`u_output<amfe.mechanical_system.MechanicalSystem.u_output>` -property. The eigenfrequencies omega are stored
in corresponding :py:attr:`T_output<amfe.mechanical_system.MechanicalSystem.T_output>` value.

.. warning::

    This will delete values in u\_output that have been stored before.

One can also use the function
:py:func:`compute_modes_pardiso(mechanical_system, n=10, u_eq=None, save=False, niter_max=40, rtol=1E-14)<amfe.reduced_basis.compute_mode_pardiso>`
to compute the vibration modes. This function uses a Lanczos iteration and the pardiso solver instead

**Craig Bampton:**

**POD:** Pass a mechanical system and get the POD basis of its u_output vectors. The returned dimension is the
dimension of the constrained system (i.e. after applying Dirichlet boundary conditions).

**Modal Derivatives:**

V is a linear basis i.e. the mode shapes and omega is the vector of corresponding eigenfrequencies. K_func is a function that returns the tangential stiffness matrix dependent on u.
M is the mass matrix. The parameter h controls the step size of the finite difference scheme that is used to compute
the derivative of the tangential stiffness matrix. The optimal step size can vary for different systems.
If you have no idea which step size you should choose, try the default value first.
The verbose option defines the amount of output information the algorithm will print in command line.
If symmetric flag is set to true, the matrix of modal derivatives will be made symmetric after calculation.
The finite_diff option can be either set to 'central' for using a central finite difference scheme (recommended) or
to 'upwind' which will use an upwind finite difference scheme.

**Static Derivatives:** The use of the function is very similar to the modal derivatives.

Augment linear bases
--------------------

The function
:py:func:`augment_with_derivatives(V, theta, M=None, tol=1E-8, symm=True)<amfe.reduced_basis.augment_with_derivatives>`
can be used to easily augment a linear basis with modal or static derivatives.
In fact this function works with any basis that is stored in a three dimensional ndarray theta.

The function expects a linear basis V, a three dimensional array theta with basis vectors to augment linear basis V
and a mass matrix if one wants to M-normalize the basis.
The tol value can be passed if one wants to truncate not important vectors by viewing at the singular values of the
matrix of all basis vectors stored in its columns. All vectors whose singular values are lesser than tol times the
largest singular value are truncated.

The function returns the augmented basis V.


Using weighting
^^^^^^^^^^^^^^^

One can use a weighting matrix W to choose the most important modal or static derivatives to augment the linear basis.
A good measure for this is :math:`W_{ij} = 1/(\omega_i \omega_j)` for example.
After the weighting matrix has been defined, the function
:py:func:`augment_with_ranked_derivatives(V, Theta, W, n, tol=1E-8, symm=True)<amfe.reduced_basis.augment_with_ranked_derivatives>`
augments the linear basis V with weighted modal/static derivatives Theta.
It returns the augmented basis with n modal/static derivates added to it.

.. todo::

    This method does not M-normalize the vectors



Reduce Mechanical Systems
-------------------------

To reduce a mechanical system by using an arbitrary basis V (of dimension of the constrained system after Dirichlet
boundary conditions are applied), call::

    >>> reduce_mechanical_system(mechanical_system, V, overwrite=False, assembly='indirect')

