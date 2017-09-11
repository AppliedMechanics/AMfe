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

**Craig Bampton:**

**POD:** Pass a mechanical system and get the POD basis of its u_output vectors. The returned dimension is the
dimension of the constrained system (i.e. after applying Dirichlet boundary conditions).




Reduce Mechanical Systems
-------------------------

