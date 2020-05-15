Solving
=======


Calling Sparse Solver
---------------------

The main method to solve a linear sparse sytem is
:py:func:`solve_sparse(A, b, matrix_type, verbose)<amfe.solver.solve_sparse>`.
Many other methods of the solver module use this method to solve linear systems of equations.
It returns the solution to A x = b and expects two to four parameters:

- A is the sparse matrix of the linear equation A x = b
- b is the right hand side vector
- matrix_type specifies the type of the matrix to choose the best solver for this type of linear equation. The following values are possible:

  - 'symm' (default): for symmetric matrices
  - 'spd': for symmetric positive definite matrices
  - 'unsymm': for not symmetric matrices (the most general case)

- verbose is a flag if the solver prints some information during solve process. It can be set to true or false (default).


This method chooses the best or fastest solver that is available. If pardiso mkl solver and its API is installed
properly, it chooses this solver. Otherwise it uses the sparse solver from scipy package.
If the matrix A which is passed is not sparse, the method will use a not sparse solver instead. Thus this method is
very flexible.


Solving static problems
-----------------------

To solve a static problem that is defined by a MechanicalSystem object, pass this object to a static solver routine
of the solver module.
You can use one of those functions:

- :py:func:`solve_nonlinear_displacement(mechanical_system, t)<amfe.solver.solve_nonlinear_displacement>`
- :py:func:`solve_linear_displacement(mechanical_system, t)<amfe.solver.solve_linear_displacement>`


The first function solves the nonlinear static problem

.. math::

    f_{int}(u) = f_{ext}(t=t)


The second function solves the linearized static problem

.. math::

    K(u=0)\ u = f_{ext}(t=t) - f_{int}(u=0)


Solving transient response
--------------------------

One can use one of two different time integration schemes to solve transient response:

1. Newmark scheme
2. Generalized Alpha scheme.

There is a linear and a nonlinear time integrator for each scheme:

.. _tab_solver_time_integrators:

.. table:: Time Integrators

    +----------------------------+-------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
    | Scheme                     | Nonlinear system                                                                    | Linearized system                                                              |
    +============================+=====================================================================================+================================================================================+
    | Gen. Alpha                 | :py:func:`integrate_nonlinear_gen_alpha<amfe.solver.integrate_nonlinear_gen_alpha>` | :py:func:`integrate_linear_gen_alpha<amfe.solver.integrate_linear_gen_alpha>`  |
    +----------------------------+-------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+
    | Newmark                    | :py:func:`integrate_linear_gen_alpha<amfe.solver.integrate_linear_gen_alpha>`       | :py:func:`integrate_linear_system<amfe.solver.integrate_linear_system>`        |
    +----------------------------+-------------------------------------------------------------------------------------+--------------------------------------------------------------------------------+

The main parameters which have to be passed to the time integrators are

- mechanical_system: A MechanicalSystem object that defines the problem to solve
- q0: The start vector (numpy array) of the displacements for time integration
- dq0: The start vector (numpy array) of the velocities for time integration
- time_range: time_steps where the results have to be saved for
- dt: step size for time integrator


Other parameters of the different time integration schemes can be looked up in their reference documentation.

.. todo::

    Show example here


The SpSolve class
-----------------

.. note::

    The SpSolve class is just a helper class for accessing the API of MKL Pardiso solver.
    You do not need to deal with this class except you know what you are doing.
    Usually you just use the
    :py:func:`solve_sparse<amfe.solver.solve_sparse>`-function described above in the first section.
    This function automatically instantiates SpSolve objects if needed and deals with them.



The SpSolve class helps to easy access the API of the Intel MKL Pardiso Solver if available.

First instantiate an SpSolve object::

    >>> solver = SpSolve(A, matrix_type='symm', verbose=False)

By calling the constructor the passed matrix A will be factorized.
The matrix_type defines the type of matrix A and chooses the right factorization algorithm.
One can choose between

- 'symm': for symmetric matrices
- 'spd': for symmetric positive definite matrices
- 'unsymm': the general case for nonsymmetric matrices

