# -*- coding: utf-8 -*-
"""
Training set generation
"""

import numpy as np
import time
import scipy as sp
from scipy.sparse.linalg import LinearOperator, splu
import multiprocessing as mp

from amfe.linalg.tools import arnoldi
from amfe.structural_dynamics import force_norm
from amfe.num_exp_toolbox import apply_async
from amfe.solver.nonlinear_solver import NewtonRaphson

__all__ = ['compute_nskts',
           'krylov_force_subspace',
           'modal_force_subspace',
           ]


def compute_nskts(K, M, F_ext_max, f_int_func, K_func,
                  no_of_moments=4,
                  no_of_static_cases=8,
                  load_factor=2,
                  no_of_force_increments=20,
                  no_of_procs=None,
                  norm='impedance',
                  verbose=True,
                  force_basis='krylov'):
    """
    Compute the Nonlinear Stochastic Krylov Training Sets (NSKTS).

    NSKTS can be used as training sets for Hyper Reduction of nonlinear systems.


    Parameters
    ----------
    K : csr_matrix
        linearized stiffness matrix for which the Krylov subspace shall be computed
    M : csr_matrix
        linearized mass matrix for which the Krylov subspace shall be compute
    F_ext_max : ndarray
        External force appearing in simulation with maximum norm
    f_int_func : callable
        f_int function with signature res = f_int(x)
    K_func : callable
        K_func is the Jacobian of the f_int_func
    no_of_moments : int, optional
        Number of moments considered in the Krylov force subspace. Default
        value is 4.
    no_of_static_cases : int, optional
        Number of stochastic static cases which are solved. Default value is 8.
    load_factor : int, optional
        Load amplification factor with which the maximum external force is
        multiplied. Default value is 2.
    no_of_force_increments : int, optional
        Number of force increments for nonlinear solver. Default value is 20.
    no_of_procs : {int, None}, optional
        Number of processes which are started parallel. For None
        no_of_static_cases processes are run.
    norm : str {'impedance', 'eucledian', 'kinetic'}, optional
        Norm which will be used to scale the higher order moments for the Krylov
        force subspace. Default value is 'impedance'.
    verbose : bool, optional
        Flag for setting verbose output. Default value is True.
    force_basis : str {'krylov', 'modal'}, optional
        Type of force basis used. Either krylov meaning the classical NSKTS or
        modal meaning the forces producing vibration modes.

    Returns
    -------
    nskts_arr : numpy.array
        Nonlinear Stochastic Krylov Training Sets. Every column in nskts_arr
        represents one NSKTS displacement field.

    Reference
    ---------
    Todo

    """
    def compute_stochastic_displacements(f_int_func, jac_f_int, F_rand, x0, u_out):
        """
        Solve a static problem for the given Force F_rand

        Parameters
        ----------
        f_int_func : callable
            f_int_function of system with signature f_int(x)
        jac_f_int : callable
            Jacobian of f_int_func with signature K(x)
        F_rand : array_like
        x0 : ndarray
            start for first search direction for newton solver
        u_out : ndarray
            preallocated array to write results into
        """

        def f_ext(t):
            return F_rand * t

        for i, t in enumerate(np.arange(1/no_of_force_increments,
                                        1+1/no_of_force_increments,
                                        1/no_of_force_increments)):

            def residual(x):
                return f_int_func(x) - f_ext(t)

            nlsolver = NewtonRaphson()

            u_out[:, i], _ = nlsolver.solve(residual, x0, jac=jac_f_int, tol=1e-8*np.linalg.norm(f_ext(t)),
                                            options={'verbose': verbose})
            x0 = u_out[:, i]
        
        return u_out

    print('*'*80)
    print('Start computing nonlinear stochastic ' +
          '{} training sets.'.format(force_basis))
    print('*'*80)
    time_1 = time.time()

    ndim = K.shape[0]

    if force_basis == 'krylov':
        F_basis = krylov_force_subspace(M, K, F_ext_max, n=no_of_moments,
                                        orth=norm)
    elif force_basis == 'modal':
        F_basis = modal_force_subspace(M, K, no_of_modes=no_of_moments,
                                       orth=norm)
    else:
        raise ValueError('Force basis type ' + force_basis + 'not valid.')

    norm_of_forces = force_norm(F_ext_max, K, M, norm=norm)
    standard_deviation = np.ravel(np.array(
            [norm_of_forces for i in range(no_of_moments)]))
    standard_deviation *= load_factor

# PARALLEL IMPLEMENTATION IS NOT WORKING ANYMORE
    # Do the parallel run
    # with mp.Pool(processes=no_of_procs) as pool:
    #    results = []
#        for i in range(no_of_static_cases):
#            F_rand = F_basis @ np.random.normal(0, standard_deviation)
#            vals = [copy.deepcopy(mechanical_system), F_rand.copy()]
#            res = apply_async(pool, compute_stochastic_displacements, vals)
#            results.append(res)
#        u_list = []
#        for res in results:
#            u = res.get()
#            u_list.append(u)
# NON PARALLEL IMPLEMENTATION
    u_list = []
    u_out = np.zeros((ndim, no_of_force_increments), dtype=float)
    for i in range(no_of_static_cases):
        F_rand = F_basis @ np.random.normal(0, standard_deviation)
        u_out[:, :] = 0.0
        u_out = compute_stochastic_displacements(f_int_func, K_func, F_rand, np.zeros(ndim), u_out)
        u_list.append(u_out.copy())

    if len(u_list) > 1:
        for number, u in enumerate(u_list):
            if u.shape[0] == 0:
                del u_list[number]
        snapshot_arr = np.concatenate(u_list, axis=1)
    else:
        snapshot_arr = np.array(u_list)

    time_2 = time.time()
    print('Finished computing nonlinear stochastic krylov training sets.')
    print('It took {0:2.2f} seconds to build the nskts.'.format(time_2 - time_1))
    return snapshot_arr


def krylov_force_subspace(M, K, b, omega=0, n=3,
                          orth='euclidean'):
    """
    Compute a krylov force subspace for the computation of snapshots needed in
    hyper reduction.

    The Krylov force basis is given as

    ..code::

        [b, M @ inv(K - omega**2 M) @ b, ...,
         (M @ inv(K - omega**2 M))**(no_of_moments-1) @ b]

    Parameters
    ----------
    M : ndarray
        Mass matrix of the system.
    K : ndarray
        Stiffness matrix of the system.
    b : ndarray
        input vector of external forcing.
    omega : float, optional
        frequency for the frequency shift of the stiffness. Default value 0.
    n : int, optional
        number of moments matched. Default value 3.
    orth : str, {'euclidean', 'impedance', 'kinetic'} optional
        flag for setting orthogonality of returned Krylov basis vectors.

        * 'euclidean' : ``V.T @ V = eye``
        * 'impedance' : ``V.T @ inv(K) @ V = eye``
        * 'kinetic' : ``V.T @ inv(K).T @ M @ inv(K) @ V = eye``

    Returns
    -------
    V : ndarray
        Krylov force basis where vectors V[:,i] give the basis vectors.

    """
    ndim = M.shape[0]
    no_of_inputs = b.size // ndim
    r = b.reshape(-1, no_of_inputs).copy()
    V = np.zeros((ndim, n * no_of_inputs))
    A = K - omega ** 2 * M

    solver = splu(A)

    def matvec(v):
        result = M @ solver.solve(v)
        return result

    linear_operator = LinearOperator(shape=A.shape, matvec=matvec)

    V = arnoldi(linear_operator, r, n, Vout=V)

    sigmas = sp.linalg.svdvals(V)

    # mass-orthogonalization of V:
    if orth == 'impedance':
        # Gram-Schmid-process
        for i in range(n*no_of_inputs):
            v = V[:, i]
            u = solver.solve(v)
            v /= np.sqrt(u @ v)
            V[:, i] = v
            weights = u.T @ V[:, i+1:]
            V[:, i+1:] -= v.reshape((-1, 1)) * weights
    if orth == 'kinetic':
        for i in range(n*no_of_inputs):
            v = V[:, i]
            u = solver.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:, i] = v
            if i+1 < n*no_of_inputs:
                weights = u.T @ M @ solver.solve(V[:, i+1:])
                V[:, i+1:] -= v.reshape((-1, 1)) * weights

    print('Krylov force basis constructed.',
          'The singular values of the basis are', sigmas)
    return V


def modal_force_subspace(M, K, no_of_modes=3, orth='euclidean'):
    """
    Force subspace spanned by the forces producing the vibration modes.
    """
    lambda_, Phi = sp.sparse.linalg.eigsh(K, M=M, k=no_of_modes, sigma=0,
                                          which='LM',
                                          maxiter=100)
    V = K @ Phi

    solver = splu(K)

    if orth == 'euclidean':
        V, _ = sp.linalg.qr(V, mode='economic')

    elif orth == 'impedance':
        omega = np.sqrt(lambda_)
        V /= omega
        # Gram-Schmid-process
#        for i in range(no_of_modes):
#            v = V[:,i]
#            u = LU_object.solve(v)
#            v /= np.sqrt(u @ v)
#            V[:,i] = v
#            weights = u.T @ V[:,i+1:]
#            V[:,i+1:] -= v.reshape((-1,1)) * weights

    elif orth == 'kinetic':
        for i in range(no_of_modes):
            v = V[:, i]
            u = solver.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:, i] = v
            if i+1 < no_of_modes:
                weights = u.T @ M @ solver.solve(V[:, i+1:])
                V[:, i+1:] -= v.reshape((-1, 1)) * weights

    print('Modal force basis constructed with orth type {}.'.format(orth))
    return V

