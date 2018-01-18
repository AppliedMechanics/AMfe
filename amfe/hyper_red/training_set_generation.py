# -*- coding: utf-8 -*-
'''
Training set generation
'''

import numpy as np
import time
import scipy as sp
import multiprocessing as mp
import copy

from ..solver import PardisoSolver, solve_nonlinear_displacement
from ..structural_dynamics import force_norm
from ..num_exp_toolbox import apply_async


__all__ = ['compute_nskts',
           'krylov_force_subspace',
           'modal_force_subspace',
          ]

def compute_nskts(mechanical_system,
                  F_ext_max=None,
                  no_of_moments=4,
                  no_of_static_cases=8,
                  load_factor=2,
                  no_of_force_increments=20,
                  no_of_procs=None,
                  norm='impedance',
                  verbose=True,
                  force_basis='krylov'):
    '''
    Compute the Nonlinear Stochastic Krylov Training Sets (NSKTS).

    NSKTS can be used as training sets for Hyper Reduction of nonlinear systems.


    Parameters
    ----------
    mechanical_system : instance of amfe.MechanicalSystem
        Mechanical System to which the NSKTS should be computed
    F_ext_max : ndarray, optional
        Maximum external force. If None is given, 10 random samples of the
        external force in the time range t = [0,1] are computed and the maximum
        value is taken. Default value is None.
    no_of_moments : int, optional
        Number of moments consiedered in the Krylov force subspace. Default
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
    nskts_arr : ndarray
        Nonlinear Stochastic Krylov Training Sets. Every column in nskts_arr
        represents one NSKTS displacement field.

    Reference
    ---------
    Todo

    '''
    def compute_stochastic_displacements(mechanical_system, F_rand):
        '''
        Solve a static problem for the given Force F_rand

        '''
        def f_ext_monkeypatched(u, du, t):
            return F_rand * t
        f_ext_tmp = mechanical_system.f_ext
        mechanical_system.f_ext = f_ext_monkeypatched

        u_arr = solve_nonlinear_displacement(mechanical_system,
                                             no_of_load_steps=no_of_force_increments,
                                             n_max_iter=no_of_force_increments,
                                             verbose=verbose,
                                             conv_abort=True,
                                             save=False)

        mechanical_system.f_ext = f_ext_tmp
        return u_arr

    print('*'*80)
    print('Start computing nonlinear stochastic ' +
          '{} training sets.'.format(force_basis))
    print('*'*80)
    time_1 = time.time()
    K = mechanical_system.K()
    M = mechanical_system.M()
    ndim = K.shape[0]
    u = du = np.zeros(ndim)
    if F_ext_max is None:
        F_ext_max = 0
        # compute the maximum external force
        for i in range(10):
            F_tmp = mechanical_system.f_ext(u, du, np.random.rand())
            if np.linalg.norm(F_tmp) > np.linalg.norm(F_ext_max):
                F_ext_max = F_tmp

    if force_basis == 'krylov':
        F_basis = krylov_force_subspace(M, K, F_ext_max,
                                        no_of_moments=no_of_moments,
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

    # Do the parallel run
    with mp.Pool(processes=no_of_procs) as pool:
        results = []
        for i in range(no_of_static_cases):
            F_rand = F_basis @ np.random.normal(0, standard_deviation)
            vals = [copy.deepcopy(mechanical_system), F_rand.copy()]
            res = apply_async(pool, compute_stochastic_displacements, vals)
            results.append(res)
        u_list = []
        for res in results:
            u = res.get()
            u_list.append(u)

    snapshot_arr = np.concatenate(u_list, axis=1)
    time_2 = time.time()
    print('Finished computing nonlinear stochastic krylov training sets.')
    print('It took {0:2.2f} seconds to build the nskts.'.format(time_2 - time_1))
    return snapshot_arr


def krylov_force_subspace(M, K, b, omega=0, no_of_moments=3,
                          orth='euclidean'):
    '''
    Compute a krylov force subspace for the computation of snapshots needed in
    hyper reduction.

    The Krylov force basis is given as

    ..code::

        [b, M @ inv(K - omega**2) @ b, ...,
         (M @ inv(K - omega**2))**(no_of_moments-1) @ b]

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
    no_of_moments : int, optional
        number of moments matched. Default value 3.
    orth : str, {'euclidean', 'impedance', 'kinetic'} optional
        flag for setting orthogonality of returnd Krylov basis vectors.

        * 'euclidean' : ``V.T @ V = eye``
        * 'impedance' : ``V.T @ inv(K) @ V = eye``
        * 'kinetic' : ``V.T @ inv(K).T @ M @ inv(K) @ V = eye``

    Returns
    -------
    V : ndarray
        Krylov force basis where vectors V[:,i] give the basis vectors.

    '''
    ndim = M.shape[0]
    no_of_inputs = b.size//ndim
    f = b.copy()
    V = np.zeros((ndim, no_of_moments*no_of_inputs))
    LU_object = PardisoSolver(K - omega**2 * M)
    b_new = f
    for i in np.arange(no_of_moments):
        V[:,i*no_of_inputs:(i+1)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        V[:,:(i+1)*no_of_inputs], R = sp.linalg.qr(V[:,:(i+1)*no_of_inputs],
                                                   mode='economic')
        f = V[:,i*no_of_inputs:(i+1)*no_of_inputs]
        u = LU_object.solve(f)
        b_new = M.dot(u)

    sigmas = sp.linalg.svdvals(V)

    # mass-orthogonalization of V:
    if orth == 'impedance':
        # Gram-Schmid-process
        for i in range(no_of_moments*no_of_inputs):
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u @ v)
            V[:,i] = v
            weights = u.T @ V[:,i+1:]
            V[:,i+1:] -= v.reshape((-1,1)) * weights
    if orth == 'kinetic':
        for i in range(no_of_moments*no_of_inputs):
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:,i] = v
            if i+1 < no_of_moments*no_of_inputs:
                weights = u.T @ M @ LU_object.solve(V[:,i+1:])
                V[:,i+1:] -= v.reshape((-1,1)) * weights


    LU_object.clear()
    print('Krylov force basis constructed.',
          'The singular values of the basis are', sigmas)
    return V


def modal_force_subspace(M, K, no_of_modes=3, orth='euclidean'):
    '''
    Force subspace spanned by the forces producing the vibration modes.
    '''
    lambda_, Phi = sp.sparse.linalg.eigsh(K, M=M, k=no_of_modes, sigma=0,
                                          which='LM',
                                          maxiter=100)
    V = K @ Phi

    LU_object = PardisoSolver(K)

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
            v = V[:,i]
            u = LU_object.solve(v)
            v /= np.sqrt(u.T @ M @ u)
            V[:,i] = v
            if i+1 < no_of_modes:
                weights = u.T @ M @ LU_object.solve(V[:,i+1:])
                V[:,i+1:] -= v.reshape((-1,1)) * weights

    LU_object.clear()
    print('Modal force basis constructed with orth type {}.'.format(orth))
    return V

