'''
Module for solving static and dynamic problems.
'''

__all__ = [
           'integrate_nonlinear_gen_alpha',
           'integrate_linear_gen_alpha',
           'solve_linear_displacement',
           'solve_nonlinear_displacement',
           'give_mass_and_stiffness',
           'integrate_linear_system',
           'integrate_nonlinear_system',
           'solve_sparse',
           'SpSolve',
           ]

import time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

pardiso_msg = '''
############################### WARNING #######################################
# The fast Intel MKL library could not be used. Please install pyMKL in order
# to exploit the full speed of your computer.
###############################################################################
'''

try:
    from pyMKL import pardisoSolver
    use_pardiso = True
except:
    use_pardiso = False
    print(pardiso_msg)


def norm_of_vector(array):
    '''
    Compute the (2-)norm of a vector.

    Parameters
    ----------
    array : ndarray
        one dimensional array

    Returns
    -------
    abs : float
        2-norm of the given array.

    '''
    return np.sqrt(array.T.dot(array))

abort_statement = '''
###############################################################################
#### The current computation has been aborted. No convergence was gained
#### within the number of given iteration steps.
###############################################################################
'''

mtypes = {'spd':2,
          'symm':-2,
          'unsymm':11}

def solve_sparse(A, b, matrix_type='symm', verbose=False):
    '''
    Abstractoin of the solution of the sparse system Ax=b using the fastest
    solver available for sparse and non-sparse matrices.

    Parameters
    ----------
    A : sp.sparse.CSR
        sparse matrix in CSR-format
    b : ndarray
        right hand side of equation
    matrixd_type : {'spd', 'symm', 'unsymm'}, optional
        Specifier for the matrix type:

        - 'spd' : symmetric positive definite
        - 'symm' : symmetric indefinite, default.
        - 'unsymm' : generally unsymmetric

    Returns
    -------
    x : ndarray
        solution of system Ax=b

    Notes
    -----
    This tool uses the Intel MKL library provided by Anaconda. If the Intel MKL
    is not installed, especially for large systems the computation time can go
    crazy. To adjust the number of threads used for the computation, it is
    recommended to use the mkl-service module provided by Anaconda:

    >>> import mkl
    >>> mkl.get_max_threads()
    2
    >>> mkl.set_num_threads(1)
    >>> mkl.get_max_threads()
    1

    '''
    if sp.sparse.issparse(A):
        if use_pardiso:
            mtype = mtypes[matrix_type]
            pSolve = pardisoSolver(A, mtype=mtype, verbose=verbose)
            x = pSolve.run_pardiso(13, b)
            pSolve.clear()
        else:
            x = sp.sparse.linalg.spsolve(A, b)
    else:
        x = sp.linalg.solve(A, b)
    return x


class SpSolve():
    '''
    Solver class for solving the sparse system Ax=b for multiple right hand
    sides b using the fastest solver available, i.e. the Intel MKL Pardiso, if
    available.
    '''
    def __init__(self, A, matrix_type='symm', verbose=False):
        '''
        Parameters
        ----------
        A : sp.sparse.CSR
            sparse matrix in CSR-format
        matrixd_type : {'spd', 'symm', 'unsymm'}, optional
            Specifier for the matrix type:

        - 'spd' : symmetric positive definite
        - 'symm' : symmetric indefinite
        - 'unsymm' : generally unsymmetric

        verbose : bool
            Flag for verbosity.
        '''
        if use_pardiso:
            mtype = mtypes[matrix_type]
            self.pSolve = pardisoSolver(A, mtype=mtype, verbose=verbose)
            self.pSolve.run_pardiso(12) # Analysis and numerical factorization
        else:
            self.pSolve = sp.sparse.linalg.splu(A)

    def solve(self, b):
        '''
        Solve the system for the given right hand side b.

        Parameters
        ----------
        b : ndarray
            right hand side of equation

        Returns
        -------
        x : ndarray
            solution of the sparse equation Ax=b

        '''
        if use_pardiso:
            x = self.pSolve.run_pardiso(33, b)
        else:
            x = self.pSolve.solve(b)
        return x

    def clear(self):
        '''
        Clear the memory, if possible.
        '''
        if use_pardiso:
            self.pSolve.clear()

        return

def integrate_nonlinear_gen_alpha(mechanical_system, q0, dq0, time_range, dt,
                                  rho_inf=0.9,
                                  rtol=1.0E-9,
                                  atol=1.0E-6,
                                  verbose=False,
                                  n_iter_max=30,
                                  conv_abort=True,
                                  write_iter=False,
                                  track_niter=True):
    '''
    Time integration of the non-linear second-order system using the
    gerneralized-alpha scheme.

    Parameters
    ----------
        mechanical_system : instance of MechanicalSystem
        Instance of MechanicalSystem, which should be integrated.
    q0 : ndarray
        Start displacement.
    dq0 : ndarray
        Start velocity.
    time_range : ndarray
        Array of discrete timesteps, at which the solution is saved.
    dt : float
        Time step size of the integrator.
    rho_inf : float, optional
        high-frequency spectral radius, has to be 0 <= rho_inf <= 1. For 1 no
        damping is apparent, for 0 maximum damping is there. Default value: 0.9
    rtol : float, optional
        Relative tolerance with respect to the maximum external force for the
        Newton-Raphson iteration. Default value: 1E-8.
    atol : float, optional
        Absolute tolerance for the Newton_Raphson iteration.
        Default value: 1E-6.
    verbose : bool, optional
        Flag setting verbose output. Default: False.
    n_iter_max : int, optional
        Number of maximum iterations per Newton-Raphson-procedure. Default
        value is 30.
    conv_abort : bool, optional
        Flag setting, if time integration is aborted in the case when no
        convergence is gained in the Newton-Raphson-Loop. Default value is
        True.
    write_iter : bool, optional
        Flag setting, if every step of the Newton-Raphson iteration is written
        to the MechanicalSystem object. Useful only for debugging, when no
        convergence is gained. Default value: False.
    track_niter : bool, optional
        Flag for the iteration-count. If True, the number of iterations in the
        Newton-Raphson-Loop is counted and saved to iteration_info in the
        mechanical system.

    References
    ----------
    .. [1]  J. Chung and G. Hulbert. A time integration algorithm for
            structural dynamics with improved numerical dissipation: the
            generalized-α method.
            Journal of applied mechanics, 60(2):371–375, 1993.
    .. [2]  O. A. Bauchau: Flexible Multibody Dynamics. Springer, 2011.
            pp. 664.


    '''
    t_clock_1 = time.time()
    iteration_info = []
    mechanical_system.clear_timesteps()

    eps = 1E-13

    alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
    alpha_f = rho_inf / (rho_inf + 1)
    beta = 0.25*(1 - alpha_m + alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f

    # initialize variables
    t = 0
    q = q0.copy()
    dq = dq0.copy()
    ddq = np.zeros_like(q0)
    f_ext = np.zeros_like(q0)
    h = dt
    abs_f_ext = atol
    no_newton_convergence_flag = False
    time_index = 0

    # time step loop
    while time_index < len(time_range):

        # write output
        if t + eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, q.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # half step size if Newton-Raphson iteration did not converge
        if no_newton_convergence_flag:
            h /= 2
            no_newton_convergence_flag = False
        else:
            # fit time stepsize
            if t + dt + eps >= time_range[time_index]:
                h = time_range[time_index] - t
            else:
                h = dt

        # save old variables
        t_old = t
        q_old = q.copy()
        dq_old = dq.copy()
        ddq_old = ddq.copy()
        f_ext_old = f_ext.copy()

        # predict new variables using old variables
        t += h
        q += h*dq + h**2 * (1/2-beta)*ddq
        dq += h * (1-gamma) * ddq
        ddq *= 0

        Jac, res, f_ext = mechanical_system.gen_alpha(q, dq, ddq,
                                                      q_old, dq_old,
                                                      ddq_old, f_ext_old, h,
                                                      t, alpha_m, alpha_f,
                                                      beta, gamma)

        abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
        res_abs = norm_of_vector(res)

        # Newton-Raphson iteration loop
        n_iter = 0
        while res_abs > rtol*abs_f_ext + atol:

            if sp.sparse.issparse(Jac):
                delta_q = - solve_sparse(Jac, res)
            else:
                delta_q = - sp.linalg.solve(Jac, res)

            # update variables
            q += delta_q
            dq += gamma/(beta*h)*delta_q
            ddq += 1/(beta*h**2)*delta_q

            # update system matrices and vectors
            Jac, res, f_ext = mechanical_system.gen_alpha(q, dq, ddq,
                                                          q_old, dq_old,
                                                          ddq_old, f_ext_old,
                                                          h, t, alpha_m,
                                                          alpha_f, beta, gamma)

            res_abs = norm_of_vector(res)
            # abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
            n_iter += 1

            if verbose:
                if sp.sparse.issparse(Jac):
                    cond_nr = 0.0
                else:
                    cond_nr = np.linalg.cond(Jac)
                print(('Iteration = {0:2d}, residual = {1:6.3E}'
                      + ', cond# of Jac: {2:6.3E}').format(n_iter, res_abs,
                                                        cond_nr))

            # write state
            if write_iter:
                t_write = t + dt/100*n_iter
                mechanical_system.write_timestep(t_write, q.copy())

            # catch failing Newton-Raphson iteration converge
            if n_iter > n_iter_max:
                if conv_abort:
                    print(abort_statement)
                    t_clock_2 = time.time()
                    print('Time for time marching integration: '
                          + '{0:6.3f}s.'.format(t_clock_2 - t_clock_1))
                    return
                t = t_old
                q = q_old.copy()
                dq = dq_old.copy()
                f_ext = f_ext_old.copy()
                no_newton_convergence_flag = True
                break

            # end of Newton-Raphson iteration loop

        print(('Time: {0:2.4f}, dt: {1:1.4f}, # iterations: {2:2d}, '
              + 'res: {3:6.2E}').format(t, h, n_iter, res_abs))
        if track_niter:
            iteration_info.append((t, n_iter, res_abs))

        # end of time step loop

    # write iteration info to mechanical system
    mechanical_system.iteration_info = np.array(iteration_info)

    # measure integration end time
    t_clock_2 = time.time()
    print('Time for time marching integration {0:4.2f} seconds'.format(
          t_clock_2 - t_clock_1))
    return


def integrate_linear_gen_alpha(mechanical_system, q0, dq0, time_range, dt,
                               rho_inf=0.9):
    '''
    Time integration of the linearized second-order system using the
    gerneralized-alpha scheme.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical System which is linearized about the zero displacement.
    q0 : ndarray
        initial displacement
    dq0 : ndarray
        initial velocity
    time_range : ndarray
        array containing the time steps to be exported
    dt : float
        time step size.
    rho_inf : float, optional
        high-frequency spectral radius, has to be 0 <= rho_inf <= 1. For 1 no
        damping is apparent, for 0 maximum damping is there. Default value: 0.9

    Returns
    -------
    None

    Parameters
    ----------
    ...
    rho_inf : float, >= 0, <= 1
        high-frequency spectral radius
    ...

    TODO

    '''
    t_clock_1 = time.time()
    eps = 1.0E-13
    mechanical_system.clear_timesteps()

    # check fitting of time step size and spacing in time range
    time_steps = time_range - np.roll(time_range, 1)
    remainder = (time_steps + eps) % dt
    if np.any(remainder > eps*10.0):
        raise ValueError('The time step size and the time range vector do not'
                         + ' fit. The time increments in the time_range '
                         + 'must be integer multiples of dt.')

    alpha_m = (2*rho_inf - 1.0)/(rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25*(1.0 - alpha_m + alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f

    # initialize variables, matrices and vectors
    t = 0
    q = q0.copy()
    dq = dq0.copy()
    # evaluate initial acceleration
    K = mechanical_system.K()
    M = mechanical_system.M()
    D = mechanical_system.D()
    f_ext = mechanical_system.f_ext(q, dq, t)
    ddq = solve_sparse(M, f_ext - K @ q)
    time_index = 0
    h = dt
    S = (1-alpha_m)*M + h*gamma*(1-alpha_f)*D + h**2*beta*(1-alpha_f)*K
    S_inv = SpSolve(S, matrix_type='symm')

    # time step loop
    while time_index < len(time_range):

        # write output
        if t + eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, q.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # fit time stepsize
        if t + eps + dt >= time_range[time_index]:
            h = time_range[time_index] - t
        else:
            h = dt

        # save old variables
        q_old = q.copy()
        dq_old = dq.copy()
        ddq_old = ddq.copy()
        f_ext_old = f_ext.copy()

        # predict new variables using old variables
        t += h
        q += h*dq + h**2*(1/2-beta)*ddq
        dq += h*(1-gamma)*ddq

        # solve system
        f_ext = mechanical_system.f_ext(q, dq, t)

        ddq = S_inv.solve(+ (1-alpha_f)*f_ext + alpha_f*f_ext_old
                          - alpha_m * M @ ddq_old
                          - K @ ((1-alpha_f)*q + alpha_f*q_old)
                          - D @ ((1-alpha_f)*dq + alpha_f*dq_old)
                         )

        # update variables
        q += h**2*beta*ddq
        dq += h*gamma*ddq
        print('Time: {0:2.4f}, dt: {1:1.4f}'.format(t, h))

        # end of time step loop
    S_inv.clear()
    # measure integration end time
    t_clock_2 = time.time()
    print('Time for linear time marching integration {0:4.2f} seconds'.format(
       t_clock_2 - t_clock_1))
    return



def integrate_nonlinear_system(mechanical_system, q0, dq0, time_range, dt,
                               alpha=0.01,
                               rtol=1E-8,
                               atol=1E-6,
                               verbose=False,
                               n_iter_max=30,
                               conv_abort=True,
                               write_iter=False,
                               track_niter=False):
    '''
    Time integrate the nonlinear system using a generalized-alpha HHT-scheme.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Instance of MechanicalSystem, which should be integrated.
    q0 : ndarray
        Start displacement.
    dq0 : ndarray
        Start velocity.
    time_range : ndarray
        Array of discrete timesteps, at which the solution is saved.
    dt : float
        Time step size of the integrator.
    alpha : float, optional
        HHT-damping factor for numerical damping. If alpha=0, no numerical
        damping is introduced, if alpha=0.3, the maximum numerical damping is
        introduced. Default value: 0.01
    rtol : float, optional
        Relative tolerance with respect to the maximum external force for the
        Newton-Raphson iteration. Default value: 1E-8.
    atol : float, optional
        Absolute tolerance for the Newton_Raphson iteration.
        Default value: 1E-6.
    verbose : bool, optional
        Flag setting verbose output. Default: False.
    n_iter_max : int, optional
        Number of maximum iterations per Newton-Raphson-procedure. Default
        value is 30.
    conv_abort : bool, optional
        Flag setting, if time integration is aborted in the case when no
        convergence is gained in the Newton-Raphson-Loop. Default value is
        True.
    write_iter : bool, optional
        Flag setting, if every step of the Newton-Raphson iteration is written
        to the MechanicalSystem object. Useful only for debugging, when no
        convergence is gained. Default value: False.
    track_niter : bool, optional
        Flag for the iteration-count. If True, the number of iterations in the
        Newton-Raphson-Loop is counted and saved to iteration_info in the
        mechanical system.

    Returns
    -------
    None

    References
    ----------
    .. [1]  M. Géradin and D. J. Rixen. Mechanical vibrations: theory and
            application to structural dynamics. John Wiley & Sons, 2014.
            pp. 564.
    .. [2]  O. A. Bauchau: Flexible Multibody Dynamics. Springer, 2011.
            pp. 664.


    '''
    t_clock_1 = time.time()
    iteration_info = [] # List tracking the number of iterations
    mechanical_system.clear_timesteps()

    eps = 1E-13

    beta = 1/4*(1 + alpha)**2
    gamma = 1/2 + alpha

    # initialize starting variables
    t = 0
    q = q0.copy()
    dq = dq0.copy()
    ddq = np.zeros_like(q0)
    h = dt

#        # predict start values for ddq:
#        ddq = linalg.spsolve(self.M, self.f_non(q, t))
    abs_f_ext = atol
    no_newton_convergence_flag = False
    time_index = 0 # index of the timestep in the time_range array
    while time_index < len(time_range):

        # write output, if necessary
        if t+eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, q.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # make half the step size if newton did not converge
        if no_newton_convergence_flag:
            h /= 2
            no_newton_convergence_flag = False
        else:
            # time tolerance fitting...
            if t + dt + eps >= time_range[time_index]:
                h = time_range[time_index] - t
            else:
                h = dt


        # saving state for recovery if no convergence is gained
        t_old = t
        q_old = q.copy()
        dq_old = dq.copy()

        # Prediction using state from previous step
        t += h
        q += h*dq + (1/2-beta)*h**2*ddq
        dq += (1-gamma)*h*ddq
        ddq *= 0

        S, res, f_ext = mechanical_system.S_and_res(q, dq, ddq, h,
                                                    t, beta, gamma)
        abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
        res_abs = norm_of_vector(res)

        # Newton-Correction-loop
        n_iter = 0
        while res_abs > rtol*abs_f_ext + atol:

            if sp.sparse.issparse(S):
                delta_q = - solve_sparse(S, res)
            else:
                delta_q = - sp.linalg.solve(S, res)

            # update state variables
            q += delta_q
            dq += gamma/(beta*h)*delta_q
            ddq += 1/(beta*h**2)*delta_q

            # update system matrices and vectors
            S, res, f_ext = mechanical_system.S_and_res(q, dq, ddq, h,
                                                        t, beta, gamma)
            res_abs = norm_of_vector(res)
            # abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
            n_iter += 1

            if verbose:
                if sp.sparse.issparse(S):
                    cond_nr = 0
                    # print('Cond# cannot be determined as S is sparse.')
                else:
                    cond_nr = np.linalg.cond(S)
                print('Iteration', n_iter,
                      'Residual: {0:4.1E}, cond# of S: {1:4.2E}'.format(
                          res_abs, cond_nr))

            # write the state for every iteration in order to watch, how
            # Newton-raphson converges (or not ;-)
            if write_iter:
                t_write = t + h/100*n_iter
                mechanical_system.write_timestep(t_write, q.copy())

            # catch when the newton loop doesn't converge
            if n_iter > n_iter_max:
                if conv_abort:
                    print(abort_statement)
                    t_clock_2 = time.time()
                    print('Time for time marching integration ' +
                          '{0:4.2f} seconds'.format(t_clock_2 - t_clock_1))
                    return
                    # raise Exception('No convergence in Newton-Loop!')
                t = t_old
                q = q_old.copy()
                dq = dq_old.copy()
                no_newton_convergence_flag = True
                break


        print('Time:', t, 'h:', h, 'No of iterations:', n_iter,
              'Residual: {0:4.2E}'.format(res_abs))
        if track_niter:
            iteration_info.append((t, n_iter, res_abs))

    # glue the array of the iterations on the mechanical system
    mechanical_system.iteration_info = np.array(iteration_info)
    # end of integration time
    t_clock_2 = time.time()
    print('Time for time marching integration {0:4.2f} seconds'.format(
        t_clock_2 - t_clock_1))
    return

def integrate_linear_system(mechanical_system, q0, dq0, time_range, dt, alpha=0):
    '''
    Perform an implicit time integration of the linearized system given with
    the linear system.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical System which is linearized about the zero displacement.
    q0 : ndarray
        initial displacement
    dq0 : ndarray
        initial velocity
    time_range : ndarray
        array containing the time steps to be exported
    dt : float
        time step size.
    alpha : float
        general damping factor of the generalized-alpha method.

    Returns
    -------
    None

    Notes
    -----
    Due to round-off-errors, the internal time step width is h and is very
    close to dt, but adjusted to fit the steps exactly.

    '''
    t_clock_1 = time.time()
    print('Starting linear time integration')
    eps = 1E-12 # epsilon for floating point round off errors
    mechanical_system.clear_timesteps()


    # Check, if the time step width and the spacing in time range fit together
    time_steps = time_range - np.roll(time_range, 1)
    remainder = (time_steps + eps) % dt
    if np.any(remainder > eps*10):
        raise ValueError(
            'The time step size and the time range vector do not fit.',
            'Make the time increments in the time_range vector integer',
            'multiples of dt.')

    beta = 1/4*(1 + alpha)**2
    gamma = 1/2 + alpha
    K = mechanical_system.K()
    M = mechanical_system.M()
    D = mechanical_system.D()
    S = M + gamma * dt * D + beta * dt**2 * K
#    S_inv = sp.sparse.linalg.splu(S)
    S_inv = SpSolve(S, matrix_type='symm')
    # S_inv.solve(rhs_vec) # method to solve the system efficiently
    print('Iteration matrix successfully factorized. Starting time marching...')
    # initialization of the state variables
    t = 0
    q = q0.copy()
    dq = dq0.copy()
    # Evaluation of the initial acceleration
    f_ext = mechanical_system.f_ext(q, dq, t)
    ddq = solve_sparse(M, f_ext - K @ q)

    h = dt
    time_index = 0
    while time_index < len(time_range):
        print('Time:', t)

        if t+eps >= time_range[time_index]:
            mechanical_system.write_timestep(t, q.copy())
            time_index += 1
            if time_index == len(time_range):
                break

        # adjustment of dt
        if t+eps + dt >= time_range[time_index]:
            h = time_range[time_index] - t
        else:
            h = dt

        # update of state
        t += h
        q, dq = (q + h*dq + (1/2-beta)*h**2*ddq, dq + (1-gamma)*h*ddq)

        # Solution of system
        f_ext = mechanical_system.f_ext(q, dq, t)
        ddq = S_inv.solve(f_ext - K @ q)

        # correction of state
        q, dq = (q + beta*h**2*ddq), dq + gamma*h*ddq

    S_inv.clear()
    t_clock_2 = time.time()
    print('Time for linar time marching integration: {0:4.2f} seconds'.format(
        t_clock_2 - t_clock_1))

    return

def solve_linear_displacement(mechanical_system, t=1, verbose=True):
    '''
    Solve the linear static problem of the mechanical system and print
    the results directly to the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of the class MechanicalSystem
        Mechanical system to be solved.
    t : float
        time for the external force call in MechanicalSystem.
    verbose : bool
        Flag for verbose output.

    Returns
    -------
    None

    '''
    if verbose:
        print('Assembling force and stiffness')
    K, f_int = mechanical_system.K_and_f(t=t)
    f_ext = mechanical_system.f_ext(None, None, t)
    mechanical_system.clear_timesteps()
    mechanical_system.write_timestep(0, f_ext*0) # write zeros
    if verbose:
        print('Start solving linear static problem')
    u = solve_sparse(K, f_ext - f_int)
    mechanical_system.write_timestep(t, u)
    if verbose:
        print('Static problem solved')
    return u


def solve_nonlinear_displacement(mechanical_system, no_of_load_steps=10,
                                 t=0, rtol=1E-8, atol=1E-14, newton_damping=1,
                                 n_max_iter=1000,
                                 smplfd_nwtn_itr=1,
                                 wrt_iter=False,
                                 verbose=True,
                                 track_niter=False,
                                 conv_abort=True,
                                 ):
    '''
    Solver for the nonlinear system applied directly on the mechanical system.

    Prints the results directly to the mechanical system

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem
    no_of_load_steps : int
        Number of equally spaced load steps which are applied in order to
        receive the solution
    t : float, optional
        time for the external force call in mechanical_system
    rtol : float, optional
        Relative tolerance to external force for estimation, when a loadstep
        has converged
    atol : float, optional
        Absolute tolerance for estimation, of loadstep has converged
    newton_damping : float, optional
        Newton-Damping factor applied in the solution routine; 1 means no damping,
        0 < newton_damping < 1 means damping
    n_max_iter : int, optional
        Maximum number of interations in the Newton-Loop
    smplfd_nwtn_itr : int, optional
          Number at which the jacobian is updated; if 1, then a full newton
          scheme is applied; if very large, it's a fixpoint iteration with
          constant jacobian
    wrt_iter : bool, optional
        export every iteration step to ParaView.
    verbose : bool, optional
        print messages if necessary
    track_niter : bool, optional
        Flag for the iteration-count. If True, the number of iterations in the
        Newton-Raphson-Loop is counted and saved to iteration_info in the
        mechanical system.
    conv_abort : bool, optional
        Flag setting, if time integration is aborted in the case when no
        convergence is gained in the Newton-Raphson-Loop. Default value is
        True.


    Returns
    -------
    None

    Examples
    ---------
    TODO

    '''
    t_clock_1 = time.time()
    iteration_info = [] # List tracking the number of iterations
    mechanical_system.clear_timesteps()

    stepwidth = 1/no_of_load_steps
    K, f_int= mechanical_system.K_and_f()
    ndof = K.shape[0]
#   Does not work for reduced systems
#     ndof = mechanical_system.dirichlet_class.no_of_constrained_dofs
    u = np.zeros(ndof)
    du = np.zeros(ndof)
    mechanical_system.write_timestep(0, u) # initial write

    for t in np.arange(stepwidth, 1+stepwidth, stepwidth):

        # prediction
        K, f_int= mechanical_system.K_and_f(u, t)
        f_ext = mechanical_system.f_ext(u, du, t)
        res = - f_int + f_ext
        abs_res = norm_of_vector(res)
        abs_f_ext = np.sqrt(f_ext @ f_ext)

        # Newton-Loop
        n_iter = 0
        while (abs_res > rtol*abs_f_ext + atol) and (n_max_iter > n_iter):
            corr = solve_sparse(K, res)
            u += corr*newton_damping
            if (n_iter % smplfd_nwtn_itr) is 0:
                K, f_int = mechanical_system.K_and_f(u, t)
                f_ext = mechanical_system.f_ext(u, du, t)
            res = - f_int + f_ext
            abs_f_ext = np.sqrt(f_ext @ f_ext)
            abs_res = norm_of_vector(res)
            n_iter += 1
            if verbose:
                print('Step', t, 'Iteration #', n_iter,
                      'Residal: {0:4.2E}'.format(abs_res))
            if wrt_iter:
                mechanical_system.write_timestep(n_iter, u)
            if (n_iter >= n_max_iter) and conv_abort:
                print(abort_statement)
                t_clock_2 = time.time()
                print('Time for static solution: ' +
                      '{0:4.2f} seconds'.format(t_clock_2 - t_clock_1))
                return
        mechanical_system.write_timestep(t, u)
        # export iteration infos if wanted
        if track_niter:
            iteration_info.append((t, n_iter, abs_res))

    # glue the array of the iterations on the mechanical system
    mechanical_system.iteration_info = np.array(iteration_info)

    t_clock_2 = time.time()
    print('Time for solving nonlinear displacements: {0:4.2f} seconds'.format(
        t_clock_2 - t_clock_1))
    return

def give_mass_and_stiffness(mechanical_system):
    '''
    Determine mass and stiffness matrix of a mechanical system.

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Instance of the class MechanicalSystem

    Returns
    -------
    M : ndarray
        Mass matrix of the mechanical system
    K : ndarray
        Stiffness matrix of the mechanical system

    '''

    K = mechanical_system.K()
    M = mechanical_system.M()
    return M, K
