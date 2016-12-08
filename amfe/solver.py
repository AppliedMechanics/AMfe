'''
Module for solving static and dynamic problems.
'''

__all__ = [
           'integrate_nonlinear_system_genAlpha',
           'integrate_linear_system_genAlpha',
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

def integrate_nonlinear_system_genAlpha(mechanical_system, q_init, dq_init,
                                        time_range, delta_t, rho_inf,
                                        rtol=1.0E-9, atol=1.0E-6, verbose=True,
                                        n_iter_max=100, conv_abort=True,
                                        write_iter=True, track_niter=True):
                                        # RT -- Ch.L. -- 6. Oktober 2016
    '''
    Time integration of the non-linear second-order system using the
    gerneralized-alpha scheme.

    Parameters
    ----------
    ...
    rho_inf : float, >= 0, <= 1
        high-frequency spectral radius
    ...

    TODO

    '''
    t_clock_1 = time.time()
    iteration_info = []
    mechanical_system.clear_timesteps()

    eps = 1E-13

    alpha_m = (2*rho_inf - 1.0)/(rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25*(1.0 - alpha_m + alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f

    # initialize variables
    t = 0
    q = q_init.copy()
    dq = dq_init.copy()
    ddq = np.zeros_like(q_init)
    f_ext = np.zeros_like(q_init)
    dt = delta_t
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
            dt /= 2.0
            no_newton_convergence_flag = False
        else:
            # fit time stepsize
            if t + delta_t + eps >= time_range[time_index]:
                dt = time_range[time_index] - t
            else:
                dt = delta_t

        # save old variables
        t_old = t
        q_old = q.copy()
        dq_old = dq.copy()
        ddq_old = ddq.copy()
        f_ext_old = f_ext.copy()

        # predict new variables using old variables
        t += dt
        q += dt*dq + dt**2*(0.5 - beta)*ddq
        dq += dt*(1.0 - gamma)*ddq
        ddq *= 0

        Jac, res, f_ext = mechanical_system.Jac_and_res_genAlpha(q, dq, ddq,
                                                                 q_old, dq_old,
                                                                 ddq_old,
                                                                 f_ext_old, dt,
                                                                 t, alpha_m,
                                                                 alpha_f, beta,
                                                                 gamma)
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
            dq += gamma/(beta*dt)*delta_q
            ddq += 1.0/(beta*dt**2)*delta_q

            # update system matrices and vectors
            Jac, res, f_ext = mechanical_system.Jac_and_res_genAlpha(q, dq, ddq,
                                                                     q_old, dq_old,
                                                                     ddq_old,
                                                                     f_ext_old, dt,
                                                                     t, alpha_m,
                                                                     alpha_f, beta,
                                                                     gamma)

            res_abs = norm_of_vector(res)
            # abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
            n_iter += 1

            if verbose:
                if sp.sparse.issparse(Jac):
                    cond_nr = 0.0
                else:
                    cond_nr = np.linalg.cond(Jac)
                print('Iteration = ', n_iter,
                      ', residual = {0:6.3E}, cond. num. of Jac. = {1:6.3E}'.format(
                      res_abs, cond_nr))

            # write state
            if write_iter:
                t_write = t + dt/100*n_iter
                mechanical_system.write_timestep(t_write, q.copy())

            # catch failing Newton-Raphson iteration converge
            if n_iter > n_iter_max:
                if conv_abort:
                    print(abort_statement)
                    t_clock_2 = time.time()
                    print('Time for time marching integration = {0:6.3f} seconds.'.format(
                        t_clock_2 - t_clock_1))
                    return
                t = t_old
                q = q_old.copy()
                dq = dq_old.copy()
                f_ext = f_ext_old.copy()
                no_newton_convergence_flag = True
                break

            # end of Newton-Raphson iteration loop

        print('========== Time = ', t, ', time step = ', dt,
              ', number of iterations = ', n_iter,
              ', residual = {0:6.3E}'.format(res_abs), ' ==========\n')
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



def integrate_linear_system_genAlpha(mechanical_system, q_init, dq_init,
                                     time_range, delta_t, rho_inf):
    '''
    Time integration of the linearized second-order system using the
    gerneralized-alpha scheme.

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
    remainder = (time_steps + eps) % delta_t
    if np.any(remainder > eps*10.0):
        raise ValueError('The time step size and the time range vector do not',
                         ' fit. Make the time increments in the time_range ',
                         'vector integer multiples of delta_t.')

    alpha_m = (2*rho_inf - 1.0)/(rho_inf + 1.0)
    alpha_f = rho_inf / (rho_inf + 1.0)
    beta = 0.25*(1.0 - alpha_m + alpha_f)**2
    gamma = 0.5 - alpha_m + alpha_f

    # initialize variables, matrices and vectors
    t = 0
    q = q_init.copy()
    dq = dq_init.copy()
    # evaluate initial acceleration
    K = mechanical_system.K()
    M = mechanical_system.M()
    f_ext = mechanical_system.f_ext(q, dq, t)
    ddq = solve_sparse(M, f_ext - K @ q)
    time_index = 0
    dt = delta_t
    S = (1.0 - alpha_m)*M + dt**2*beta*(1.0 - alpha_f)*K
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
        if t + eps + delta_t >= time_range[time_index]:
            dt = time_range[time_index] - t
        else:
            dt = delta_t

        # save old variables
        q_old = q.copy()
        #dq_old = dq.copy()
        ddq_old = ddq.copy()
        f_ext_old = f_ext.copy()

        # predict new variables using old variables
        t += dt
        q += dt*dq + dt**2*(0.5 - beta)*ddq
        dq += dt*(1.0 - gamma)*ddq

        # solve system
        f_ext = mechanical_system.f_ext(q, dq, t)
        f_ext_f = (1.0 - alpha_f)*f_ext + alpha_f*f_ext_old
        ddq = S_inv.solve(
            f_ext_f - alpha_m * M @ ddq_old - (1.0 - alpha_f) * K @ q - alpha_f * K @ q_old)

        # update variables
        q += dt**2*beta*ddq
        dq += dt*gamma*ddq

        print('========== Time = ', t, ', time step = ', dt, ' ==========\n')

        # end of time step loop

    # measure integration end time
    t_clock_2 = time.time()
    print('Time for time marching integration {0:4.2f} seconds'.format(
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
    q : ndarray
        Displacement of the linear system

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
    u = linalg.spsolve(K, f_ext - f_int)
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


class HHTConstrained():
    '''
    Generalized-alpha integration scheme for constrained mechanical systems.

    The integrator solves the DAE on the direct given index, i.e. index 3 DAEs
    are solved directly without index reduction technique. In order to keep the
    algorithm stable, scaling of the equations is used to avoid instabilities
    due to small time steps. Secondly, an Augmented Lagrange Term is used to
    stabilize the algorithm.
    '''

    def __init__(self, delta_t=1E-3, alpha=0, verbose=True, n_iter_max=40):
        '''
        Initialization of the integration scheme.

        Parameters
        ----------
        delta_t : float, optional
            time step size of the time marching procedure. Default value 1E-3.
        alpha : float, optional
            damping parameter of the integration scheme.
            alpha has to be in range 0 < alpha < 0.3. Default value 0.
        verbose : bool, optional
            flag for verbose output. Default value True.
        n_iter_max : int, optional
            number of maximal iterations in the newton correction loop. If
            maximum number of iterations is reached, the iteration is aborted.
            If limit is reached, usually either the time step size is to large
            or the jacobian of the system is wrong.  Default value 40.

        Returns
        -------
        None

        References
        ----------
        Bauchau, Olivier Andre: Flexible multibody dynamics, volume 176.
        Springer Science & Business Media, 2010.

        '''
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = delta_t
        self.eps = 1E-8
        self.atol = 1E-11
        self.verbose = verbose
        self.n_iter_max = n_iter_max

        # Stuff which has to be overloaded later on:
        self.constrained_system = None
        self.s = None
        self.f_non = None


    def set_constrained_system(self, constrained_system):
        '''
        Set a constrained system for the integrator.

        Parameters
        ----------
        constrained_system : instance of class ConstrainedSystem
            Constrained system

        Returns
        -------
        None

        '''
        self.constrained_system = constrained_system
        # compute the spectral radius of the system matrices in order to receive
        # the scaling factor s:
        ndof = self.constrained_system.ndof
        q0 = sp.zeros(ndof)
        dq0 = sp.zeros(ndof)
        mr = sp.linalg.norm(self.constrained_system.M(q0, dq0), sp.inf)
        dr = sp.linalg.norm(self.constrained_system.D(q0, dq0), sp.inf)
        kr = sp.linalg.norm(self.constrained_system.K(q0, dq0), sp.inf)
        self.s = mr + self.delta_t*dr + self.delta_t**2*kr
        self.f_non = self.constrained_system.f_non


    def integrate(self, q_start, dq_start, time_range):
        '''
        Integrates the nonlinear constrained system unsing the HHT-scheme.

        Parameters
        ----------
        q_start : ndarray
            initial generalized position of the system.
        dq_start : ndarray
            initial generalized velocity of the system.
        time_range : ndarray
            Array of the time steps to be exported by the integration scheme.

        Returns
        -------
        q : ndarray
            Matrix of the generalized positions of the system corresponding
            to the time points given in the time_range array. q[i] is the
            generalized position of the i-th time step in time_range,
            i.e. time_range[i].
        dq : ndarray
            Matrix of the generalized velocities of the system corresponding
            to the time points given in time_range array.
        lambda : ndarray
            Matrix of the Lagrange multipliers associated with the constraints
            of the system. The columns of lambda correspond to the vectors of
            the time steps given in time_range.

        Examples
        --------
        TODO

        Notes
        -----
        This is a deprecated function. The constraint handling is better
        handled within the `MechanicalSystem` class.
        '''
        # some internal variables defined for better readabiltiy of code
        beta = self.beta
        gamma = self.gamma
        s = self.s
        const_sys = self.constrained_system

        ndof = const_sys.ndof
        ndof_const = const_sys.ndof_const
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros(ndof)
        lambda_ = np.zeros(ndof_const)
        res = np.zeros(ndof + ndof_const)
        S = sp.zeros((ndof+ndof_const, ndof+ndof_const))

        q_global = []
        dq_global = []
        lambda_global = []
        t = 0
        time_index = 0 # index of the timestep in the time_range array
        write_flag = False

        # catch start value 0:
        if time_range[0] < 1E-12:
            q_global.append(q)
            dq_global.append(dq)
            time_index = 1
        else:
            time_index = 0

        # predict start values for ddq:
#        ddq = linalg.spsolve(const_sys.M(q, dq), self.f_non(q))

        while time_index < len(time_range):
            # time tolerance fitting...
            if t + self.delta_t + 1E-8 >= time_range[time_index]:
                dt = time_range[time_index] - t
                if dt < 1E-8:
                    dt = 1E-7
                write_flag = True
                time_index += 1
            else:
                dt = self.delta_t

            t += dt
            # Prediction
            q += dt*dq + (1/2 - beta)*dt**2*ddq
            dq += (1-gamma)*dt*ddq
            ddq *= 0
            lambda_ *= 1 # leave it the way it was...

            # checking residual and convergence
            B = const_sys.B(q, dq, t)
            K = const_sys.K(q, dq)
            D = const_sys.D(q, dq)
            M = const_sys.M(q, dq)
            K += B.T.dot(B) * s # take care of the augmentation term
            S[:ndof, :ndof] = K + gamma/(dt*beta)*D + 1/(beta*dt**2)*M
            S[:ndof, ndof:] = s / dt**2 * B.T
            S[ndof:, :ndof] = s / dt**2 * B
            C = const_sys.C(q, dq, t)
            f_non= const_sys.f_non(q, dq)
            f_ext = const_sys.f_ext(q, dq, t)
            res[:ndof] = M.dot(ddq) + f_non - f_ext
            res[ndof:] = C * s/dt**2
            res_abs = norm_of_vector(res)

            # Newton-Correction-loop
            n_iter = 0
            while res_abs > self.eps*norm_of_vector(f_non+ f_ext):

                # solve the system
                delta_x = - linalg.spsolve(S, res)
                delta_q = delta_x[:ndof]
                delta_lambda = delta_x[ndof:]

                # update of system state
                q   += delta_q
                dq  += gamma/(beta*dt)*delta_q
                ddq += 1/(beta*dt**2)*delta_q
                lambda_ += delta_lambda

                # build jacobian
                B = const_sys.B(q, dq, t)
                K = const_sys.K(q, dq)
                D = const_sys.D(q, dq)
                M = const_sys.M(q, dq)
                K += B.T.dot(B) * s/dt**2 # take care of the augmentation term
                S[:ndof, :ndof] = K + gamma/(beta*dt)*D + 1/(beta*dt**2)*M
                S[:ndof, ndof:] = B.T * s/dt**2
                S[ndof:, :ndof] = B * s/dt**2

                # build right hand side
                C = const_sys.C(q, dq, t)
                f_non= const_sys.f_non(q, dq)
                f_ext = const_sys.f_ext(q, dq, t)
                res[:ndof] = M.dot(ddq) + f_non + s/dt**2*B.T.dot(lambda_) - f_ext
                res[ndof:] = C * s/dt**2
                res_abs = norm_of_vector(res)

                n_iter += 1
                if self.verbose:
                    print('Iteration', n_iter, 'Residual:', res_abs,
                          'Cond-Nr of S:', np.linalg.cond(S))
                if n_iter > self.n_iter_max:
                    raise Exception('Maximum number of iterations reached.'
                                    'The process will be aborted. Current time step:', t)

            print('Time:', t, 'No of iterations:', n_iter, 'residual:', res_abs)


            # Writing if necessary:
            if write_flag:
                # writing to the constrained system, if possible
                # TODO: the functionality for the constrained system to read
                # TODO: and write timesteps is not implemented yet.
                if False:# self.mechanical_system:
                    self.constrained_system.write_timestep(t, q)
                else:
                    q_global.append(q.copy())
                    dq_global.append(dq.copy())
                    lambda_global.append(lambda_.copy())
                write_flag = False
        # end of time loop


        return np.array(q_global), np.array(dq_global), np.array(lambda_global)
