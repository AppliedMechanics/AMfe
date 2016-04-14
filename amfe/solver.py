'''
Module for solving static and dynamic problems.
'''

__all__ = ['NewmarkIntegrator', 'solve_linear_displacement',
            'solve_nonlinear_displacement', 'give_mass_and_stiffness',
            'HHTConstrained', 'integrate_linear_system']

import time
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg

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
#### The current time integration has been aborted. No convergence was gained 
#### within the number of given iteration steps.  
###############################################################################
'''

class NewmarkIntegrator():
    '''
    Newmark-integration scheme using generalized alpha routine for nonlinear
    second order systems

    Notes
    ------
    This integration scheme is an unconditionally stable implicit nonlinear
    integration scheme. The unconditional stabiltiy refers to the stability of
    the solution itself, not on the solutin procedure. With a too tight
    tolerance (eps > 1E8) the solution might not converge, as the solution
    procedure can not lower the residual below a threshold that's higher than
    the eps. In general the solution should converge in less than ten iteration
    steps.

    Examples
    --------
    TODO

    References
    ----------
    .. [1]  M. Géradin and D. J. Rixen. Mechanical vibrations: theory and
            application to structural dynamics. John Wiley & Sons, 2014.
            pp. 564.

    '''

    def __init__(self, mechanical_system, alpha=0.001, verbose=False, 
                 n_iter_max=30, conv_abort=True):
        '''
        Parameters
        ----------
        mechanical_system : instance of amfe.MechanicalSystem
            mechanical system equipped with an iteration matrix S and a residual.
        alpha : float, optional
            damping factor of the generalized alpha routine for numerical damping
        verbose : bool, optional
            flag for making the integration process verbose, i.e. printing the
            residual for every correction in the newton iteration
        n_iter_max : int, optional
            number of maximum iteration in the newton correction process
        conv_abort : bool, optional
            flag indicating, if the integration is aborted when no convergence 
            is gained. 

        Returns
        -------
        None

        '''
        self.beta = 1/4*(1 + alpha)**2
        self.gamma = 1/2 + alpha
        self.delta_t = 1E-3
        self.rtol = 1E-8
        self.atol = 1E-10
        self.newton_damping = 1.0
        self.verbose = verbose
        self.n_iter_max = n_iter_max
        self.mechanical_system = mechanical_system
        self.write_iter = False
        self.conv_abort = conv_abort

    def integrate(self, q_start, dq_start, time_range):
        '''
        Integrates the system using generalized alpha method.

        Parameters
        -----------
        q_start : ndarray
            initial displacement of the constrained system in voigt notation
        dq_start : ndarray
            initial velocity of the constrained system in voigt notation
        time_range : ndarray
            vector containing the time points at which the state is written to
            the output

        Returns
        --------
        q : ndarray
            displacements of the system with q[:, i] being the
            displacement of the i-th timestep in voigt notation
        dq : ndarray
            velocities of the system with dq[:, i] being the velocity of the
            i-th timestep in voigt notation

        Examples
        ---------
        TODO

        '''
        t_clock_1 = time.time()
        eps = 1E-13

        beta = self.beta
        gamma = self.gamma
        # initialize starting variables
        t = 0
        q = q_start.copy()
        dq = dq_start.copy()
        ddq = np.zeros_like(q_start)

#        # predict start values for ddq:
#        ddq = linalg.spsolve(self.M, self.f_non(q, t))
        abs_f_ext = self.atol
        no_newton_convergence_flag = False
        time_index = 0 # index of the timestep in the time_range array
        while time_index < len(time_range):

            # write output, if necessary
            if t+eps >= time_range[time_index]:
                self.mechanical_system.write_timestep(t, q.copy())
                time_index += 1
                if time_index == len(time_range):
                    break


            # time tolerance fitting...
            if t + self.delta_t + eps >= time_range[time_index]:
                dt = time_range[time_index] - t
            else:
                dt = self.delta_t

            # make half the step size if newton did not converge
            if no_newton_convergence_flag:
                dt /= 2
                no_newton_convergence_flag = False

            # saving state for recovery if no convergence is gained
            t_old = t
            q_old = q.copy()
            dq_old = dq.copy()

            # Prediction using state from previous step
            t += dt
            q += dt*dq + (1/2-beta)*dt**2*ddq
            dq += (1-gamma)*dt*ddq
            ddq *= 0

            S, res, f_ext = self.mechanical_system.S_and_res(q, dq, ddq, dt,
                                                             t, beta, gamma)
            abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
            res_abs = norm_of_vector(res)

            # Newton-Correction-loop
            n_iter = 0
            while res_abs > self.rtol*abs_f_ext + self.atol:

                if sp.sparse.issparse(S):
                    delta_q = - sp.sparse.linalg.spsolve(S, res)
                else:
                    delta_q = - sp.linalg.solve(S, res)

                # update state variables
                q += delta_q
                dq += gamma/(beta*dt)*delta_q
                ddq += 1/(beta*dt**2)*delta_q

                # update system matrices and vectors
                S, res, f_ext = self.mechanical_system.S_and_res(q, dq, ddq, dt,
                                                                 t, beta, gamma)
                res_abs = norm_of_vector(res)
                abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
                n_iter += 1

                if self.verbose:
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
                if self.write_iter:
                    t_write = t + dt/100*n_iter
                    self.mechanical_system.write_timestep(t_write, q.copy())

                # catch when the newton loop doesn't converge
                if n_iter > self.n_iter_max:
                    if self.conv_abort:
                        print(abort_statement)
                        return
                        # raise Exception('No convergence in Newton-Loop!')
                    t = t_old
                    q = q_old.copy()
                    dq = dq_old.copy()
                    no_newton_convergence_flag = True
                    break
                    

            print('Time:', t, 'No of iterations:', n_iter,
                  'Residual: {0:4.2E}'.format(res_abs))
        # end of time loop
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

    Note
    ----
    Due to round-off-errors, the internal time step width is h and is very
    close to dt, but adjusted to fit the steps exactly.
    '''
    t_clock_1 = time.time()
    print('Starting linear time integration')
    eps = 1E-12 # epsilon for floating point round off errors
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
    S = M + beta * dt**2 * K
    S_inv = sp.sparse.linalg.splu(S)
    # S_inv.solve(rhs_vec) # method to solve the system efficiently
    print('Iteration matrix successfully factorized. Starting time marching...')
    # initialization of the state variables
    t = 0
    q = q0.copy()
    dq = dq0.copy()
    # Evaluation of the initial acceleration
    f_ext = mechanical_system.f_ext(q, dq, t)
    ddq = sp.sparse.linalg.spsolve(M, f_ext - K @ q)

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
    mechanical_system.write_timestep(0, f_ext*0) # write zeros
    if verbose:
        print('Start solving linear static problem')
    u = linalg.spsolve(K, f_ext - f_int)
    mechanical_system.write_timestep(1, u)
    if verbose:
        print('Static problem solved')


def solve_nonlinear_displacement(mechanical_system, no_of_load_steps=10,
                                 t=0, rtol=1E-8, atol=1E-14, newton_damping=1,
                                 n_max_iter=1000, smplfd_nwtn_itr=1,
                                 wrt_iter=False, verbose=True):
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

    Returns
    -------
    None

    Examples
    ---------
    TODO

    '''
    t_clock_1 = time.time()
    stepwidth = 1/no_of_load_steps
    ndof = mechanical_system.dirichlet_class.no_of_constrained_dofs
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
            corr = linalg.spsolve(K, res)
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
        mechanical_system.write_timestep(t, u)
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

        Note
        ----
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
