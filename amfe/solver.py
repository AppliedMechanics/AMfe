# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Module that contains solvers for solving systems in AMfe.
"""


import scipy as sp
import numpy as np
import time

from .mechanical_system import *
from .linalg import *

__all__ = [
    # 'choose_solver',
    'Solver',
    'NonlinearStaticsSolver',
    'LinearStaticsSolver',
    'NonlinearDynamicsSolver',
    'LinearDynamicsSolver',
    'GeneralizedAlphaNonlinearDynamicsSolver',
    'JWHAlphaNonlinearDynamicsSolver',
    'GeneralizedAlphaLinearDynamicsSolver',
    'JWHAlphaLinearDynamicsSolver',
    'JWHAlphaNonlinearDynamicsSolverStateSpace',
    'JWHAlphaLinearDynamicsSolverStateSpace',
    'ConstrainedNonlinearDynamicsSolver'
]


abort_statement = '''
###################################################
#### The current computation has been aborted. ####
#### No convergence was gained within the      ####
#### number of given iteration steps.          ####
###################################################
'''


# Most general solver class
# -------------------------
class Solver:
    '''
    Most general solver class for the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.
    '''

    def __init__(self, mechanical_system, **options):
        self.mechanical_system = mechanical_system

        # read options
        if 'linear_solver' in options:
            self.linear_solver = options['linear_solver']
        else:
            print('Attention: No linear solver was given, setting linear_solver = PardisoSolver.')
            self.linear_solver = PardisoSolver(A=None)

        if 'linear_solver_options' in options:
            self.linear_solver.set_options(**options['linear_solver_options'])
        return

    def solve(self):
        return None


# Solver classes for all statics solver
# -------------------------------------
class NonlinearStaticsSolver(Solver):
    '''
    Class for solving the nonlinear static problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        if 'number_of_load_steps' in options:
            self.number_of_load_steps = options['number_of_load_steps']
        else:
            print('Attention: No number of load steps was given, setting number_of_load_steps = 10.')
            self.number_of_load_steps = 10

        if 'relative_tolerance' in options:
            self.relative_tolerance = options['relative_tolerance']
        elif 'rtol' in options:
            self.relative_tolerance = options['rtol']
        else:
            print('Attention: No relative tolerance was given, setting relative_tolerance = 1.0e-9.')
            self.relative_tolerance = 1.0e-9

        if 'absolute_tolerance' in options:
            self.absolute_tolerance = options['absolute_tolerance']
        elif 'atol' in options:
            self.absolute_tolerance = options['atol']
        else:
            print('Attention: No absolute tolerance was given, setting absolute_tolerance = 1.0e-6.')
            self.absolute_tolerance = 1.0e-6

        if 'newton_damping' in options:
            self.newton_damping = options['newton_damping']
        else:
            print('Attention: No newton damping was given, setting newton_damping = 1.0.')
            self.newton_damping = 1.0

        if 'max_number_of_iterations' in options:
            self.max_number_of_iterations = options['max_number_of_iterations']
        else:
            print('Attention: No maximum number of iterations was given, setting max_number_of_iterations = 1000.')
            self.max_number_of_iterations = 1000

        if 'simplified_newton_iterations' in options:
            self.simplified_newton_iterations = options['simplified_newton_iterations']
        else:
            print('Attention: No number of simplified Newton iterations was given, ' \
                  + 'setting simplified_newton_iterations = 1.')
            self.simplified_newton_iterations = 1

        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            print('Attention: No verbose was given, setting verbose = False.')
            self.verbose = False

        if 'track_iterations' in options:
            self.track_iterations = options['track_iterations']
        else:
            print('Attention: No track number of iterations was given, setting track_iterations = False.')
            self.track_iterations = False

        if 'write_iterations' in options:
            self.write_iterations = options['write_iterations']
        else:
            print('Attention: No write iterations was given, setting write_iterations = False.')
            self.write_iterations = False

        if 'convergence_abort' in options:
            self.convergence_abort = options['convergence_abort']
        else:
            print('Attention: No convergence abort was given, setting convergence_abort = True.')
            self.convergence_abort = True

        if 'save_solution' in options:
            self.save_solution = options['save_solution']
        else:
            print('Attention: No save solution was given, setting save_solution = True.')
            self.save_solution = True

        self.iteration_info = np.array([])
        return

    def solve(self):
        '''
        Solves the nonlinear static problem of the mechanical system.
            
        Parameters
        ----------

        Returns
        -------
        displacement : ndarray, shape(ndim, number_of_load_steps)
            Static displacement field (solution) at load steps; q[:,-1] is the final displacement at pseudo time 1.0.
        '''

        # start time measurement
        t_clock_start = time.time()

        # TODO: getter method insert getter method for ndof (K is not needed)
        K, f_int = self.mechanical_system.K_and_f()
        ndof = K.shape[0]

        # initialize u, du
        u = np.zeros(ndof)
        du = np.zeros(ndof)
        # initialize variables and set parameters
        if self.save_solution:
            self.mechanical_system.clear_timesteps()
            # write initial state
            self.mechanical_system.write_timestep(0.0, u)

        iteration_info = []
        u_output = []
        stepwidth = 1/self.number_of_load_steps

        # load step loop: t goes from (0+stepwidth) to 1
        for t in np.arange(stepwidth, 1.0 + stepwidth, stepwidth):

            # Calculate residuum:
            K, f_int = self.mechanical_system.K_and_f(u, t)
            f_ext = self.mechanical_system.f_ext(u, du, t)
            res = -f_int + f_ext

            # calculate norms
            abs_res = euclidean_norm_of_vector(res)
            abs_f_ext = euclidean_norm_of_vector(f_ext)

            # Newton iteration loop
            iteration = 0
            while (abs_res > self.relative_tolerance*abs_f_ext + self.absolute_tolerance) and \
                    (self.max_number_of_iterations > iteration):

                # solve for displacement correction
                self.linear_solver.set_A(K)
                delta_u = self.linear_solver.solve(res)

                # correct displacement
                u += delta_u*self.newton_damping

                # update K, f_int and f_ext if not a simplified newton iteration
                if (iteration % self.simplified_newton_iterations) is 0:
                    K, f_int = self.mechanical_system.K_and_f(u, t)
                    f_ext = self.mechanical_system.f_ext(u, du, t)
                # update residuum and norms
                res = -f_int + f_ext
                abs_f_ext = euclidean_norm_of_vector(f_ext)
                abs_res = euclidean_norm_of_vector(res)
                iteration += 1

                if self.verbose:
                    print('Step: {0:1.3f}, iteration#: {1:3d}, residual: {2:6.3E}'.format(t, iteration, abs_res))

                if self.write_iterations:
                    self.mechanical_system.write_timestep(t + iteration*0.000001, u)

                # exit, if max iterations exceeded
                if (iteration >= self.max_number_of_iterations) and self.convergence_abort:
                    u_output = np.array(u_output).T
                    print(abort_statement)
                    t_clock_end = time.time()
                    print('Time for static solution: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
                    return u_output

            # end of Newton iteration loop

            if self.save_solution:
                self.mechanical_system.write_timestep(t, u)
            u_output.append(u.copy())

            if self.track_iterations:
                iteration_info.append((t, iteration, abs_res))

        # end of load step loop

        self.iteration_info = np.array(iteration_info)
        u_output = np.array(u_output).T
        t_clock_end = time.time()
        print('Time for solving nonlinear displacements: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return u_output


class LinearStaticsSolver(Solver):
    '''
    Class for solving the linear static problem of the mechanical system linearized around zero-displacement.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be linearized at zero displacement and solved.
    options : Dictionary
        Options for solver.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        if 't' in options:
            self.t = options['t']
        else:
            print('Attention: No pseudo time evaluation was given, setting t = 1.0.')
            self.t = 1.0
        return

    def solve(self):
        '''
        Solves the linear static problem of the mechanical system linearized around zero-displacement.

        Returns
        -------
        u : ndaray
            Static displacement field (solution).
        '''

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='spd')
        self.mechanical_system.clear_timesteps()

        print('Assembling external force and stiffness...')
        K = self.mechanical_system.K(u=None, t=self.t)
        f_ext = self.mechanical_system.f_ext(u=None, du=None, t=self.t)
        self.mechanical_system.write_timestep(0.0, 0.0*f_ext)  # write initial state

        print('Start solving linear static problem...')
        self.linear_solver.set_A(K)
        u = self.linear_solver.solve(f_ext)
        self.mechanical_system.write_timestep(self.t, u)  # write deformed state
        print('Static problem solved.')
        return u


# General solver class for all dynamics solver
# --------------------------------------------
class NonlinearDynamicsSolver(Solver):
    '''
    General class for solving the nonlinear dynamic problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        initial_conditions : dict {'q0': numpy.array, 'dq0': numpy.array}
            Initial conditions/initial displacement and initial velocity for solver.
        t0 : float
            Initial time.
        t_end : float
            End time.
        dt : float
            Time step size for time integration.
        dt_output : float
            Time step size for output.
        relative_tolerance : float

        absolute_tolerance : float

        max_number_of_iterations : int

        convergence_abort : Boolean

        verbose : Boolean
            If true, show some more information in command line.
        write_iterations : Boolean
            If true, write iteration steps.
        track_iterations : Boolean

    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        self.options = dict(**options)

        if ('initial_conditions' in options) and ('q0' in options['initial_conditions']):
            q0 = options['initial_conditions']['q0']
            if len(q0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of q0 not valid for mechanical system.')
        else:
            print('Attention: No initial displacement was given, setting q0 = 0.')
            q0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        if ('initial_conditions' in options) and ('dq0' in options['initial_conditions']):
            dq0 = options['initial_conditions']['dq0']
            if len(dq0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of dq0 is not valid for mechanical system.')
        else:
            print('Attention: No initial velocity was given, setting dq0 = 0.')
            dq0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        self.initial_conditions = {'q0': q0, 'dq0': dq0}

        if 't0' in options:
            self.t0 = options['t0']
        else:
            print('Attention: No initial time was given, setting t0 = 0.0.')
            self.t0 = 0.0

        if 't_end' in options:
            self.t_end = options['t_end']
        else:
            print('Attention: No end time was given, setting t_end = 1.0.')
            self.t_end = 1.0

        if 'dt' in options:
            self.dt = options['dt']
        else:
            print('Attention: No time step size was given, setting dt = 1.0e-4.')
            self.dt = 1.0e-4

        if 'output_frequency' in options:
            self.output_frequency = options['output_frequency']
        else:
            print('Attention: No output frequency was given, setting output_frequency = 1.')
            self.output_frequency = 1

        if 'relative_tolerance' in options:
            self.relative_tolerance = options['relative_tolerance']
        elif 'rtol' in options:
            self.relative_tolerance = options['rtol']
        else:
            print('Attention: No relative tolerance was given, setting relative_tolerance = 1.0e-9.')
            self.relative_tolerance = 1.0e-9

        if 'absolute_tolerance' in options:
            self.absolute_tolerance = options['absolute_tolerance']
        elif 'atol' in options:
            self.absolute_tolerance = options['atol']
        else:
            print('Attention: No absolute tolerance was given, setting absolute_tolerance = 1.0e-6.')
            self.absolute_tolerance = 1.0e-6

        if 'max_number_of_iterations' in options:
            self.max_number_of_iterations = options['max_number_of_iterations']
        else:
            print('Attention: No maximum number of iterations was given, setting max_number_of_iterations = 30.')
            self.max_number_of_iterations = 30

        if 'convergence_abort' in options:
            self.convergence_abort = options['convergence_abort']
        else:
            print('Attention: No convergence abort was given, setting convergence_abort = True.')
            self.convergence_abort = True

        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            print('Attention: No verbose was given, setting verbose = False.')
            self.verbose = False

        if 'write_iterations' in options:
            self.write_iterations = options['write_iterations']
        else:
            print('Attention: No write iterations was given, setting write_iterations = False.')
            self.write_iterations = False

        if 'track_iterations' in options:
            self.track_iterations = options['track_iterations']
        else:
            print('Attention: No track number of iterations was given, setting track_iterations = False.')
            self.track_iterations = False
        if 'use_v' in options:
            self.use_additional_variable_v = options['use_additional_variable_v']
        else:
            self.use_additional_variable_v = False
        return

    def overwrite_parameters(self, **options):
        pass

    def predict(self, q, dq, v, ddq):
        pass

    def newton_raphson(self, q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old):
        pass

    def correct(self, q, dq, v, ddq, delta_q):
        pass

    def solve(self):
        '''
        Solves the nonlinear dynamic problem of the mechanical system.
        '''

        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='sid')
        self.mechanical_system.clear_timesteps()
        self.iteration_info = []
        t = self.t0
        q = self.initial_conditions['q0'].copy()
        dq = self.initial_conditions['dq0'].copy()
        if self.use_additional_variable_v:
            v = self.initial_conditions['dq0'].copy()
        else:
            v = np.empty((0, 0))
        ddq = np.zeros_like(q)
        f_ext = np.zeros_like(q)
        abs_f_ext = self.absolute_tolerance

        # write output of initial conditions
        self.mechanical_system.write_timestep(t, q.copy())

        # time step loop
        output_index = 0
        while t < self.t_end:

            # save old variables
            q_old = q.copy()
            dq_old = dq.copy()
            v_old = v.copy()
            ddq_old = ddq.copy()
            f_ext_old = f_ext.copy()
            t_old = t

            # predict new variables
            output_index += 1
            t += self.dt

            q, dq, v, ddq = self.predict(q, dq, v, ddq)

            Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
            abs_f_ext = max(abs_f_ext, euclidean_norm_of_vector(f_ext))
            res_abs = euclidean_norm_of_vector(res)

            # Newton-Raphson iteration loop
            iteration = 0
            while res_abs > self.relative_tolerance*abs_f_ext + self.absolute_tolerance:

                iteration += 1

                # catch failing convergence
                if iteration > self.max_number_of_iterations:
                    iteration -= 1
                    if self.convergence_abort:
                        print(abort_statement)
                        t_clock_end = time.time()
                        print('Time for time marching integration: {0:6.3f}s.'.format(t_clock_end - t_clock_start))
                        return
                    break

                # solve for displacement correction
                self.linear_solver.set_A(Jac)
                delta_q = -self.linear_solver.solve(res)

                # correct variables
                q, dq, v, ddq = self.correct(q, dq, v, ddq, delta_q)

                # update system quantities
                Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                res_abs = euclidean_norm_of_vector(res)

                if self.verbose:
                    if sp.sparse.issparse(Jac):
                        cond_nr = 0.0
                    else:
                        cond_nr = np.linalg.cond(Jac)
                    print('Iteration: {0:3d}, residual: {1:6.3E}, condition# of Jacobian: {2:6.3E}'.format(
                        iteration, res_abs, cond_nr))

                if self.write_iterations:
                    t_write = t + self.dt/1000000*iteration
                    self.mechanical_system.write_timestep(t_write, q.copy())

                # end of Newton-Raphson iteration loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, q.copy())
                output_index = 0

            if self.track_iterations:
                self.iteration_info.append((t, iteration, res_abs))

            print('Time: {0:3.6f}, #iterations: {1:3d}, residual: {2:6.3E}'.format(t, iteration, res_abs))

            # end of time step loop

        self.linear_solver.clear()

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return

    def solve_with_adaptive_time_step(self, dt_start, dt_min, dt_max, change_factor_min, change_factor_max,
                                      safety_factor, failing_newton_convergence_factor, trust_value,
                                      relative_dt_tolerance, max_dt_iterations, failing_dt_convergence_abort=True):
        '''
        Solves the nonlinear dynamic problem of the mechanical system with adaptive time step.

        Parameters
        ----------
        dt_start : float
            Starting time step size.
        dt_min : float
            Minimal time step size, i.e. lower bound.
        dt_max : float
            Maximal time step size, i.e. upper bound.
        change_factor_min : float
            Minimal change factor for time step size, i.e. lower bound. 0 < change_factor_min <= 1 required.
            0.1 <= change_factor_min <= 0.5 recommended.
        change_factor_max : float
            Maximal change factor for time step size, i.e. upper bound. 1 <= change_factor_max < infinity required.
            1.5 <= change_factor_max <= 5 recommended.
        safety_factor : float
            Safty factor for time step size change. 0 < safty_factor < 1 required. 0.8 <= safty_factor < 0.95
            recommended.
        failing_newton_convergence_factor : float
            Change factor for time step size for failing Newton-Raphson convergence.
            0 < failing_newton_convergence_factor < 1 required. 0.5 <= failing_newton_convergence_factor <= 0.8
            recommended.
        trust_value : float
            Trust value for new time step size, i.e. parameter for PT1 low-pass filtering in case of increasing time
            step sizes (dt_new_used = trust_value*dt_new_calculated + (1 - trust_value)*dt_old). 0 < trust_value <= 1
            required. 0 < trust_value << 1 recommended.
        relative_dt_tolerance : float
            Tolerance for relative local time discretization error. absolute local time discretization error =
            relative_dt_tolerance*maximal displacement so far.
        max_dt_iterations : int
            Maximal number of time step size adaption iterations per time step.
        failing_dt_convergence_abort : Boolean
            If True abort simulation, otherwise proceed with last result to next time step, when exceeding maximal
            number of time step size adaption iterations. Default failing_dt_convergence_abort = True.


        References
        ----------
           [1]  O. C. Zienkiewicz and Y. M. Xie (1991): A simple error estimator and adaptive time stepping procedure
                for dynamic analysis. Earthquake Engineering & Structural Dynamics 20(9) 871--887.
                DOI: 10.1002/eqe.4290200907
           [2]  M. Mayr, W.A. Wall and M.W. Gee (2018): Adaptive time stepping for fluid-structure interaction solvers.
                Finite Elements in Analysis and Design 14155--69. DOI: 10.1016/j.finel.2017.12.002
        '''

        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='sid')
        self.mechanical_system.clear_timesteps()
        self.iteration_info = []
        t = self.t0
        self.dt = dt_start
        q = self.initial_conditions['q0'].copy()
        dq = self.initial_conditions['dq0'].copy()
        if self.use_additional_variable_v:
            v = self.initial_conditions['dq0'].copy()
        else:
            v = np.empty((0, 0))
        ddq = np.zeros_like(q)
        max_q = 0.0
        f_ext = np.zeros_like(q)
        abs_f_ext = self.absolute_tolerance
        self.dt_info = [dt_start]

        # write output of initial conditions
        self.mechanical_system.write_timestep(t, q.copy())

        # time step loop
        output_index = 0
        while t < self.t_end:

            # save old variables
            q_old = q.copy()
            dq_old = dq.copy()
            v_old = v.copy()
            ddq_old = ddq.copy()
            f_ext_old = f_ext.copy()
            t_old = t

            output_index += 1
            abs_local_dt_err = 1.0e16
            no_newton_convergence = False

            dt_iteration = 0
            while abs_local_dt_err > relative_dt_tolerance*max_q:

                dt_iteration += 1

                # catch failing dt convergence
                if dt_iteration > max_dt_iterations:
                    dt_iteration -= 1
                    if failing_dt_convergence_abort:
                        print(abort_statement)
                        t_clock_end = time.time()
                        print('Time for time marching integration: {0:6.3f}s.'.format(t_clock_end - t_clock_start))
                        return
                    break

                # update max displacement
                max_q = max(max_q, np.max(q))

                # reset variables
                q = q_old.copy()
                dq = dq_old.copy()
                v = v_old.copy()
                ddq = ddq_old.copy()
                f_ext = f_ext_old.copy()
                t = t_old

                # predict new variables
                t += self.dt
                q, dq, v, ddq = self.predict(q, dq, v, ddq)

                Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                abs_f_ext = max(abs_f_ext, euclidean_norm_of_vector(f_ext))
                res_abs = euclidean_norm_of_vector(res)

                # Newton-Raphson iteration loop
                newton_iteration = 0
                while res_abs > self.relative_tolerance * abs_f_ext + self.absolute_tolerance:

                    newton_iteration += 1

                    # catch failing Newton-Raphson convergence
                    if newton_iteration > self.max_number_of_iterations:
                        newton_iteration -= 1
                        no_newton_convergence = True
                        break

                    # solve for displacement correction
                    self.linear_solver.set_A(Jac)
                    delta_q = -self.linear_solver.solve(res)

                    # correct variables
                    q, dq, v, ddq = self.correct(q, dq, v, ddq, delta_q)

                    # update system quantities
                    Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                    res_abs = euclidean_norm_of_vector(res)

                    if self.verbose:
                        if sp.sparse.issparse(Jac):
                            cond_nr = 0.0
                        else:
                            cond_nr = np.linalg.cond(Jac)
                        print('Iteration: {0:3d}, residual: {1:6.3E}, condition# of Jacobian: {2:6.3E}'.format(
                            newton_iteration, res_abs, cond_nr))

                    if self.write_iterations:
                        t_write = t + self.dt / 1000000 * newton_iteration
                        self.mechanical_system.write_timestep(t_write, q.copy())

                    # end of Newton-Raphson iteration loop

                # update time step
                if no_newton_convergence:  # reduce time step to defined percentage
                    self.dt *= failing_newton_convergence_factor
                    if self.dt < dt_min:
                        self.dt = dt_min
                    abs_local_dt_err = 1.0e16
                    no_newton_convergence = False
                else:  # evaluate local temporal discretization error and update time step accordingly
                    abs_local_dt_err = self.estimate_local_time_discretization_error(ddq, ddq_old)
                    kappa = np.cbrt(relative_dt_tolerance*max_q/abs_local_dt_err)
                    dt_new = min(dt_max, max(min(change_factor_max,
                                                 max(change_factor_min, safety_factor*kappa))*self.dt, dt_min))
                    if (dt_new > self.dt) and (len(self.dt_info) > 1):
                        self.dt = trust_value*dt_new + (1 - trust_value)*self.dt_info[-1]
                    else:
                        self.dt = dt_new

            # end time step adaption loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, q.copy())
                output_index = 0

            if self.track_iterations:
                self.iteration_info.append((t, newton_iteration, res_abs))

            # track final time step size
            self.dt_info.append(self.dt)

            print('Time: {0:3.6f}, #dt-iterations: {1:3d}, #NR-iterations: {2:3d}, NR-residual: {3:6.3E}'.format(
                t, dt_iteration, newton_iteration, res_abs))

            # end of time step loop

        self.linear_solver.clear()

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        pass


class LinearDynamicsSolver(Solver):
    '''
    General class for solving the linear dynamic problem of the mechanical system linearized around zero-displacement.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.

    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        self.options = dict(**options)

        if ('initial_conditions' in options) and ('q0' in options['initial_conditions']):
            q0 = options['initial_conditions']['q0']
            if len(q0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of q0 not valid for mechanical system.')
        else:
            print('Attention: No initial displacement is given, setting q0 = 0.')
            q0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        if ('initial_conditions' in options) and ('dq0' in options['initial_conditions']):
            dq0 = options['initial_conditions']['dq0']
            if len(dq0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of dq0 is not valid for mechanical system.')
        else:
            print('Attention: No initial velocity is given, setting dq0 = 0.')
            dq0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        self.initial_conditions = {'q0':q0, 'dq0':dq0}

        if 't0' in options:
            self.t0 = options['t0']
        else:
            print('Attention: No initial time was given, setting t0 = 0.0.')
            self.t0 = 0.0

        if 't_end' in options:
            self.t_end = options['t_end']
        else:
            print('Attention: No end time was given, setting t_end = 1.0.')
            self.t_end = 1.0

        if 'dt' in options:
            self.dt = options['dt']
        else:
            print('Attention: No time step size was, setting dt = 1.0e-4.')
            self.dt = 1.0e-4

        if 'output_frequency' in options:
            self.output_frequency = options['output_frequency']
        else:
            print('Attention: No output frequency was given, setting output_frequency = 1.')
            self.output_frequency = 1

        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            print('Attention: No verbose was given, setting verbose = False.')
            self.verbose = False
        return

    def overwrite_parameters(self, **options):
        pass

    def effective_stiffness(self):
        pass

    def effective_force(self, q_old, dq_old, v_old, ddq_old, t, t_old):
        pass

    def update(self, q, q_old, dq_old, v_old, ddq_old):
        pass

    def solve(self):
        '''
        Solves the linear dynamic problem of the mechanical system linearized around zero-displacement.
        '''

        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='spd')
        self.mechanical_system.clear_timesteps()
        t = self.t0
        q = self.initial_conditions['q0'].copy()
        dq = self.initial_conditions['dq0'].copy()
        if self.use_additional_variable_v:
            v = self.initial_conditions['dq0'].copy()
        else:
            v = np.empty((0, 0))

        # evaluate initial acceleration and LU-decompose effective stiffness
        K_eff = self.effective_stiffness()

        self.linear_solver.set_A(self.mechanical_system.M_constr)
        ddq = self.linear_solver.solve(self.mechanical_system.f_ext(q, dq, t)
                                       - self.mechanical_system.D_constr@dq \
                                       - self.mechanical_system.K_constr@q)

        self.linear_solver.set_A(K_eff)
        if hasattr(self.linear_solver, 'factorize'):
            self.linear_solver.factorize()

        # write output of initial conditions
        self.mechanical_system.write_timestep(t, q.copy())

        # time step loop
        output_index = 0
        while t < self.t_end:

            # save old variables
            q_old = q.copy()
            dq_old = dq.copy()
            v_old = v.copy()
            ddq_old = ddq.copy()
            t_old = t

            # solve system
            output_index += 1
            t += self.dt
            f_eff = self.effective_force(q_old, dq_old, v_old, ddq_old, t, t_old)

            q = self.linear_solver.solve(f_eff)

            # update variables
            dq, v, ddq = self.update(q, q_old, dq_old, v_old, ddq_old)

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, q.copy())
                output_index = 0

            print('Time: {0:3.6f}'.format(t))

            # end of time step loop

        self.linear_solver.clear()

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return


class NonlinearDynamicsSolverStateSpace(Solver):
    '''
    General class for solving the nonlinear dynamic problem of the state-space system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver:
        initial_conditions : dict {'x0': numpy.array}
            Initial conditions/states for solver.
        t0 : float
            Initial time.
        t_end : float
            End time.
        dt : float
            Time step size for time integration.
        dt_output : float
            Time step size for output.
        relative_tolerance : float

        absolute_tolerance : float

        max_number_of_iterations : int

        convergence_abort : Boolean

        verbose : Boolean
            If true, show some more information in command line.
        write_iterations : Boolean
            If true, write iteration steps.
        track_iterations : Boolean

    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        self.options = dict(**options)

        if ('initial_conditions' in options) and ('x0' in options['initial_conditions']):
            x0 = options['initial_conditions']['x0']
            if len(x0) != 2*self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of x0 not valid for mechanical system.')
        else:
            print('Attention: No initial state is given, setting x0 = 0.')
            x0 = np.zeros(2*self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        self.initial_conditions = {'x0': x0}

        if 't0' in options:
            self.t0 = options['t0']
        else:
            print('Attention: No initial time was given, setting t0 = 0.0.')
            self.t0 = 0.0

        if 't_end' in options:
            self.t_end = options['t_end']
        else:
            print('Attention: No end time was given, setting t_end = 1.0.')
            self.t_end = 1.0

        if 'dt' in options:
            self.dt = options['dt']
        else:
            print('Attention: No time step size was given, setting dt = 1.0e-4.')
            self.dt = 1.0e-4

        if 'output_frequency' in options:
            self.output_frequency = options['output_frequency']
        else:
            print('Attention: No output frequency was given, setting output_frequency = 1.')
            self.output_frequency = 1

        if 'relative_tolerance' in options:
            self.relative_tolerance = options['relative_tolerance']
        else:
            print('Attention: No relative tolerance was given, setting relative_tolerance = 1.0e-9.')
            self.relative_tolerance = 1.0E-9

        if 'absolute_tolerance' in options:
            self.absolute_tolerance = options['absolute_tolerance']
        else:
            print('Attention: No absolute tolerance was given, setting absolute_tolerance = 1.0e-6.')
            self.absolute_tolerance = 1.0E-6

        if 'max_number_of_iterations' in options:
            self.max_number_of_iterations = options['max_number_of_iterations']
        else:
            print('Attention: No maximum number of iterations was given, setting max_number_of_iterations = 30.')
            self.max_number_of_iterations = 30

        if 'convergence_abort' in options:
            self.convergence_abort = options['convergence_abort']
        else:
            print('Attention: No convergence abort was given, setting convergence_abort = True.')
            self.convergence_abort = True

        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            print('Attention: No verbose was given, setting verbose = False.')
            self.verbose = False

        if 'write_iterations' in options:
            self.write_iterations = options['write_iterations']
        else:
            print('Attention: No write iterations was given, setting write_iterations = False.')
            self.write_iterations = False

        if 'track_iterations' in options:
            self.track_iterations = options['track_iterations']
        else:
            print('Attention: No track iterations was given, setting track_iterations = False.')
            self.track_iterations = False
        return

    def overwrite_parameters(self, **options):
        pass

    def predict(self, x, dx):
        pass

    def newton_raphson(self, x, dx, t, x_old, dx_old, t_old):
        pass

    def correct(self, x, dx, delta_x):
        pass

    def solve(self):
        '''
        Solves the nonlinear dynamic problem of the state-space system.
        '''


        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='nonsym')
        self.mechanical_system.clear_timesteps()
        self.iteration_info = []
        t = self.t0
        x = self.initial_conditions['x0'].copy()
        dx = np.zeros_like(x)
        F_ext = np.zeros_like(x)
        abs_F_ext = self.absolute_tolerance

        # write output of initial conditions
        self.mechanical_system.write_timestep(t, x.copy())

        # time step loop
        output_index = 0
        while t < self.t_end:

            # save old variables
            x_old = x.copy()
            dx_old = dx.copy()
            F_ext_old = F_ext.copy()
            t_old = t

            # predict new variables
            output_index += 1
            t += self.dt
            x, xq = self.predict(x, dx)

            Jac, Res, F_ext = self.newton_raphson(x, dx, t, x_old, dx_old, t_old)
            abs_F_ext = max(abs_F_ext, euclidean_norm_of_vector(F_ext))
            Res_abs = euclidean_norm_of_vector(Res)

            # Newton-Raphson iteration loop
            iteration = 0
            while Res_abs > self.relative_tolerance*abs_F_ext + self.absolute_tolerance:

                iteration += 1

                # catch failing convergence
                if iteration > self.max_number_of_iterations:
                    iteration -= 1
                    if self.convergence_abort:
                        print(abort_statement)
                        self.iteration_info = np.array(self.iteration_info)
                        t_clock_end = time.time()
                        print('Time for time marching integration: {0:6.3f}s.'.format(t_clock_end - t_clock_start))
                        return
                    break

                # solve for state correction
                self.linear_solver.set_A(Jac)
                delta_x = -self.linear_solver.solve(Res)

                # correct variables
                x, dx = self.correct(x, dx, delta_x)

                # update system quantities
                Jac, Res, F_ext = self.newton_raphson(x, dx, t, x_old, dx_old, t_old)
                Res_abs = euclidean_norm_of_vector(Res)

                if self.verbose:
                    if sp.sparse.issparse(Jac):
                        cond_nr = 0.0
                    else:
                        cond_nr = np.linalg.cond(Jac)
                    print('Iteration: {0:3d}, residual: {1:6.3E}, condition# of Jacobian: {2:6.3E}'.format(
                        iteration, Res_abs, cond_nr))

                if self.write_iterations:
                    t_write = t + self.dt/1000000*iteration
                    self.mechanical_system.write_timestep(t_write, x.copy())

                # end of Newton-Raphson iteration loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, x.copy())
                output_index = 0

            if self.track_iterations:
                self.iteration_info.append((t, iteration, Res_abs))

            print('Time: {0:3.6f}, #iterations: {1:3d}, residual: {2:6.3E}'.format(t, iteration, Res_abs))

            # end of time step loop

        self.linear_solver.clear()

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return


class LinearDynamicsSolverStateSpace(Solver):
    '''
    General class for solving the linear dynamic problem of the state-space system linearized around zero-state.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver.

    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        self.options = dict(**options)

        if ('initial_conditions' in options) and ('x0' in options['initial_conditions']):
            x0 = options['initial_conditions']['x0']
            if len(x0) != 2*self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
                raise ValueError('Error: Dimension of x0 not valid for mechanical system.')
        else:
            print('Attention: No initial state is given, setting x0 = 0.')
            x0 = np.zeros(2*self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        self.initial_conditions = {'x0': x0}

        if 't0' in options:
            self.t0 = options['t0']
        else:
            print('Attention: No initial time was given, setting t0 = 0.0.')
            self.t0 = 0.0

        if 't_end' in options:
            self.t_end = options['t_end']
        else:
            print('Attention: No end time was given, setting t_end = 1.0.')
            self.t_end = 1.0

        if 'dt' in options:
            self.dt = options['dt']
        else:
            print('Attention: No time step size was, setting dt = 1.0e-4.')
            self.dt = 1.0e-4

        if 'output_frequency' in options:
            self.output_frequency = options['output_frequency']
        else:
            print('Attention: No output frequency was given, setting output_frequency = 1.')
            self.output_frequency = 1

        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            print('Attention: No verbose was given, setting verbose = False.')
            self.verbose = False
        return

    def overwrite_parameters(self, **options):
        pass

    def effective_stiffness(self):
        pass

    def effective_force(self, x_old, dx_old, t, t_old):
        pass

    def update(self, x, x_old, dx_old):
        pass

    def solve(self):
        '''
        Solves the linear dynamic problem of the state-space system linearized around zero-state.
        '''

        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.linear_solver = self.linear_solver(mtype='nonsym')
        self.mechanical_system.clear_timesteps()
        t = self.t0
        x = self.initial_conditions['x0'].copy()

        # evaluate initial derivative and LU-decompose effective stiffness
        K_eff = self.effective_stiffness()

        self.linear_solver.set_A(self.mechanical_system.E_constr)
        dx = self.linear_solver.solve(self.mechanical_system.A_constr@x + self.mechanical_system.F_ext(x, t))

        self.linear_solver.set_A(K_eff)
        if hasattr(self.linear_solver, 'factorize'):
            self.linear_solver.factorize()

        # write output of initial conditions
        self.mechanical_system.write_timestep(t, x.copy())

        # time step loop
        output_index = 0
        while t < self.t_end:

            # save old variables
            x_old = x.copy()
            dx_old = dx.copy()
            t_old = t

            # solve system
            output_index += 1
            t += self.dt
            F_eff = self.effective_force(x_old, dx_old, t, t_old)

            x = self.linear_solver.solve(F_eff)

            # update variables
            dx = self.update(x, x_old, dx_old)

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, x.copy())
                output_index = 0

            print('Time: {0:3.6f}'.format(t))

            # end of time step loop

        self.linear_solver.clear()

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(t_clock_end - t_clock_start))
        return


# Special solvers derived from above
# ----------------------------------

class GeneralizedAlphaNonlinearDynamicsSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the generalized-alpha time
    integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolver
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  N.M. Newmark (1959): A method of computation for structural dynamics. Journal of the Engineering Mechanics
            Division (Proceedings of the American Society of Civil Engineers) 85 67--94.
       [2]  H.M. Hilber, T.J.R. Hughes and R.L. Taylor (1977): Improved numerical dissipation for time integration
            algorithms in structural dynamics. Earthquake Engineering and Structural Dynamics 5(3) 283--292.
            DOI: 10.1002/eqe.4290050306.
       [3]  W.L. Wood, M. Bossak and O.C. Zienkiewicz (1980): An alpha modification of Newmark's method. International
            Journal for Numerical Methods in Engineering 15(10) 1562--1566. DOI: 10.1002/nme.1620151011.
       [4]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
            numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
            DOI: 10.1115/1.2900803.
       [5]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = False

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (2*self.rho_inf - 1)/(self.rho_inf + 1)
        self.alpha_f = self.rho_inf/(self.rho_inf + 1)
        self.beta = 0.25*(1 - self.alpha_m + self.alpha_f)**2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        return

    def set_wbz_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as WBZ-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_m set rho_inf = (1 + alpha_m)/(1 - alpha_m) with -1 <= alpha_m <= 0.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = (rho_inf - 1)/(rho_inf + 1)
        self.alpha_f = 0.0
        self.beta = 0.25*(1 - self.alpha_m)**2
        self.gamma = 0.5 - self.alpha_m
        return

    def set_hht_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as HHT-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 1/2 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_f set rho_inf = (1 - alpha_f)/(1 + alpha_f) with 0 <= alpha_m <= 1/3.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = 0.0
        self.alpha_f = (1 - rho_inf)/(1 + rho_inf)
        self.beta = 0.25*(1 + self.alpha_f)**2
        self.gamma = 0.5 + self.alpha_f
        return

    def set_newmark_beta_parameters(self, beta=0.25, gamma=0.5):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as Newmark-beta scheme.

        Parameters
        ----------
        beta : float
            Default value beta = 1/4.
        gamma : float
            Default value gamma = 1/2.
        Unconditional stability for beta >= gamma/2 >= 1/4. Unconditionally stability and second-order accuracy but no
        numerical damping for beta >= 1/4 and gamma = 1/2. Unconditionally stability, second-order accuracy and best
        following of phase but no numerical damping for beta = 1/4  and gamma = 1/2 (corresponds to trapezoidal rule,
        default values). Alternative parametrization as Newmark-beta scheme with alpha-damping (modified average
        constant acceleration) -- in general not second-order accurate -- via beta = 1/4*(1 + alpha)^2 and
        gamma = 1/2 + alpha with damping alpha >= 0.
        '''

        self.rho_inf = None
        self.alpha_m = 0.0
        self.alpha_f = 0.0
        self.beta = beta
        self.gamma = gamma
        return

    def set_parameters(self, alpha_m, alpha_f, beta, gamma):
        '''
        Overwrite standard parameters for the nonlinear generalized-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        beta : float

        gamma : float

        Second-order accuracy for gamma = 1/2 - alpha_m + alpha_f. Unconditional stability for
        alpha_m <= alpha_f <= 1/2 and beta >= 1/4 + 1/2*(alpha_f - alpha_m).
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.beta = beta
        self.gamma = gamma
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear generalized-alpha time integration scheme.
        '''

        q += self.dt*dq + self.dt**2*(0.5 - self.beta)*ddq
        dq += self.dt*(1 - self.gamma)*ddq
        ddq *= 0
        return q, dq, v, ddq

    def newton_raphson(self, q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear generalized-alpha time integration scheme.
        '''

        if self.mechanical_system.M_constr is None:
            self.mechanical_system.M()

        ddq_m = (1 - self.alpha_m)*ddq + self.alpha_m*ddq_old
        q_f = (1 - self.alpha_f)*q + self.alpha_f*q_old
        dq_f = (1 - self.alpha_f)*dq + self.alpha_f*dq_old
        t_f = (1 - self.alpha_f)*t + self.alpha_f*t_old

        K_f, f_f = self.mechanical_system.K_and_f(q_f, t_f)

        f_ext_f = self.mechanical_system.f_ext(q_f, dq_f, t_f)

        if self.mechanical_system.D_constr is None:
            Jac = -(1 - self.alpha_m)/(self.beta*self.dt**2)*self.mechanical_system.M_constr - (1 - self.alpha_f)*K_f
            res = f_ext_f - self.mechanical_system.M_constr@ddq_m - f_f
        else:
            Jac = -(1 - self.alpha_m)/(self.beta*self.dt**2)*self.mechanical_system.M_constr \
                  - (1 - self.alpha_f)*self.gamma/(self.beta*self.dt)*self.mechanical_system.D_constr \
                  - (1 - self.alpha_f)*K_f

            res = f_ext_f - self.mechanical_system.M_constr@ddq_m \
                  - self.mechanical_system.D_constr@dq_f - f_f
        return Jac, res, f_ext_f

    def correct(self, q, dq, v, ddq, delta_q):
        '''
        Correct variables for the nonlinear generalized-alpha time integration scheme.
        '''

        q += delta_q
        dq += self.gamma/(self.beta*self.dt)*delta_q
        ddq += 1/(self.beta*self.dt**2)*delta_q
        return q, dq, v, ddq

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        '''
        Returns an estimate for the absolute local time discretization error for the nonlinear generalized-alpha time
        integration scheme.
        '''

        abs_local_dt_err = (self.beta - 1/6)*self.dt**2*length_scaled_euclidean_norm_of_vector(ddq - ddq_old)
        return abs_local_dt_err


class JWHAlphaNonlinearDynamicsSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the JWH-alpha time integration
    scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolver
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = True

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (3 - self.rho_inf)/(2*(1 + self.rho_inf))
        self.alpha_f = 1/(1 + self.rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def set_parameters(self, alpha_m, alpha_f, gamma):
        '''
        Overwrite standard parameters for the nonlinear JWH-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        gamma : float

        Second-order accuracy for gamma = 1/2 + alpha_m - alpha_f. Unconditional stability for
        alpha_m >= alpha_f >= 1/2.
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += self.dt*(self.alpha_m - self.gamma)/self.alpha_m*dq \
             + self.dt*self.gamma/self.alpha_m*v \
             + self.alpha_f*self.dt**2*self.gamma*(1 - self.gamma)/self.alpha_m*ddq
        dq += 1/self.alpha_m*(v - dq) + self.alpha_f*self.dt*(1 - self.gamma)/self.alpha_m*ddq
        v += self.dt*(1 - self.gamma)*ddq
        ddq *= 0
        return q, dq, v, ddq

    def newton_raphson(self, q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear JWH-alpha time integration scheme.
        '''

        if self.mechanical_system.M_constr is None:
            self.mechanical_system.M()

        ddq_m = self.alpha_m*ddq + (1 - self.alpha_m)*ddq_old
        q_f = self.alpha_f*q + (1 - self.alpha_f)*q_old
        v_f = self.alpha_f*v + (1 - self.alpha_f)*v_old
        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        K_f, f_f = self.mechanical_system.K_and_f(q_f, t_f)

        f_ext_f = self.mechanical_system.f_ext(q_f, v_f, t_f)

        if self.mechanical_system.D_constr is None:
            Jac = -self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2)*self.mechanical_system.M_constr \
                  - self.alpha_f*K_f
            res = f_ext_f - self.mechanical_system.M_constr@ddq_m - f_f
        else:
            Jac = -self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2)*self.mechanical_system.M_constr \
                  - self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.D_constr \
                  - self.alpha_f*K_f
            res = f_ext_f - self.mechanical_system.M_constr@ddq_m - self.mechanical_system.D_constr@v_f - f_f
        return Jac, res, f_ext_f

    def correct(self, q, dq, v, ddq, delta_q):
        '''
        Correct variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += delta_q
        dq += 1/(self.gamma*self.dt)*delta_q
        v += self.alpha_m/(self.alpha_f*self.gamma*self.dt)*delta_q
        ddq += self.alpha_m/(self.alpha_f*self.gamma**2*self.dt**2)*delta_q
        return q, dq, v, ddq

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        '''
        Returns an estimate for the absolute local time discretization error for the nonlinear JWH-alpha time
        integration scheme.
        '''

        raise ValueError('Error: Adaptive time stepping is not yet implemented for time integration with the '
                         + 'JWH-alpha scheme. Use the generalized-alpha scheme instead.')
        return


class GeneralizedAlphaLinearDynamicsSolver(LinearDynamicsSolver):
    '''
    Class for solving the linear dynamic problem of the mechanical system linearized around zero-displacement using the
    generalized-alpha time integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolver
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  N.M. Newmark (1959): A method of computation for structural dynamics. Journal of the Engineering Mechanics
            Division (Proceedings of the American Society of Civil Engineers) 85 67--94.
       [2]  H.M. Hilber, T.J.R. Hughes and R.L. Taylor (1977): Improved numerical dissipation for time integration
            algorithms in structural dynamics. Earthquake Engineering and Structural Dynamics 5(3) 283--292.
            DOI: 10.1002/eqe.4290050306.
       [3]  W.L. Wood, M. Bossak and O.C. Zienkiewicz (1980): An alpha modification of Newmark's method. International
            Journal for Numerical Methods in Engineering 15(10) 1562--1566. DOI: 10.1002/nme.1620151011.
       [4]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
            numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
            DOI: 10.1115/1.2900803.
       [5]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = False

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (2 * self.rho_inf - 1) / (self.rho_inf + 1)
        self.alpha_f = self.rho_inf / (self.rho_inf + 1)
        self.beta = 0.25 * (1 - self.alpha_m + self.alpha_f) ** 2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        return

    def set_wbz_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as WBZ-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_m set rho_inf = (1 + alpha_m)/(1 - alpha_m) with -1 <= alpha_m <= 0.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = (rho_inf - 1)/(rho_inf + 1)
        self.alpha_f = 0.0
        self.beta = 0.25*(1 - self.alpha_m)**2
        self.gamma = 0.5 - self.alpha_m
        return

    def set_hht_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as HHT-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 1/2 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_f set rho_inf = (1 - alpha_f)/(1 + alpha_f) with 0 <= alpha_m <= 1/3.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = 0.0
        self.alpha_f = (1 - rho_inf)/(1 + rho_inf)
        self.beta = 0.25*(1 + self.alpha_f)**2
        self.gamma = 0.5 + self.alpha_f
        return

    def set_newmark_beta_parameters(self, beta=0.25, gamma=0.5):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as Newmark-beta scheme.

        Parameters
        ----------
        beta : float
            Default value beta = 1/4.
        gamma : float
            Default value gamma = 1/2.
        Unconditional stability for beta >= gamma/2 >= 1/4. Unconditionally stability and second-order accuracy but no
        numerical damping for beta >= 1/4 and gamma = 1/2. Unconditionally stability, second-order accuracy and best
        following of phase but no numerical damping for beta = 1/4  and gamma = 1/2 (corresponds to trapezoidal rule,
        default values). Alternative parametrization as Newmark-beta scheme with alpha-damping (modified average
        constant acceleration) -- in general not second-order accurate -- via beta = 1/4*(1 + alpha)^2 and
        gamma = 1/2 + alpha with damping alpha >= 0.
        '''

        self.rho_inf = None
        self.alpha_m = 0.0
        self.alpha_f = 0.0
        self.beta = beta
        self.gamma = gamma
        return

    def set_parameters(self, alpha_m, alpha_f, beta, gamma):
        '''
        Overwrite standard parameters for the nonlinear generalized-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        beta : float

        gamma : float

        Second-order accuracy for gamma = 1/2 - alpha_m + alpha_f. Unconditional stability for
        alpha_m <= alpha_f <= 1/2 and beta >= 1/4 + 1/2*(alpha_f - alpha_m).
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.beta = beta
        self.gamma = gamma
        return

    def effective_stiffness(self):
        '''
        Return effective stiffness matrix for linear generalized-alpha time integration
        scheme.
        '''

        self.mechanical_system.M()
        self.mechanical_system.D_constr = self.mechanical_system.D()
        self.mechanical_system.K_constr = self.mechanical_system.K()

        K_eff = (1 - self.alpha_m)/(self.beta*self.dt**2)*self.mechanical_system.M_constr \
                + (1 - self.alpha_f)*self.gamma/(self.beta*self.dt)*self.mechanical_system.D_constr \
                + (1 - self.alpha_f)*self.mechanical_system.K_constr
        return K_eff

    def effective_force(self, q_old, dq_old, v_old, ddq_old, t, t_old):
        '''
        Return actual effective force for linear generalized-alpha time integration scheme.
        '''

        t_f = (1 - self.alpha_f)*t + self.alpha_f*t_old

        f_ext_f = self.mechanical_system.f_ext(None, None, t_f)

        F_eff = ((1 - self.alpha_m)/(self.beta*self.dt**2)*self.mechanical_system.M_constr \
                + (1 - self.alpha_f)*self.gamma/(self.beta*self.dt)*self.mechanical_system.D_constr \
                - self.alpha_f*self.mechanical_system.K_constr)@q_old \
                + ((1 - self.alpha_m)/(self.beta*self.dt)*self.mechanical_system.M_constr \
                - (self.gamma*(self.alpha_f - 1) + self.beta)/self.beta*self.mechanical_system.D_constr)@dq_old \
                + (-(0.5*(self.alpha_m - 1) + self.beta)/self.beta*self.mechanical_system.M_constr \
                - (1 - self.alpha_f)*(self.beta - 0.5*self.gamma)*self.dt/self.beta*self.mechanical_system.D_constr)@ddq_old \
                + f_ext_f
        return F_eff

    def update(self, q, q_old, dq_old, v_old, ddq_old):
        '''
        Return actual velocity and acceleration for linear generalized-alpha time integration scheme.
        '''

        ddq = 1/(self.beta*self.dt**2)*(q - q_old) - 1/(self.beta*self.dt)*dq_old - (0.5 - self.beta)/self.beta*ddq_old
        v = np.empty((0,0))
        dq = dq_old + self.dt*((1 - self.gamma)*ddq_old + self.gamma*ddq)
        return dq, v, ddq


class JWHAlphaLinearDynamicsSolver(LinearDynamicsSolver):
    '''
    Class for solving the linear dynamic problem of the mechanical system linearized around zero-displacement using the
    JWH-alpha time integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolver
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = True

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (3 - self.rho_inf) / (2 * (1 + self.rho_inf))
        self.alpha_f = 1 / (1 + self.rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def set_parameters(self, alpha_m, alpha_f, gamma):
        '''
        Overwrite standard parameters for the nonlinear JWH-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        gamma : float

        Second-order accuracy for gamma = 1/2 + alpha_m - alpha_f. Unconditional stability for
        alpha_m >= alpha_f >= 1/2.
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        return

    def effective_stiffness(self):
        '''
        Return effective stiffness matrix for linear JWH-alpha time integration scheme.
        '''

        self.mechanical_system.M()
        self.mechanical_system.D_constr = self.mechanical_system.D()
        self.mechanical_system.K_constr = self.mechanical_system.K()

        K_eff = self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2)*self.mechanical_system.M_constr \
                + self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.D_constr \
                + self.alpha_f*self.mechanical_system.K_constr
        return K_eff

    def effective_force(self, q_old, dq_old, v_old, ddq_old, t, t_old):
        '''
        Return actual effective force for linear JWH-alpha time integration scheme.
        '''

        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        f_ext_f = self.mechanical_system.f_ext(None, None, t_f)

        F_eff = (-(1 - self.alpha_f)*self.mechanical_system.K_constr \
                + self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.D_constr \
                + self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2)*self.mechanical_system.M_constr)@q_old \
                + (-(self.gamma - self.alpha_m)/self.gamma*self.mechanical_system.D_constr \
                - self.alpha_m*(self.gamma - self.alpha_m)/(self.alpha_f*self.gamma**2*self.dt)*self.mechanical_system.M_constr)@dq_old \
                + (self.alpha_m/(self.alpha_f*self.gamma*self.dt)*self.mechanical_system.M_constr)@v_old \
                + (-(self.gamma - self.alpha_m)/self.gamma*self.mechanical_system.M_constr)@ddq_old \
                + f_ext_f
        return F_eff

    def update(self, q, q_old, dq_old, v_old, ddq_old):
        '''
        Return actual velocity and acceleration for linear JWH-alpha time integration scheme.
        '''

        dq = 1/(self.gamma*self.dt)*(q - q_old) + (self.gamma - 1)/self.gamma*dq_old
        v = self.alpha_m/self.alpha_f*dq + (1 - self.alpha_m)/self.alpha_f*dq_old \
            + (self.alpha_f - 1)/self.alpha_f*v_old
        ddq = 1/(self.gamma*self.dt)*(v - v_old) + (self.gamma - 1)/self.gamma*ddq_old
        return dq, v, ddq


class JWHAlphaNonlinearDynamicsSolverStateSpace(NonlinearDynamicsSolverStateSpace):
    '''
    Class for solving the nonlinear dynamic problem of the state-space system using the JWH-alpha time integration
    scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolverStateSpace
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
            numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (3 - self.rho_inf)/(2*(1 + self.rho_inf))
        self.alpha_f = 1/(1 + self.rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def set_parameters(self, alpha_m, alpha_f, gamma):
        '''
        Overwrite standard parameters for the nonlinear JWH-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        gamma : float

        Second-order accuracy for gamma = 1/2 + alpha_m - alpha_f. Unconditional stability for
        alpha_m >= alpha_f >= 1/2.
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        return

    def predict(self, x, dx):
        '''
        Predict variables for the nonlinear JWH-alpha time integration scheme.
        '''

        x += self.dt*(1 - self.gamma)*dx
        dx *= 0
        return x, dx

    def newton_raphson(self, x, dx, t, x_old, dx_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear JWH-alpha time integration scheme.
        '''

        if self.mechanical_system.E_constr is None:
            self.mechanical_system.E(x, t)

        dx_m = self.alpha_m*dx + (1 - self.alpha_m)*dx_old
        x_f = self.alpha_f*x + (1 - self.alpha_f)*x_old
        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        A_f, F_f = self.mechanical_system.A_and_F(x_f, t_f)

        F_ext_f = self.mechanical_system.F_ext(x_f, t_f)

        Jac = self.alpha_f*A_f - self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.E_constr
        Res = F_f + F_ext_f - self.mechanical_system.E_constr@dx_m

        return Jac, Res, F_ext_f

    def correct(self, x, dx, delta_x):
        '''
        Correct variables for the nonlinear JWH-alpha time integration scheme.
        '''

        x += delta_x
        dx += 1/(self.gamma*self.dt)*delta_x
        return x, dx


class JWHAlphaLinearDynamicsSolverStateSpace(LinearDynamicsSolverStateSpace):
    '''
    Class for solving the linear dynamic problem of the state-space system linearized around zero-state using the
    JWH-alpha time integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolverStateSpace
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
            numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (3 - self.rho_inf) / (2 * (1 + self.rho_inf))
        self.alpha_f = 1 / (1 + self.rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def set_parameters(self, alpha_m, alpha_f, gamma):
        '''
        Overwrite standard parameters for the nonlinear JWH-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        gamma : float

        Second-order accuracy for gamma = 1/2 + alpha_m - alpha_f. Unconditional stability for
        alpha_m >= alpha_f >= 1/2.
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        return

    def effective_stiffness(self):
        '''
        Return effective stiffness matrix for linear JWH-alpha time integration scheme.
        '''

        self.mechanical_system.E()
        self.mechanical_system.A_constr = self.mechanical_system.A()

        K_eff = self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.E_constr \
                - self.alpha_f*self.mechanical_system.A_constr
        return K_eff

    def effective_force(self, x_old, dx_old, t, t_old):
        '''
        Return actual effective force for linear JWH-alpha time integration scheme.
        '''

        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        F_ext_f = self.mechanical_system.F_ext(None, t_f)

        F_eff = (self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.E_constr \
                 + (1 - self.alpha_f)*self.mechanical_system.A_constr)@x_old \
                 + ((self.alpha_m - self.gamma)/self.gamma*self.mechanical_system.E_constr)@dx_old \
                 + F_ext_f
        return F_eff

    def update(self, x, x_old, dx_old):
        '''
        Return actual derivative for linear JWH-alpha time integration scheme.
        '''

        dx = 1/(self.gamma*self.dt)*(x - x_old) + (self.gamma - 1)/self.gamma*dx_old
        return dx


class ConstrainedNonlinearDynamicsSolver(NonlinearDynamicsSolver):
    pass


# This could be a dictionary for a convenient mapping of scheme names (strings) to their solver classes
# solvers_available = {'NonlinearStatics' : NonlinearStaticsSolver,
#                      'LinearStatics' : LinearStaticsSolver,
#                      'NonlinearGeneralizedAlpha' : GeneralizedAlphaNonlinearDynamicsSolver,
#                      'NonlinearJWHAlpha' : JWHAlphaNonlinearDynamicsSolver,
#                      'LinearGeneralizedAlpha' : GeneralizedAlphaLinearDynamicsSolver,
#                      'LinearJWHAlpha' : JWHAlphaLinearDynamicsSolver}


# def choose_solver(mechanical_system, **options):
#     if type(mechanical_system) == MechanicalSystem:
#         solvertype = options['solvertype']
#
#     solver = solvers_available[solvertype](**options)
#     return solver

