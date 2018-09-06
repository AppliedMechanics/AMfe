#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all nonlinear dynamics solvers.
"""

import numpy as np
from scipy.sparse import issparse
from time import time

from .solver import abort_statement, Solver
from ..mechanical_system import ConstrainedMechanicalSystem
from ..linalg.norms import vector_norm
from ..linalg.linearsolvers import PardisoSolver

__all__ = [
    'NonlinearDynamicsSolver'
]


class NonlinearDynamicsSolver(Solver):
    '''
    General class for solving the nonlinear dynamic problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        linear_solver : Instance of LinearSolver
            Linear solver object for linear equation system in solver. Default PardisoSolver(...).
        initial_conditions : dict {'q0': numpy.array, 'dq0': numpy.array}
            Initial conditions/initial displacement and initial velocity for solver. Default 0 and 0.
        t0 : float
            Initial time. Default 0.
        t_end : float
            End time. Default 1.
        dt : float
            Time step size for time integration. Default 1e-4.
        dt_output : float
            Time step size for output. Default 1.
        relative_tolerance : float
            Default 1e-9.
        absolute_tolerance : float
            Default 1e-6.
        max_number_of_iterations : int
            Default 30.
        convergence_abort : Boolean
            Default True.
        verbose : Boolean
            If True, show some more information in command line. Default False.
        write_iterations : Boolean
            If True, write iteration steps. Default False.
        track_iterations : Boolean
            If True, save iteration infos. Default False.
    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        self.mechanical_system = mechanical_system
        self.dt_info = list()
        # read options
        if 'linear_solver' in options:
            self.linear_solver = options['linear_solver']
        else:
            if isinstance(mechanical_system, ConstrainedMechanicalSystem):
                print('Attention: No linear solver object was given, setting linear_solver = PardisoSolver() with'
                      'options for saddle point problems.')
                self.linear_solver = PardisoSolver(A=None, mtype='sid', saddle_point=True)
            else:
                print('Attention: No linear solver object was given, setting linear_solver = PardisoSolver(...).')
                self.linear_solver = PardisoSolver(A=None, mtype='sid')

        if ('initial_conditions' in options) and ('q0' in options['initial_conditions']):
            q0 = options['initial_conditions']['q0']
            # TODO: The following section is commented out because this prevents solving reduced mechanical systems,
            # TODO: because these systems do not have a ndof property.
            # if len(q0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
            #     raise ValueError('Error: Dimension of q0 not valid for mechanical system.')
        else:
            print('Attention: No initial displacement was given, setting q0 = 0.')
            q0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        if ('initial_conditions' in options) and ('dq0' in options['initial_conditions']):
            dq0 = options['initial_conditions']['dq0']
            # TODO: The following section is commented out because this prevents solving reduced mechanical systems,
            # TODO: because these systems do not have a ndof property. (See above.)
            # if len(dq0) != self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
            #     raise ValueError('Error: Dimension of dq0 is not valid for mechanical system.')
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
        self.iteration_info = list()
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
        t_clock_start = time()

        # initialize variables and set parameters
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
            res_abs = vector_norm(res, 2)
            abs_f_ext = max(abs_f_ext, vector_norm(f_ext, 2))

            # Newton-Raphson iteration loop
            iteration = 0
            while res_abs > self.relative_tolerance * abs_f_ext + self.absolute_tolerance:

                iteration += 1

                # catch failing convergence
                if iteration > self.max_number_of_iterations:
                    iteration -= 1
                    if self.convergence_abort:
                        print(abort_statement)
                        t_clock_end = time()
                        print('Time for solving nonlinear dynamic problem: {0:6.3f} seconds.' \
                              .format(t_clock_end - t_clock_start))
                        return
                    break

                # solve for displacement correction
                self.linear_solver.set_A(Jac)
                delta_q = -self.linear_solver.solve(res)

                # correct variables
                q, dq, v, ddq = self.correct(q, dq, v, ddq, delta_q)

                # update system quantities
                Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                res_abs = vector_norm(res, 2)

                if self.write_iterations:
                    t_write = t + self.dt / 1000000 * iteration
                    self.mechanical_system.write_timestep(t_write, q.copy())

                if self.track_iterations:
                    self.iteration_info.append((t, iteration, res_abs))

                if self.verbose:
                    if issparse(Jac):
                        cond_nr = 0.0
                    else:
                        cond_nr = np.linalg.cond(Jac)
                    print('Iteration: {0:3d}, residual: {1:6.3E}, condition: {2:6.3E}.' \
                          .format(iteration, res_abs, cond_nr))

                # end of Newton-Raphson iteration loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, q.copy())
                output_index = 0

            print('Time: {0:3.6f}, iterations: {1:3d}, residual: {2:6.3E}.'.format(t, iteration, res_abs))

            # end of time step loop

        self.linear_solver.clear()

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time()
        print('Time for solving nonlinear dynamic problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return

    def solve_with_adaptive_time_stepping(self, dt_init=1.e-4, dt_min=1.e-12, dt_max=1.e0, mod_fac_min=.1,
                                          mod_fac_max=10., safety_fac=.95, fail_res_conv_fac=.5, trust_val=.1,
                                          rel_dt_err_tol=1.e-6, abs_dt_err_tol=0., max_dt_iter=13,
                                          fail_dt_conv_abort=True):
        '''
        Solves the nonlinear dynamic problem of the mechanical system with adaptive time stepping.

        Parameters
        ----------
        dt_init : float, optional
            Initial time step size. Default 1.e-4.
        dt_min : float, optional
            Minimal time step size, i.e. lower bound. Default 1.e-12.
        dt_max : float, optional
            Maximal time step size, i.e. upper bound. Default 1.e0.
        mod_fac_min : float, optional
            Minimal modification factor for time step size, i.e. lower bound. 0 < mod_fac_min <= 1 required.
            0.1 <= mod_fac_min <= 0.5 recommended. Default .1.
        mod_fac_max : float, optional
            Maximal modification factor for time step size, i.e. upper bound. 1 <= mod_fac_max < infinity required.
            1.5 <= mod_fac_max <= 5 recommended. Default 10..
        safety_fac : float, optional
            Safty factor for time step size modification. 0 < safty_factor < 1 required. 0.8 <= safty_factor < 0.95
            recommended. Default .95.
        fail_res_conv_fac : float, optional
            Modification factor for time step size for failing residual Newton-Raphson convergence.
            0 < fail_res_conv_fac < 1 required. 0.5 <= fail_res_conv_fac <= 0.8 recommended. Default .5.
        trust_val : float, optional
            Trust value for new time step size, i.e. parameter for PT1 low-pass filtering in case of increasing time
            step sizes (dt_new_used = trust_val*dt_new_calculated + (1 - trust_val)*dt_old). 0 < trust_val <= 1
            required. 0 < trust_val << 1 recommended. Default .1.
        rel_dt_err_tol : float, optional
            Relative tolerance for relative local time discretization error. Default 1.e-6.
        abs_dt_err_tol : float, optional
            Absolute tolerance for relative local time discretization error. Default 0..
        max_dt_iter : int, optional
            Maximal number of time step size adaption iterations per time step. Default 13.
        fail_dt_conv_abort : Boolean, optional
            If True abort simulation, otherwise proceed with last result to next time step when exceeding maximal
            number of time step size adaption iterations. Default True.

        References
        ----------
           [1]  O. C. Zienkiewicz and Y. M. Xie (1991): A simple error estimator and adaptive time stepping procedure
                for dynamic analysis. Earthquake Engineering & Structural Dynamics 20(9) 871--887.
                DOI: 10.1002/eqe.4290200907
           [2]  M. Mayr, W.A. Wall and M.W. Gee (2018): Adaptive time stepping for fluid-structure interaction solvers.
                Finite Elements in Analysis and Design 14155--69. DOI: 10.1016/j.finel.2017.12.002
        '''

        # start time measurement
        t_clock_start = time()

        # initialize variables and set parameters
        t = self.t0
        self.dt = dt_init
        self.dt_info = [dt_init]
        q = self.initial_conditions['q0'].copy()
        max_q = vector_norm(q, 2)
        dq = self.initial_conditions['dq0'].copy()
        if self.use_additional_variable_v:
            v = self.initial_conditions['dq0'].copy()
        else:
            v = np.empty((0, 0))
        f_ext = self.mechanical_system.f_ext(q, dq, t)
        self.linear_solver.set_A(self.mechanical_system.M(q, t))
        max_f_ext = self.absolute_tolerance
        ddq = self.linear_solver.solve(f_ext - self.mechanical_system.D(q, t) @ dq - self.mechanical_system.f_int(q, t))
        self.iteration_info = [(t, 0, 0, 0)]
        self.mechanical_system.write_timestep(t, q.copy())

        # time step loop
        time_step = 0
        output_index = 0
        while t < self.t_end:
            time_step += 1
            output_index += 1

            # save old variables
            q_old = q.copy()
            dq_old = dq.copy()
            v_old = v.copy()
            ddq_old = ddq.copy()
            t_old = t
            dt_old = self.dt
            max_q_old = max_q
            max_f_ext_old = max_f_ext

            # time step adaption loop
            dt_iter = 0  # number of time-step adaptations
            sum_res_iter = 0  # residual iterations cummulated over all timestep adaptions
            abs_dt_err = 1.e16  # local error through timestep discretization
            hit_dt_min = False  # flag for checking if minimal dt used
            no_res_conv = False  # flag for checking if residual convergence failed
            while abs_dt_err > (rel_dt_err_tol * max_q + abs_dt_err_tol):
                dt_iter += 1

                # catch failing dt convergence wrt. max iterations
                if dt_iter > max_dt_iter:
                    dt_iter -= 1
                    if fail_dt_conv_abort:
                        print(abort_statement)
                        t_clock_end = time()
                        print('Time for solving nonlinear dynamic problem {0:6.3f} seconds.'
                              .format(t_clock_end - t_clock_start))
                        return
                    break

                # reset variables
                q = q_old.copy()
                dq = dq_old.copy()
                v = v_old.copy()
                ddq = ddq_old.copy()
                t = t_old

                # predict new variables
                t += self.dt
                if t > self.t_end:
                    t = self.t_end
                    self.dt = self.t_end - t_old
                q, dq, v, ddq = self.predict(q, dq, v, ddq)
                max_q = max(max_q_old, vector_norm(q, 2))

                # do first residual evaluation
                Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                abs_res = vector_norm(res, 2)
                max_f_ext = max(max_f_ext_old, vector_norm(f_ext, 2))

                # Residual Newton-Raphson iteration loop
                res_iter = 0
                while abs_res > (self.relative_tolerance * max_f_ext + self.absolute_tolerance):
                    res_iter += 1
                    sum_res_iter += 1

                    # catch failing residual Newton-Raphson convergence
                    if res_iter > self.max_number_of_iterations:
                        res_iter -= 1
                        no_res_conv = True
                        break

                    # correct state variables
                    self.linear_solver.set_A(Jac)
                    delta_q = -self.linear_solver.solve(res)
                    q, dq, v, ddq = self.correct(q, dq, v, ddq, delta_q)

                    # do next residual evaluation
                    Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old)
                    abs_res = vector_norm(res, 2)
                    max_f_ext = max(max_f_ext_old, vector_norm(f_ext, 2))

                    if self.write_iterations:
                        t_write = t + dt_min / 1000 * sum_res_iter
                        self.mechanical_system.write_timestep(t_write, q.copy())

                    if self.track_iterations:
                        self.iteration_info.append((t, dt_iter, res_iter, abs_res))

                    if self.verbose:
                        if issparse(Jac):
                            cond_nr = 0.0
                        else:
                            cond_nr = np.linalg.cond(Jac)
                        print('        res iter {0:d}, res {1:.3e}, cond {2:.3e}.'.format(res_iter, abs_res, cond_nr))
                # end residual Newton-Raphson iteration loop

                # update time step size
                if no_res_conv:
                    # catch failing dt convergence w.r.t. hitting dt_min
                    if hit_dt_min:
                        print(abort_statement)
                        t_clock_end = time()
                        print('Time for solving nonlinear dynamic problem {0:.3f} seconds.'
                              .format(t_clock_end - t_clock_start))
                        return
                    abs_dt_err = 1.e16
                    self.dt *= fail_res_conv_fac
                    if self.dt <= dt_min:
                        self.dt = dt_min
                        hit_dt_min = True
                    no_res_conv = False
                else:
                    abs_dt_err = self.estimate_local_time_discretization_error(ddq, ddq_old)
                    kappa = np.cbrt((rel_dt_err_tol * max_q + abs_dt_err_tol) / abs_dt_err)
                    kappa *= safety_fac
                    # bound kappa by mod_fac_min and mod_fac_max
                    kappa = np.clip(kappa, mod_fac_min, mod_fac_max)
                    self.dt *= kappa
                    if self.dt <= dt_min:
                        # catch failing dt convergence wrt. hitting dt_min
                        if hit_dt_min and (self.dt < dt_min):
                            print(abort_statement)
                            t_clock_end = time()
                            print('Time for solving nonlinear dynamic problem {0:.3f} seconds.'
                                  .format(t_clock_end - t_clock_start))
                            return
                        self.dt = dt_min
                        hit_dt_min = True
                    elif self.dt > dt_max:
                        self.dt = dt_max
                    if (self.dt > dt_old) and (time_step > 1):
                        self.dt = trust_val * self.dt + (1 - trust_val) * dt_old

                if self.verbose:
                    print('    dt iter {0:d}, abs_dt_err {1:.3e}, dt {2:.3e}.'.format(dt_iter, abs_dt_err, self.dt))
            # end time step adaption loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, q.copy())
                output_index = 0

            # track final time step size
            self.dt_info.append(self.dt)

            print('t {0:.9f}, dt iter {1:d}, sum_res iter {2:d}.\n'.format(t, dt_iter, sum_res_iter))
        # end time step loop

        self.linear_solver.clear()

        # end time measurement
        t_clock_end = time()
        print('Time for solving nonlinear dynamic problem {0:.3f} seconds.'.format(t_clock_end - t_clock_start))
        return

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        pass
