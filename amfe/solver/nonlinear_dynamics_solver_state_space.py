#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all state-space nonlinear dynamics solvers.
"""

import numpy as np
from scipy.sparse import issparse
from time import time

from .solver import abort_statement, Solver
from ..linalg.norms import vector_norm
from ..linalg.linearsolvers import PardisoSolver

__all__ = [
    'NonlinearDynamicsSolverStateSpace'
]


class NonlinearDynamicsSolverStateSpace(Solver):
    '''
    General class for solving the nonlinear dynamic problem of the state-space system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver:
        linear_solver : Instance of LinearSolver
            Linear solver object for linear equation system in solver. Default PardisoSolver(...).
        initial_conditions : dict {'x0': numpy.array}
            Initial conditions/states for solver. Default 0.
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

        # read options
        if 'linear_solver' in options:
            self.linear_solver = options['linear_solver']
        else:
            # TODO: What is a good choice for constrained systems in state space?
            print('Attention: No linear solver object was given, setting linear_solver = PardisoSolver(...).')
            self.linear_solver = PardisoSolver(A=None, mtype='nonsym')

        if ('initial_conditions' in options) and ('x0' in options['initial_conditions']):
            x0 = options['initial_conditions']['x0']
            # TODO: The following section is commented out because this prevents solving reduced mechanical systems
            # TODO: because these systems do not have a ndof property.
            # if len(x0) != 2*self.mechanical_system.dirichlet_class.no_of_constrained_dofs:
            #     raise ValueError('Error: Dimension of x0 is not valid for state-space system.')
        else:
            print('Attention: No initial state was given, setting x0 = 0.')
            x0 = np.zeros(2 * self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
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
        t_clock_start = time()

        # initialize variables and set parameters
        t = self.t0
        x = self.initial_conditions['x0'].copy()
        F_ext = self.mechanical_system.F_ext(x, t)
        abs_F_ext = self.absolute_tolerance
        self.linear_solver.set_A(self.mechanical_system.E(x, t))
        dx = self.linear_solver.solve(F_ext + self.mechanical_system.F_int(x, t))
        self.iteration_info = [(t, 0., 0.)]
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
            Res_abs = vector_norm(Res, 2)
            abs_F_ext = max(abs_F_ext, vector_norm(F_ext, 2))

            # Newton-Raphson iteration loop
            iteration = 0
            while Res_abs > self.relative_tolerance * abs_F_ext + self.absolute_tolerance:

                iteration += 1

                # catch failing convergence
                if iteration > self.max_number_of_iterations:
                    iteration -= 1
                    if self.convergence_abort:
                        print(abort_statement)
                        self.iteration_info = np.array(self.iteration_info)
                        t_clock_end = time()
                        print('Time for solving nonlinear dynamic problem: {0:6.3f} seconds.' \
                              .format(t_clock_end - t_clock_start))
                        return
                    break

                # solve for state correction
                self.linear_solver.set_A(Jac)
                delta_x = -self.linear_solver.solve(Res)

                # correct variables
                x, dx = self.correct(x, dx, delta_x)

                # update system quantities
                Jac, Res, F_ext = self.newton_raphson(x, dx, t, x_old, dx_old, t_old)
                Res_abs = vector_norm(Res, 2)

                if self.write_iterations:
                    t_write = t + self.dt / 1000000 * iteration
                    self.mechanical_system.write_timestep(t_write, x.copy())

                if self.track_iterations:
                    self.iteration_info.append((t, iteration, Res_abs))

                if self.verbose:
                    if issparse(Jac):
                        cond_nr = 0.0
                    else:
                        cond_nr = np.linalg.cond(Jac)
                    print('Iteration: {0:3d}, residual: {1:6.3E}, condition: {2:6.3E}.' \
                          .format(iteration, Res_abs, cond_nr))

                # end of Newton-Raphson iteration loop

            # write output
            if output_index == self.output_frequency:
                self.mechanical_system.write_timestep(t, x.copy())
                output_index = 0

            print('Time: {0:3.6f}, iterations: {1:3d}, residual: {2:6.3E}.'.format(t, iteration, Res_abs))

            # end of time step loop

        self.linear_solver.clear()

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time()
        print('Time for solving nonlinear dynamic problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return
