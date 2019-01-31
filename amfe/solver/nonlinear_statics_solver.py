#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Nonlinear statics solver.
"""

import numpy as np
from time import time

from .solver import abort_statement, Solver
from ..linalg.norms import vector_norm
from ..linalg.linearsolvers import PardisoLinearSolver

__all__ = [
    'NonlinearStaticsSolver'
]


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
        self.mechanical_system = mechanical_system
        # read options
        if 'f_ext' in options:  # This option is necessary to compute the NSKTS
            self.f_ext = options['f_ext']
        else:
            self.f_ext = mechanical_system.f_ext

        if 'linear_solver' in options:
            self.linear_solver = options['linear_solver']
        else:
            print('Attention: No linear solver object was given, setting linear_solver = PardisoSolver(...).')
            self.linear_solver = PardisoLinearSolver()
            if 'constrained' in options:
                if options['constrained']:
                    self.linear_solver = PardisoLinearSolver()
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

        self.iteration_info = list()
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
        t_clock_start = time()

        # initialize variables and set parameters
        # TODO: Insert getter method for ndof. (K is not needed.) ndof = self.mechanical_system.dirichlet_class.no_...
        # TODO: ...of_constrained_dofs only works for non-reduced systems!
        K, f_int = self.mechanical_system.K_and_f()
        ndof = K.shape[0]
        u = np.zeros(ndof)
        du = np.zeros(ndof)
        self.iteration_info = []
        self.mechanical_system.clear_timesteps()
        if self.save_solution:
            # write initial state
            self.mechanical_system.write_timestep(0.0, u)
        u_output = []
        stepwidth = 1 / self.number_of_load_steps

        # load step loop: pseudo time t goes from 0 + stepwidth to 1
        for t in np.arange(stepwidth, 1.0 + stepwidth, stepwidth):

            # Calculate residuum:
            K, f_int = self.mechanical_system.K_and_f(u, t)
            f_ext = self.f_ext(u, du, t)
            res = -f_int + f_ext
            abs_f_ext = vector_norm(f_ext, 2)
            abs_res = vector_norm(res, 2)

            # Newton iteration loop
            iteration = 0
            while abs_res > self.relative_tolerance * abs_f_ext + self.absolute_tolerance:

                iteration += 1

                # catch failing convergence
                if iteration > self.max_number_of_iterations:
                    if self.convergence_abort:
                        u_output = np.array(u_output).T
                        print(abort_statement)
                        t_clock_end = time.time()
                        print('Time for solving nonlinear static problem: {0:6.3f} seconds.' \
                              .format(t_clock_end - t_clock_start))
                        return u_output
                    break

                # solve for displacement correction
                self.linear_solver.set_A(K)
                delta_u = self.linear_solver.solve(res)

                # correct displacement
                u += delta_u * self.newton_damping

                # update K, f_int and f_ext if not a simplified newton iteration
                if (iteration % self.simplified_newton_iterations) is 0:
                    K, f_int = self.mechanical_system.K_and_f(u, t)
                    f_ext = self.f_ext(u, du, t)
                # update residuum and norms
                res = -f_int + f_ext
                abs_f_ext = vector_norm(f_ext, 2)
                abs_res = vector_norm(res, 2)

                if self.write_iterations:
                    self.mechanical_system.write_timestep(t + iteration * 0.000001, u)

                if self.track_iterations:
                    self.iteration_info.append((t, iteration, abs_res))

                if self.verbose:
                    print('Step: {0:1.3f}, iteration: {1:3d}, residual: {2:6.3E}.'.format(t, iteration, abs_res))

            # end of Newton iteration loop

            if self.save_solution:
                self.mechanical_system.write_timestep(t, u)
            u_output.append(u.copy())

        # end of load step loop

        u_output = np.array(u_output).T
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time()
        print('Time for solving nonlinear static problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return u_output
