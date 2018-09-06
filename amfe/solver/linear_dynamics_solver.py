#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all linear dynamics solvers.
"""

import numpy as np
from time import time

from .solver import Solver
from ..mechanical_system import ConstrainedMechanicalSystem
from ..linalg.linearsolvers import PardisoSolver

__all__ = [
    'LinearDynamicsSolver'
]


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
        self.mechanical_system = mechanical_system
        self.M = self.mechanical_system.M()
        self.D = self.mechanical_system.D()
        self.K = self.mechanical_system.K()

        # read options
        if 'linear_solver' in options:
            self.linear_solver = options['linear_solver']
        else:
            if isinstance(mechanical_system,ConstrainedMechanicalSystem):
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
            #     raise ValueError('Error: Dimension of q0 is not valid for mechanical system.')
        else:
            print('Attention: No initial displacement was given, setting q0 = 0.')
            q0 = np.zeros(self.mechanical_system.dirichlet_class.no_of_constrained_dofs)
        if ('initial_conditions' in options) and ('dq0' in options['initial_conditions']):
            dq0 = options['initial_conditions']['dq0']
            # TODO: The following section is commented out because this prevents solving reduced mechanical systems
            # TODO: because these systems do not have a ndof property.
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
        self.use_additional_variable_v = None
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
        t_clock_start = time()

        # initialize variables and set parameters
        self.mechanical_system.clear_timesteps()
        t = self.t0
        q = self.initial_conditions['q0'].copy()
        dq = self.initial_conditions['dq0'].copy()
        if self.use_additional_variable_v:
            v = self.initial_conditions['dq0'].copy()
        else:
            v = np.empty((0, 0))

        # evaluate initial acceleration
        self.linear_solver.set_A(self.M)
        ddq = self.linear_solver.solve(self.mechanical_system.f_ext(q, dq, t) - self.D @ dq - self.K @ q)

        # LU-decompose effective stiffness
        K_eff = self.effective_stiffness()
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

            print('Time: {0:3.6f}.'.format(t))

            # end of time step loop

        self.linear_solver.clear()

        # end time measurement
        t_clock_end = time()
        print('Time for solving linear dynamic problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return
