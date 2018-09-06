#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Linear statics solver.
"""

from time import time

from .solver import Solver
from ..mechanical_system import ConstrainedMechanicalSystem
from ..linalg.linearsolvers import PardisoSolver

__all__ = [
    'LinearStaticsSolver'
]


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
        self.mechanical_system = mechanical_system

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

        # start time measurement
        t_clock_start = time()

        # initialize variables and set parameters
        self.mechanical_system.clear_timesteps()

        print('Assembling external force and stiffness...')
        K = self.mechanical_system.K(u=None, t=self.t)
        f_ext = self.mechanical_system.f_ext(u=None, du=None, t=self.t)

        # write initial state
        self.mechanical_system.write_timestep(0.0, 0.0 * f_ext)

        print('Solving linear static problem...')
        self.linear_solver.set_A(K)
        u = self.linear_solver.solve(f_ext)

        # write deformed state
        self.mechanical_system.write_timestep(self.t, u)

        # end time measurement
        t_clock_end = time()
        print('Time for solving linear static problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return u
