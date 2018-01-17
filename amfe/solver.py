"""
Module that contains solvers for solving systems in AMfe
"""

#IDEAS:
#
# First define options for Solver
#   options1 = {'beta': 0.5, 'gamma': 0.3}
#   options2 = {'rho': 0.9, 'beta': 0.5}
#
# Second instantiate Solver instances
#   solver1 = NonlinearGeneralizedAlphaSolver(options1)
#   solver2 = LinearGeneralizedAlphaSolver(options2)
#
# Third call the solve method and pass a system to solve
#   mysys = amfe.MechanicalSystem()
#   solver1.solve(mysys)
#   solver2.solve(mysys)
#
#
# Optional: Generate Shortcut for MechanicalSystem-class mysys.solve(solver1)
#
# The first way is advantageous: Example:
# Solve Method of Solver class:
#   def solve(mechanical_system):
#       if type(mechanical_system)== 'ConstrainedSystem':
#           raise ValueError('This kind of system cannot be solved by this solver, use ConstrainedSolver for ConstrainedSystems instead')
#       K = mechanical_system.K
#       res = ...
#       solver.solve(self)
#

import numpy as np
import scipy as sp

from .mechanical_system import *
from .linalg import *

__all__ = ['choose_solver',
           'Solver',
           'NonlinearDynamicsSolver',
           'NonlinearGeneralizedAlphaSolver',
           'ConstraintSystemSolver',
           'StateSpaceSolver']

abort_statement = '''
###############################################################################
#### The current computation has been aborted. No convergence was gained
#### within the number of given iteration steps.
###############################################################################
'''


def choose_solver(mechanical_system, options):

    if type(mechanical_system) == MechanicalSystem:
        solvertype = 'GenAlpha'

    solver = solvers_available[solvertype](options)
    return solver


class Solver:
    def __init__(self, options):
        pass

class NonlinearDynamicsSolver(Solver):
    def __init__(self, options):
        super().__init__(options)

        if 'linsolver' in options:
            self.linsolver = options['linsolver']
        else:
            self.linsolver = PardisoSolver

        if 'linsolveroptions' in options:
            self.linsolveroptions = options['linsolveroptions']

    def solve(self):
        # some things to do...
        # prediction

        # solve jacobian
        J = np.array([[1,0],[0,1]])
        if self.linsolveroptions:
            solver = self.linsolver(J, options=self.linsolveroptions)

        # correction

class NonlinearGeneralizedAlphaSolver(NonlinearDynamicsSolver):
    pass

class ConstraintSystemSolver(NonlinearDynamicsSolver):
    pass

class StateSpaceSolver(Solver):
    pass

solvers_available = {'GenAlpha': NonlinearGeneralizedAlphaSolver,
                    }