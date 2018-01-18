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


# General Solver Class
# --------------------
class Solver:
    def __init__(self, options):
        pass

# General Solver Class for all Statics solver
# -------------------------------------------

class NonlinearStaticsSolver(Solver):
    def __init__(self, mechanical_system, options):
        pass

class LinearStaticsSolver(Solver):
    '''
    Solves the linear static problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be linearized at zero displacement and solved.
    '''
    def __init__(self, mechanical_system, linearsolver=PardisoSolver, options):
        super().__init__(options)
        self.mechanical_system = mechanical_system

    def solve(self,t):
        '''
        Solves the linear static problem of the mechanical system.
            
        Parameters
        ----------
        t : float
            Time for evaluation of external force in MechanicalSystem.
    
        Returns
        -------
        q : ndaray
            Static solution displacement field.
        '''

        # prepare mechanical_system
        self.mechanical_system.clear_timesteps()

        print('Assembling external force and stiffness')
        K = self.mechanical_system.K(u=None, t=t)
        f_ext = self.mechanical_system.f_ext(u=None, du=None, t=t)
        self.mechanical_system.write_timestep(0, 0*f_ext) # write undeformed state

        print('Start solving linear static problem')
        q = solve_sparse(K, f_ext)
        self.mechanical_system.write_timestep(t, q) # write deformed state
        print('Static problem solved')
        return q


# General Solver Class for all Dynamics solver
# --------------------------------------------

class NonlinearDynamicsSolver(Solver):
    def __init__(self, mechanical_system, q0, dq0, options):
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

class LinearDynamicsSolver(Solver):
    pass

# Special solvers derived from above
# ---------------------------------

class NonlinearGeneralizedAlphaSolver(NonlinearDynamicsSolver):
    pass

class ConstraintSystemSolver(NonlinearDynamicsSolver):
    pass

class StateSpaceSolver(Solver):
    pass

# This could be a dictionary for a convenient mapping of scheme names (strings) to their solver classes
solvers_available = {'GenAlpha': NonlinearGeneralizedAlphaSolver,
                    }


def choose_solver(mechanical_system, options):

    if type(mechanical_system) == MechanicalSystem:
        solvertype = 'GenAlpha'

    solver = solvers_available[solvertype](options)
    return solver
