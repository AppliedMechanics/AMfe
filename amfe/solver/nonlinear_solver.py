#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Solver-module to solve nonlinear boundary value problems.
"""

import logging
import numpy as np
from scipy.sparse import issparse
from amfe.linalg.linearsolvers import ScipySparseLinearSolver

from .tools import MemoizeJac

from ..linalg.norms import vector_norm

__all__ = [
    'NewtonRaphson'
]

abort_statement = '''
###############################################################################
#### The current computation has been aborted.                             ####
#### No convergence was gained within the number of given iteration steps. ####
###############################################################################
'''


class NewtonRaphson:
    """
    Class for solving nonlinear boundary value problem with a classical Newton-Raphson technique.
    It requires evaluation of the residuals' first derivative in every iteration.
    """
    def __init__(self):
        self._options = dict()
        self.callback = None
        return

    def _set_options(self, options):
        """
        Optional method to change convergence-properties during runtime.
        
        Parameters
        ----------
        options: dict()
            options dictionary can have following keys:
            atol : float
                absolute tolerance
            maxiter : int
                maximum number of iterations
            linear_solver : instance of one of the included linear solver classes
                Default: SciPy Sparse Solver
            track_condition_number : bool
                Flag if condition number is evaluated and printed in each iteration
            verbose : bool
                Flag for verbose mode
            
        Returns
        -------
        None
        """
        options.setdefault('atol', 1.0e-08)
        options.setdefault('maxiter', 10)
        options.setdefault('linear_solver', ScipySparseLinearSolver())
        options.setdefault('linear_solver_kwargs', dict())
        options.setdefault('track_condition_number', False)
        options.setdefault('verbose', False)

        self._options = options

        return

    @staticmethod
    def _abs(res):
        if np.isscalar(res):
            res_abs = abs(res)
        else:
            res_abs = vector_norm(res, 2)
        return res_abs
    
    def solve(self, residual, x0, args=(), jac=None, tol=None, callback=None, options=None):
        """
        Iteration loop of standard Newton-Raphson solver for nonlinear problems.
        
        Parameters
        ----------
        residual : function
            provides the current residual of the problem dependant of the current solution
            signature def residual(q), return ndarray
        x0 : ndarray
            initial solution array
        args : tuple
            extra arguments for call of residual and jac
        jac : {function, bool}
            provides the current jacobian of the problem w.r.t. the current solution
            def jac(q) return ndarray,
            if jac is boolean and True it is assumed that the residual function also provides the Jacobian
        tol : float
            tolerance (if not already set in options)
        callback : function, optional
            Optional callback function. It is called AFTER every iteration as callback(x, f), where x is the
            current solution  and f the corresponding residual
        options : dict
            further options
            
        Returns
        -------
        q : ndarray
            solution
        iteration : int
            iterations, needed to find the solution
        """
        # Convert args to tuple
        if not isinstance(args, tuple):
            args = (args, )

        # Check if jac is callable and convert to MemoizeJac if jac is true
        if not callable(jac):
            if bool(jac):
                residual = MemoizeJac(residual)
                jac = residual.derivative
            else:
                jac = None

        # Check if tol is passed and write tol in atol if atol is not in options
        if tol is not None:
            if 'atol' in options:
                logger = logging.getLogger(__name__)
                logger.warning('Attention: atol option has been set in nonlinear solver options,'
                               'but it is called with another tol. The tol has no effect.'
                               'The atol in options dictionary will be used')
            options.setdefault('atol', tol)

        # Parse options
        if options is None:
            options = dict()
        self._set_options(options)

        # Set callback function
        self.callback = callback
        
        # Initialize
        iteration = 0
        q = x0.copy()
        res = residual(q, *args)
        res_abs = self._abs(res)
        if self._options['verbose']:
            print('Iteration: {0:3d}, residual: {1:6.3E}'.format(iteration, res_abs))

        while res_abs > self._options['atol']:
            iteration += 1
            if self._options['verbose']:
                print('Iteration ', iteration, ' started...')

            # catch failing convergence
            if iteration > self._options['maxiter']:
                print(abort_statement)
                return q, (iteration, res_abs)

            # Update jacobian
            Jac = jac(q, *args)

            # solve for correction
            if np.isscalar(Jac):
                delta_q = 1/Jac * -res
            else:
                delta_q = self._options['linear_solver'].solve(Jac, -res, **self._options['linear_solver_kwargs'])

            # correct variables
            q += delta_q

            # Call callback
            if callback is not None:
                self.callback(q, res)

            # Update residual
            res = residual(q, *args)
            res_abs = self._abs(res)

            # end of Newton-Raphson iteration loop
            if self._options['track_condition_number']:
                if not np.isscalar(Jac):
                    if issparse(Jac):
                        cond_nr = np.nan
                    else:
                        cond_nr = np.linalg.cond(Jac)
                else:
                    cond_nr = Jac
                print('Iteration: {0:3d}, residual: {1:6.3E}, condition: {2:6.3E}.'.format(iteration, res_abs, cond_nr))
            else:
                if self._options['verbose']:
                    print('Iteration: {0:3d}, residual: {1:6.3E}.'.format(iteration, res_abs))
        return q, (iteration, res_abs)
