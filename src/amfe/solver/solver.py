#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abstract super class of all solvers.
"""

from copy import deepcopy
from time import time
from amfe.linalg.linearsolvers import *
from amfe.solver.nonlinear_solver import *
from amfe.solver.translators import MechanicalSystem
from amfe.solver.initializer import *
from amfe.component import StructuralComponent
from amfe.solver.integrator import *

from math import isclose


__all__ = [
    'Solver',
    'SolverFactory'
]


# Solver-class with stepping
class Solver:
    def __init__(self):
        return

    def solve(self, *args, **kwargs):
        """

        Parameters
        ----------
        args
            args[0] should be write_callback!
        kwargs

        Returns
        -------

        """
        return


class TransientSolver(Solver):
    def __init__(self, integrator, accelerationinitializer):
        super().__init__()
        self._integrator = integrator
        self._accelerationinitializer = accelerationinitializer

    def solve(self, write_callback, t0, q0, dq0, t_end, t_eval=None):

        # Initialize first timestep
        t, q, dq, ddq = self._initialize(t0, q0, dq0)

        # Call write timestep for initial conditions
        write_callback(t, q, dq, ddq)

        # --- Run timeintegration ---
        # start time measurement
        t_clock_start = time()
        # Run Loop
        while not isclose(t, t_end) and t < t_end:
            t, q, dq, ddq = self._integrator.step(t, q, dq, ddq)
            write_callback(t, q, dq, ddq)
        # end time measurement
        t_clock_end = time()
        print('Time for solving problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return

    async def solve_async(self, write_callback, t0, q0, dq0, t_end, t_eval=None):

        # Initialize first timestep
        t, q, dq, ddq = self._initialize(t0, q0, dq0)

        # Call write timestep for initial conditions
        await write_callback(t, q, dq, ddq)

        # --- Run timeintegration ---
        # start time measurement
        t_clock_start = time()
        # Run Loop
        while t < t_end:
            t, q, dq, ddq = self._integrator.step(t, q, dq, ddq)
            await write_callback(t, q, dq, ddq)
        # end time measurement
        t_clock_end = time()
        print('Time for solving problem: {0:6.3f} seconds.'.format(t_clock_end - t_clock_start))
        return

    def _initialize(self, t0, q0, dq0):
        # --- Compute initial acceleration ---
        ddq0 = self._accelerationinitializer.get_acceleration(t0, q0, dq0)

        # --- Set initial state ---
        # Call set initial conditions to store initial state
        t = t0
        q = q0
        dq = dq0
        ddq = ddq0
        return t, q, dq, ddq


class SolverFactory:
    analysis_types = ['static',
                      'transient',
                      'modal'
                      ]

    linear_solvers = {'scipy-sparse': ScipySparseLinearSolver,
                      'pardiso': PardisoLinearSolver,
                      'scipy-cg': ScipyConjugateGradientLinearSolver,
                      }

    nonlinear_solvers = {'newton': NewtonRaphson()
                         }

    acceleration_intializers = ['zero',
                                'linear',
                               ]

    def __init__(self):

        self.integrators = {'genalpha': self._create_integrator_object_genalpha,
                            'newmarkbeta': self._create_integrator_object_newmarkbeta,
                            'loadstepping': self._create_integrator_object_loadstepping
                            }

        self._solver = None
        self._analysis_type = None
        self._integrator = None
        self._linear_solver = None
        self._linear_solver_kwargs = dict()
        self._no_of_timesteps = None
        self._dt_initial = None
        self._nonlinear_solver = None
        self._newton_maxiter = 10
        self._newton_atol = 1.0e-8
        self._newton_rtol = 0.0
        self._newton_track_condition_number = False
        self._newton_verbose = False
        self._newton_callback = None
        self._acceleration_initializer = None
        self._acceleration_initializer_linear_solver = None
        self._system = None
        self._alpha_f = None
        self._alpha_m = None
        self._beta = None
        self._gamma = None
        self._rho_inf = None
        self._async = False
        return

    # --------------------------------------- SETTER METHODS ---------------------------------------------------------
    def set_system(self, translator):
        if isinstance(translator, (StructuralComponent, MechanicalSystem)):
            self._system = translator
        else:
            raise NotImplementedError('The Solver Factory currently cannot handle this type of component')
        return

    def set_async(self, flag):
        if isinstance(flag, bool):
            self._async = flag

    def set_acceleration_intializer(self, key):
        if key in self.acceleration_intializers:
            self._acceleration_initializer = key

    def set_analysis_type(self, key):
        if key in self.analysis_types:
            self._analysis_type = key

    def set_integrator(self, key):
        if key in self.integrators:
            self._integrator = key

    def set_linear_solver(self, key):
        if key in self.linear_solvers:
            self._linear_solver = key

    def set_linear_solver_option(self, key, value):
        self._linear_solver_kwargs.update({key: value})

    def set_nonlinear_solver(self, key):
        if key in self.nonlinear_solvers:
            self._nonlinear_solver = key

    def set_newton_maxiter(self, no):
        self._newton_maxiter = no
        return

    def set_newton_atol(self, atol):
        self._newton_atol = atol
        return

    def set_newton_rtol(self, rtol):
        self._newton_rtol = rtol

    def set_newton_verbose(self, verbose):
        self._newton_verbose = verbose

    def set_newton_callback(self, func):
        self._newton_callback = func

    def set_newton_track_condition_number(self, flag):
        if isinstance(flag, bool):
            self._newton_track_condition_number = flag
        else:
            raise ValueError('flag must be boolean')
        return

    def set_no_of_timesteps(self, no):
        self._no_of_timesteps = no
        return

    def set_timestep_size(self, dt):
        self._dt_initial = dt
        return

    def set_dt_initial(self, dt):
        return self.set_timestep_size(dt)

    # ---------------------------------------- MAIN CREATION METHOD --------------------------------------------------
    def create_solver(self):
        """

        Returns
        -------
        solver : Solver

        """
        if self._analysis_type == 'transient':
            return self._create_transient_solver()
        elif self._analysis_type == 'static':
            return self._create_static_solver()
        elif self._analysis_type is None:
            raise ValueError('The analysis type has not been set before, call set_analysis_type(type)')
        else:
            raise NotImplementedError('Analysis type {} has not been implemented so far'.format(self._analysis_type))

    # ---------------------------------------- 2nd level DISTINGUISH ANALYSIS TYPE -----------------------------------
    def _create_static_solver(self):
        linear_solver = self.linear_solvers[self._linear_solver]()
        linear_solver_kwargs = self._linear_solver_kwargs
        if not self._system.system_is_linear:
            self._integrator = 'loadstepping'
            return self._create_transient_solver()
        else:
            return linear_solver, linear_solver_kwargs

    def _create_transient_solver(self):
        linear_solver = self.linear_solvers[self._linear_solver]()
        linear_solver_kwargs = self._linear_solver_kwargs
        if self._system.system_is_linear:
            return self._create_linear_transient_solver(linear_solver, linear_solver_kwargs)
        else:
            return self._create_nonlinear_transient_solver(linear_solver, linear_solver_kwargs)

    def _create_newton_solver(self, linear_solver, linear_solver_kwargs):
        if self._nonlinear_solver is not None:
            nonlinear_solver = deepcopy(self.nonlinear_solvers[self._nonlinear_solver])
            nonlinear_solver_options = {'linear_solver': linear_solver,
                                        'linear_solver_kwargs': linear_solver_kwargs,
                                        'atol': self._newton_atol,
                                        'rtol': self._newton_rtol,
                                        'maxiter': self._newton_maxiter,
                                        'track_condition_number': self._newton_track_condition_number,
                                        'verbose': self._newton_verbose,
                                        }
            if self._newton_callback is not None:
                nonlinear_solver_options.update({'callback': self._newton_callback})
            return nonlinear_solver, nonlinear_solver_options
        else:
            raise ValueError('Nonlinearsolver must be set')

    def _create_nonlinear_transient_solver(self, linear_solver, linear_solver_options):
        nonlinear_solver, nonlinear_solver_options = self._create_newton_solver(linear_solver, linear_solver_options)
        integrator = self._create_integrator_object()
        integrator.dt = self._dt_initial
        integration_stepper = self._create_integration_stepper(integrator, 'nonlinear')
        integration_stepper.nonlinear_solver_func = nonlinear_solver.solve
        integration_stepper.nonlinear_solver_options = nonlinear_solver_options
        accelerationinitializer = self._create_acceleration_initializer(linear_solver, linear_solver_options)
        # Create Solver Object
        return TransientSolver(integration_stepper, accelerationinitializer)

    def _create_linear_transient_solver(self, linear_solver, linear_solver_options):
        integrator = self._create_integrator_object()
        integrator.dt = self._dt_initial
        integration_stepper = self._create_integration_stepper(integrator, 'linear')
        integration_stepper.linear_solver_func = linear_solver.solve
        integration_stepper.linear_solver_options = linear_solver_options
        accelerationinitializer = self._create_acceleration_initializer(linear_solver, linear_solver_options)
        # Create Solver Object
        return TransientSolver(integration_stepper, accelerationinitializer)

    def _create_acceleration_initializer(self, linear_solver, linear_solver_kwargs):
        if self._acceleration_initializer is not None:
            if self._acceleration_initializer == 'zero':
                return NullAccelerationInitializer()
            elif self._acceleration_initializer == 'linear':
                lin_solver_func = linear_solver.solve
                return LinearAccelerationInitializer(self._system.M, self._system.f_int, self._system.f_ext,
                                                     self._system.K, self._system.D, lin_solver_func, linear_solver_kwargs)
            else:
                raise ValueError('acceleration initializer {} unknown'.format(self._acceleration_initializer))
        else:
            lin_solver_func = linear_solver.solve
            return LinearAccelerationInitializer(self._system.M, self._system.f_int, self._system.f_ext,
                                                 self._system.K, self._system.D, lin_solver_func, linear_solver_kwargs)

    # ------------------------------------------ CREATE INTEGRATOR OBJECTS -------------------------------------------
    def _create_integration_stepper(self, integrator, opt):
        if opt is 'linear':
            return LinearIntegrationStepper(integrator)
        elif opt is 'nonlinear':
            return NonlinearIntegrationStepper(integrator)
        else:
            raise ValueError('Unsupported type of integration-stepper')

    def _create_integrator_object(self):
        if self._integrator in self.integrators:
            return self.integrators[self._integrator]()
        else:
            error_msg = 'There is no integrator of type ' + self._integrator + ' implemented yet'
            raise ValueError(error_msg)

    def _create_integrator_object_genalpha(self):
        integrator = GeneralizedAlpha(self._system.M, self._system.f_int, self._system.f_ext, self._system.K,
                                      self._system.D)
        if self._alpha_f is not None:
            integrator.alpha_f = self._alpha_f
        if self._alpha_m is not None:
            integrator.alpha_m = self._alpha_m
        if self._beta is not None:
            integrator.beta = self._beta
        if self._gamma is not None:
            integrator.gamma = self._gamma
        return integrator

    def _create_integrator_object_newmarkbeta(self):
        integrator = NewmarkBeta(self._system.M, self._system.f_int, self._system.f_ext, self._system.K, self._system.D)
        if self._beta is not None:
            integrator.beta = self._beta
        if self._gamma is not None:
            integrator.gamma = self._gamma
        return integrator

    def _create_integrator_object_loadstepping(self):
        integrator = NonlinearStatic(self._system.f_int, self._system.f_ext, self._system.K)
        return integrator

    def _create_integrator_object_nonlinear_static(self):
        integrator = NonlinearStatic(self._system.f_int, self._system.f_ext, self._system.K)
        return integrator
