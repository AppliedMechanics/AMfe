"""
Module that contains solvers for solving systems in AMfe.
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
import time

from .mechanical_system import *
from .linalg import *

__all__ = ['choose_solver',
           'Solver',
           'NonlinearStaticsSolver',
           'LinearStaticsSolver',
           'NonlinearDynamicsSolver',
           'NonlinearGeneralizedAlphaSolver',
           'NonlinearJWHAlphaSolver',
           'ConstraintSystemSolver',
           'StateSpaceSolver']

abort_statement = '''
###############################################################################
#### The current computation has been aborted. No convergence was gained
#### within the number of given iteration steps.
###############################################################################
'''


# General solver class
# --------------------
class Solver:
    '''
    General solver class for the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.
    '''

    def __init__(self, mechanical_system, options):
        self.mechanical_system = mechanical_system

        if 'linsolver' in options:
            self.linsolver = options['linsolver']
        else:
            self.linsolver = PardisoSolver

        if 'linsolveroptions' in options:
            self.linsolveroptions = options['linsolveroptions']
        return

    def solve(self):
        pass


# General solver class for all statics solver
# -------------------------------------------
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

    def __init__(self, mechanical_system, options):
        super().__init__(mechanical_system, options)
        # TBD
        return
    
    def solve(self):
        # TBD
        return

class LinearStaticsSolver(Solver):
    '''
    Class for solving the linear static problem of the mechanical system.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be linearized at zero displacement and solved.
    options : Dictionary
        Options for solver.
    '''

    def __init__(self, mechanical_system, linearsolver=PardisoSolver, options):
        super().__init__(mechanical_system, options)
        return

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
            Static displacement field (solution).
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
        Options for solver.

    References
    ----------
       [1]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
            application to structural dynamics. ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, options):
        super().__init__(mechanical_system, options)

        # read options
        if 'dt_output' in options:
            self.dt_output = options['dt_output']
        else:
            self.dt_output = None
        if 'rtol' in options:
            self.rtol = options['rtol']
        else:
            self.rtol = 1.0E-9
        if 'atol' in options:
            self.atol = options['atol']
        else:
            self.atol = 1.0E-6
        if 'n_iter_max' in options:
            self.n_iter_max = options['n_iter_max']
        else:
            self.n_iter_max = 30
        if 'conv_abort' in options:
            self.conv_abort = options['conv_abort']
        else:
            self.conv_abort = True
        if 'verbose' in options:
            self.verbose = options['verbose']
        else:
            self.verbose = False
        if 'write_iter' in options:
            self.write_iter = options['write_iter']
        else:
            self.write_iter = False
        if 'track_niter' in options:
            self.track_niter = options['track_niter']
        else:
            self.track_niter = False
        return

    def set_parameters(self, dt, options):
        pass

    def predict(self, q, dq, v, ddq):
        pass

    def newton_raphson(self, q, dq, v, ddq, q_old, dq_old, v_old, ddq_old, t, t_old):
        pass

    def correct(self, q, dq, v, ddq, delta_q):
        pass

    def solve(self, q0, dq0, t0, t_end, dt, options):
        '''
        Solves the nonlinear dynamic problem of the mechanical system.
    
        Parameters
        ----------
        q0 : ndarray
            Start displacement.
        dq0 : ndarray
            Start velocity.
        t0 : float
            Start time.
        t_end : float
            End time.
        dt : float
            Time step size.
        options : Dictionary
            Options for solver.
        '''

        # start time measurement
        t_clock_start = time.time()

        # initialize variables and set parameters
        self.mechanical_system.clear_timesteps()
        self.iteration_info = []
        t = t0
        if self.dt_output is not None:
            time_range = np.arange(t0, t_end, self.dt_output)
        else:
            time_range = np.arange(t0, t_end, dt)
        q = q0.copy()
        dq = dq0.copy()
        if use_v:
            v = dq0.copy()
        else:
            v = np.empty((0,0))
        ddq = np.zeros_like(q0)
        f_ext = np.zeros_like(q0)
        abs_f_ext = atol
        time_index = 0
        eps = 1E-13
        self.set_parameters(dt, options)

        # time step loop
        while time_index < len(time_range):
    
            # write output
            if t + eps >= time_range[time_index]:
                self.mechanical_system.write_timestep(t, q.copy())
                time_index += 1
                if time_index == len(time_range):
                    break

            # save old variables
            q_old = q.copy()
            dq_old = dq.copy()
            v_old = v.copy()
            ddq_old = ddq.copy()
            f_ext_old = f_ext.copy()
            t_old = t

            # predict new variables
            t += dt
            self.predict(q, dq, v, ddq)

            Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, q_old, dq_old, v_old, \
                                                  ddq_old, t, t_old)

            abs_f_ext = max(abs_f_ext, norm_of_vector(f_ext))
            res_abs = norm_of_vector(res)

            # Newton-Raphson iteration loop
            n_iter = 0
            while res_abs > rtol*abs_f_ext + atol:

                self.linsolver.set_A(Jac)
                delta_q = -self.linsolver.solve(res)
    
                # update variables
                self.correct(q, dq, v, ddq, delta_q)
    
                # update system
                Jac, res, f_ext = self.newton_raphson(q, dq, v, ddq, q_old, dq_old, \
                                                      v_old, ddq_old, t, t_old)

                res_abs = norm_of_vector(res)
                n_iter += 1

                if self.verbose:
                    if sp.sparse.issparse(Jac):
                        cond_nr = 0.0
                    else:
                        cond_nr = np.linalg.cond(Jac)
                    print(('Iteration: {0:3d}, residual: {1:6.3E}, condition# of '
                           + 'Jacobian: {2:6.3E}').format(n_iter, res_abs, cond_nr))

                # write iterations
                if self.write_iter:
                    t_write = t + dt/1000000*n_iter
                    self.mechanical_system.write_timestep(t_write, q.copy())

                # catch failing converge
                if n_iter > n_iter_max:
                    if self.conv_abort:
                        print(abort_statement)
                        self.iteration_info = np.array(self.iteration_info)
                        t_clock_end = time.time()
                        print('Time for time marching integration: '
                              + '{0:6.3f}s.'.format(t_clock_end - t_clock_start))
                        return

                    t = t_old
                    q = q_old.copy()
                    dq = dq_old.copy()
                    v = v_old.copy()
                    f_ext = f_ext_old.copy()
                    break

                # end of Newton-Raphson iteration loop

            print(('Time: {0:3.6f}, #iterations: {1:3d}, '
                   + 'residual: {2:6.3E}').format(t, n_iter, res_abs))
            if self.track_niter:
                self.iteration_info.append((t, n_iter, res_abs))

            # end of time step loop

        # save iteration info
        self.iteration_info = np.array(self.iteration_info)

        # end time measurement
        t_clock_end = time.time()
        print('Time for time marching integration: {0:6.3f} seconds'.format(
              t_clock_end - t_clock_start))
        return


class LinearDynamicsSolver(Solver):
    def __init__(self, mechanical_system, options):
        super().__init__(mechanical_system, options)
        # TBD
        return

    def solve(self):
        # TBD
        pass


# Special solvers derived from above
# ---------------------------------

class NonlinearGeneralizedAlphaSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the 
    generalized-alpha time integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural
            dynamics with improved numerical dissipation: the generalized-alpha method.
            Journal of Applied Mechanics 60(2) 371--375.
       [2]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
            application to structural dynamics. ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, options):
        super().__init__(mechanical_system, options)
        self.use_v = False
        return

    def set_parameters(self, dt, options):
        '''
        Set parameters for the nonlinear generalized-alpha time integration scheme.
        '''

        self.dt = dt
        if 'rho_inf' in options:
            rho_inf = options['rho_inf']
        else:
            rho_inf = 0.9

        self.alpha_m = (2*rho_inf - 1)/(rho_inf + 1)
        self.alpha_f = rho_inf/(rho_inf + 1)
        self.beta = 0.25*(1 - self.alpha_m + self.alpha_f)**2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear generalized-alpha time integration scheme.
        '''

        q += self.dt*dq + self.dt**2*(0.5 - self.beta)*ddq
        dq += self.dt*(1 - self.gamma)*ddq
        ddq *= 0
        return

    def newton_raphson(self, q, dq, v, ddq, q_old, dq_old, v_old, ddq_old, t, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear generalized-alpha time 
        integration scheme.
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
            Jac = -(1 - self.alpha_m)/(self.beta*self.dt**2) \
                    *self.mechanical_system.M_constr \
                  - (1 - self.alpha_f)*K_f
            res = f_ext_f - self.mechanical_system.M_constr@ddq_m - f_f
        else:
            Jac = -(1 - self.alpha_m)/(self.beta*self.dt**2) \
                    *self.mechanical_system.M_constr \
                  - (1 - self.alpha_f)*self.gamma/(self.beta*self.dt) \
                    *self.mechanical_system.D_constr \
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
        return


class NonlinearJWHAlphaSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the 
    JWH-alpha time integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver.

        References
        ----------
           [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha
                method for integrating the filtered Navier-Stokes equations with a
                stabilized finite element method. Computer Methods in Applied Mechanics and
                Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
           [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the
                first-order generalised-alpha scheme for structural dynamic problems.
                Computers and Structures 193 226--238. DOI 10.1016/j.compstruc.2017.08.013.
           [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and
                application to structural dynamics. ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, options):
        super().__init__(mechanical_system, options)
        self.use_v = True
        return

    def set_parameters(self, dt, options):
        '''
        Set parameters for the nonlinear JWH-alpha time integration scheme.
        '''

        self.dt = dt
        if 'rho_inf' in options:
            rho_inf = options['rho_inf']
        else:
            rho_inf = 0.9

        self.alpha_m = (3 - rho_inf)/(2*(1 + rho_inf))
        self.alpha_f = 1/(1 + rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += self.dt*(self.alpha_m - self.gamma)/self.alpha_m*dq \
             + self.dt*self.gamma/self.alpha_m*v \
             + self.alpha_f*self.dt**2*self.gamma*(1 - self.gamma)/self.alpha_m*dv
        dq += 1/self.alpha_m*(v - dq) \
              + self.alpha_f*self.dt*(1 - self.gamma)/self.alpha_m*dv
        v += self.dt*(1 - self.gamma)*dv
        dv *= 0
        return

    def newton_raphson(self, q, dq, v, ddq, q_old, dq_old, v_old, ddq_old, t, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear JWH-alpha time 
        integration scheme.
        '''

        if self.mechanical_system.M_constr is None:
            self.mechanical_system.M()

        dv_m = self.alpha_m*dv + (1 - self.alpha_m)*dv_old
        q_f = self.alpha_f*q + (1 - self.alpha_f)*q_old
        v_f = self.alpha_f*v + (1 - self.alpha_f)*v_old
        t_f = self.alpha_f*t + (1 - self.alpha_f)*t_old

        K_f, f_f = self.mechanical_system.K_and_f(q_f, t_f)

        f_ext_f = self.mechanical_system.f_ext(q_f, v_f, t_f)

        if self.mechanical_system.D_constr is None:
            Jac = -self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2) \
                    *self.mechanical_system.M_constr \
                  - self.alpha_f*K_f
            res = f_ext_f - self.mechanical_system.M_constr@dv_m - f_f
        else:
            Jac = -self.alpha_m**2/(self.alpha_f*self.gamma**2*self.dt**2) \
                    *self.mechanical_system.M_constr \
                  - self.alpha_m/(self.gamma*self.dt)*self.mechanical_system.D_constr \
                  - self.alpha_f*K_f
            res = f_ext_f - self.mechanical_system.M_constr@dv_m \
                  - self.mechanical_system.D_constr@v_f - f_f
        return Jac, res, f_ext_f

    def correct(self, q, dq, v, ddq, delta_q):
        '''
        Correct variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += delta_q
        dq += 1/(self.gamma*self.dt)*delta_q
        v += self.alpha_m/(self.alpha_f*self.gamma*self.dt)*delta_q
        dv += self.alpha_m/(self.alpha_f*self.gamma**2*self.dt**2)*delta_q
        return


class ConstraintSystemSolver(NonlinearDynamicsSolver):
    # TBD
    pass

class StateSpaceSolver(Solver):
    # TBD
    pass

# This could be a dictionary for a convenient mapping of scheme names (strings) to their solver classes
solvers_available = {'GeneralizedAlpha': NonlinearGeneralizedAlphaSolver,
                     'JWHAlpha': NonlinearJWHAlphaSolver}


def choose_solver(mechanical_system, options):

    if type(mechanical_system) == MechanicalSystem:
        solvertype = 'GeneralizedAlpha'

    solver = solvers_available[solvertype](options)
    return solver


def norm_of_vector(array):
    '''
    Compute the 2-norm of a vector.

    Parameters
    ----------
    array : ndarray
        one dimensional array

    Returns
    -------
    abs : float
        2-norm of the given array.

    '''
    return np.sqrt(array.T.dot(array))

