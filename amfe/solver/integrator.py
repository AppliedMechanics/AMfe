#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Integrator-module. Provides a collection of integration-techniques, that provides

- stepping-functionalities for solving the timestep by calling a solver
- integrated jacobian
- integrated residual
- predictor
- corrector

In case of static problems the StaticDummyIntegrator has to be called.
"""

import numpy as np
from copy import copy

from amfe.linalg import vector_norm

__all__ = [
    'IntegratorBase',
    'NonlinearStaticIntegrator',
    'GeneralizedAlpha',
    'NewmarkBeta',
    'WBZAlpha',
    'HHTAlpha'
]


class IntegratorBase:
    def __init__(self):
        self.dt = None
        self._t_n = None
        self._q_n = None
        self._dq_n = None
        self._ddq_n = None

        self._t_p = None
        self._q_p = None
        self._dq_p = None
        self._ddq_p = None

    def step(self, t_n, q_n, dq_n, ddq_n):
        raise NotImplementedError('Step function was not implemented for subclass')


class LinearIntegrator(IntegratorBase):
    def __init__(self):
        super().__init__()
        self.linear_solver_func = None
        self.linear_solver_kwargs = dict()
        self._S = None

    def step(self, t_n, q_n, dq_n, ddq_n):
        """
        Stepper method for solving a time-step by predicting the solution, then solving the system with a selected solver and finally correcting the solution.

        Parameters
        ----------
        t_n : float
        q_n : ndarray
        dq_n : ndarray
        ddq_n : ndarray

        Returns
        -------
        t_n+1 : float
        q_n+1 : ndarray
        dq_n+1 : ndarray
        ddq_n+1 : ndarray
        """

        print('Solution of time-step ', t_n, ' started...')
        A, b = self.get_A_b(q_n, dq_n, ddq_n, t_n)
        ddq = self.linear_solver_func(A, b, **self.linear_solver_kwargs)
        self.set_correction(ddq)

        # Return the new solution
        return self._t_p, self._q_p, self._dq_p, self._ddq_p

    def set_correction(self, ddq):
        self._ddq_p = ddq

    def get_A_b(self, q_n, dq_n, ddq_n, t_n):
        raise NotImplementedError('get_A_b is not implemented')


class NonlinearIntegrator(IntegratorBase):
    def __init__(self):
        super().__init__()
        self.nonlinear_solver_func = None
        self._nonlinear_solver_options = dict()
        self._additional_callbacks = ()
        self._rtol = 0.0
        self._atol = 1e-8
        self._rtol_scaling_cache = 0.0

    @property
    def nonlinear_solver_options(self):
        atol = self._atol + self._rtol * self._rtol_scaling_cache
        nonlinear_solver_options = copy(self._nonlinear_solver_options)
        nonlinear_solver_options.update({'atol': atol})
        return nonlinear_solver_options

    @nonlinear_solver_options.setter
    def nonlinear_solver_options(self, dic):
        additional_callbacks = dic.pop('callback', ())
        if not isinstance(additional_callbacks, tuple):
            additional_callbacks = (additional_callbacks, )
        self._additional_callbacks = additional_callbacks
        self._rtol = dic.pop('rtol', self._rtol)
        self._atol = dic.pop('atol', self._atol)
        self._nonlinear_solver_options = dic

    def newton_callback(self, x_p, res):
            # Call user defined callbacks
            for additional_callback in self._additional_callbacks:
                additional_callback(x_p, res)
            # Call default_callback
            self._default_newton_callback(x_p, res)
            return

    def step(self, t_n, q_n, dq_n, ddq_n):
        """
        Stepper method for solving a time-step by predicting the solution, then solving the system with a selected solver and finally correcting the solution.
        
        Parameters
        ----------
        t_n : float
        q_n : ndarray
        dq_n : ndarray
        ddq_n : ndarray
        
        Returns
        -------
        t_n+1 : float
        q_n+1 : ndarray
        dq_n+1 : ndarray
        ddq_n+1 : ndarray
        """

        # Predict start for Newton-iteration
        self.set_prediction(q_n, dq_n, ddq_n, t_n)

        # Add right callbacks to options and correct atol
        nonlinear_solver_options = self.nonlinear_solver_options

        # Start Newton Iterations
        q_p, iteration_info = self.nonlinear_solver_func(self.residual, self._q_p, (), self.jacobian, None,
                                                         self.newton_callback, nonlinear_solver_options)

        # Correct all states with the new solution
        self.set_correction(q_p)

        # Print out Info:
        print('Time: {0:3.6f}, iterations: {1:3d}, residual: {2:6.3E}.'.format(self._t_p, iteration_info[0], iteration_info[1]))

        # Return the new solution
        return self._t_p, self._q_p, self._dq_p, self._ddq_p

    def _default_newton_callback(self, q_p, res):
        self.set_correction(q_p)

    def residual(self, q_p):
        raise NotImplementedError('Prediction is not implemented for this Integrator')
    
    def jacobian(self, q_p):
        raise NotImplementedError('Prediction is not implemented for this Integrator')
    
    def set_prediction(self, q_n, dq_n, ddq_n, t_n):
        raise NotImplementedError('Prediction is not implemented for this Integrator')
    
    def set_correction(self, q_p):
        raise NotImplementedError('Correction is not implemented for this Integrator')


class NonlinearStaticIntegrator(NonlinearIntegrator):
    def __init__(self, f_int, f_ext, K):
        super().__init__()
        self._f_int = f_int
        self._f_ext = f_ext
        self._K = K
        
    def residual(self, q_p):
        zero_array = np.zeros_like(q_p)
        f_ext = self._f_ext(q_p, zero_array, zero_array, self._t_p)
        res = - self._f_int(q_p, zero_array, zero_array, self._t_p) + f_ext
        return res
        
    def jacobian(self, q_p):
        return -self._K(q_p, self._dq_p, self._ddq_p, self._t_p)
    
    def set_prediction(self, q_n, dq_n, ddq_n, t_n):
        zero_array = np.zeros_like(q_n)
        self._t_n = t_n
        self._q_n = q_n
        self._dq_n = zero_array
        self._ddq_n = zero_array

        self._t_p = self._t_n + self.dt
        self._q_p = q_n.copy()
        self._dq_p = zero_array
        self._ddq_p = zero_array

        self._rtol_scaling_cache = vector_norm(self._f_ext(self._q_p, self._dq_p, self._ddq_p, self._t_p))
        return
    
    def set_correction(self, q_p):
        self._q_p = q_p
        self._t_p = self._t_n + self.dt
        return


class GeneralizedAlpha(NonlinearIntegrator):
    def __init__(self, M, f_int, f_ext, K, D, alpha_m=0.4210526315789474, alpha_f=0.4736842105263158,
                 beta=0.27700831024930755, gamma=0.5526315789473684):
        """
        Generalized-alpha integration scheme.
    
        Parameters
        ----------
        M : function
            Mass Matrix function, signature M(q, dq, ddq, t)
        f_int : function
            Internal restoring force function, signature f_int(q, dq, ddq, t)
        f_ext : function
            External force function, signature, f_ext(q, dq, ddq, t)
        K : function
            Jacobian of f_int, signature K(q, dq, ddq, t)
        D : function
            Linear viscous damping matrix, signature D(q, dq, ddq, t)
        alpha_m : float
            Mass-type matrix shifting-factor. Default value is calculated from rho_inf.
        alpha_f : float
            Internal-forces shifting-factor. Default value is calculated from rho_inf.
        beta : float
            Newmark-parameter. Default value is calculated from alpha_m and alpha_f.
        gamma : float
            Newmark-parameter. Default value is calculated from alpha_m and alpha_f.

        Notes
        -----
        Default parameters are set in this way:
        Assuming rho_inf = 0.9 all Parameters are set accordingly. These formulas help:

            alpha_m = (2 * rho_inf - 1) / (rho_inf + 1)
            alpha_f = rho_inf / (rho_inf + 1)
            beta = 0.25 * (1 - alpha_m + alpha_f) ** 2
            gamma = 0.5 - alpha_m + alpha_f

        References
        ----------
           [1]  N.M. Newmark (1959): A method of computation for structural dynamics. Journal of the Engineering Mechanics
                Division (Proceedings of the American Society of Civil Engineers) 85 67--94.
           [2]  H.M. Hilber, T.J.R. Hughes and R.L. Taylor (1977): Improved numerical dissipation for time integration
                algorithms in structural dynamics. Earthquake Engineering and Structural Dynamics 5(3) 283--292.
                DOI: 10.1002/eqe.4290050306.
           [3]  W.L. Wood, M. Bossak and O.C. Zienkiewicz (1980): An alpha modification of Newmark's method. International
                Journal for Numerical Methods in Engineering 15(10) 1562--1566. DOI: 10.1002/nme.1620151011.
           [4]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
                numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
                DOI: 10.1115/1.2900803.
           [5]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
                ISBN 978-1-118-90020-8.
        """

        super().__init__()
        # Set function handles for calling in residual and jacobian
        self.M = M
        self.f_int = f_int
        self.f_ext = f_ext
        self.K = K
        self.D = D

        # Set timeintegration parameters
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.beta = beta
        self.gamma = gamma

    @staticmethod
    def _get_midstep(alpha, x_n, x_p):
        return (1 - alpha) * x_p + alpha * x_n
                
    def residual(self, q_p):
        """
        Return residual for the generalized-alpha time integration scheme.
        """

        t_m = self._get_midstep(self.alpha_m, self._t_n, self._t_p)
        q_m = self._get_midstep(self.alpha_m, self._q_n, q_p)
        dq_m = self._get_midstep(self.alpha_m, self._dq_n, self._dq_p)
        ddq_m = self._get_midstep(self.alpha_m, self._ddq_n, self._ddq_p)

        t_f = self._get_midstep(self.alpha_f, self._t_n, self._t_p)
        q_f = self._get_midstep(self.alpha_f, self._q_n, q_p)
        dq_f = self._get_midstep(self.alpha_f, self._dq_n, self._dq_p)
        ddq_f = self._get_midstep(self.alpha_f, self._ddq_n, self._ddq_p)

        M = self.M(q_m, dq_m, t_m)
        D = self.D(q_f, dq_f, ddq_f, t_f)
        f_int_f = self.f_int(q_f, dq_f, ddq_f, t_f)
        f_ext_f = self.f_ext(q_f, dq_f, ddq_f, t_f)

        self._rtol_scaling_cache = vector_norm(f_ext_f)
        res = f_ext_f - M @ ddq_m - D @ dq_f - f_int_f
        return res

    def jacobian(self, q_p):
        """
        Return Jacobian for the generalized-alpha time integration scheme.
        """
        t_m = self._get_midstep(self.alpha_m, self._t_n, self._t_p)
        q_m = self._get_midstep(self.alpha_m, self._q_n, q_p)
        dq_m = self._get_midstep(self.alpha_m, self._dq_n, self._dq_p)

        t_f = self._get_midstep(self.alpha_f, self._t_n, self._t_p)
        q_f = self._get_midstep(self.alpha_f, self._q_n, q_p)
        dq_f = self._get_midstep(self.alpha_f, self._dq_n, self._dq_p)
        ddq_f = self._get_midstep(self.alpha_f, self._ddq_n, self._ddq_p)

        M = self.M(q_m, dq_m, t_m)
        D = self.D(q_f, dq_f, ddq_f, t_f)
        K = self.K(q_f, dq_f, ddq_f, t_f)

        Jac = -(1 - self.alpha_m) / (self.beta * self.dt ** 2) * M - (1 - self.alpha_f) * self.gamma / (
                self.beta * self.dt) * D - (1 - self.alpha_f) * K

        return Jac

    def set_prediction(self, q_n, dq_n, ddq_n, t_n):
        """
        Predict variables for the generalized-alpha time integration scheme.
        """
        self._t_n = t_n
        self._q_n = q_n.copy()
        self._dq_n = dq_n.copy()
        self._ddq_n = ddq_n.copy()

        self._q_p = self._q_n + self.dt * dq_n + self.dt ** 2 * (0.5 - self.beta) * ddq_n
        self._dq_p = self._dq_n + self.dt * (1 - self.gamma) * ddq_n
        self._ddq_p = np.zeros_like(self._q_p)
        self._t_p = t_n + self.dt

        self._rtol_scaling_cache = vector_norm(self.f_ext(self._q_p, self._dq_p, self._ddq_p, self._t_p))
        return
    
    def set_correction(self, q_p):
        """
        Correct variables for the generalized-alpha time integration scheme.
        """
        delta_q_p = q_p - self._q_p

        self._q_p[:] = q_p[:]
        self._dq_p += self.gamma / (self.beta * self.dt) * delta_q_p
        self._ddq_p += 1 / (self.beta * self.dt ** 2) * delta_q_p
        return
# Not tested yet    


class NewmarkBeta(GeneralizedAlpha):
    def __init__(self, M, f_int, f_ext, K, D, beta=0.25, gamma=0.5):
        """
        Newmark-beta integration scheme.

        Parameters
        ----------
        M : function
            Mass Matrix function, signature M(q, dq, ddq, t)
        f_int : function
            Internal restoring force function, signature f_int(q, dq, ddq, t)
        f_ext : function
            External force function, signature, f_ext(q, dq, ddq, t)
        K : function
            Jacobian of f_int, signature K(q, dq, ddq, t)
        D : function
            Linear viscous damping matrix, signature D(q, dq, ddq, t)
        beta : float
            Default value beta = 1/4.
        gamma : float
            Default value gamma = 1/2.

        Unconditional stability for beta >= gamma/2 >= 1/4. Unconditionally stability and second-order accuracy but no
        numerical damping for beta >= 1/4 and gamma = 1/2. Unconditionally stability, second-order accuracy and best
        following of phase but no numerical damping for beta = 1/4  and gamma = 1/2 (corresponds to trapezoidal rule,
        default values). Alternative parametrization as Newmark-beta scheme with alpha-damping (modified average
        constant acceleration) -- in general not second-order accurate -- via beta = 1/4*(1 + alpha)^2 and
        gamma = 1/2 + alpha with damping alpha >= 0.
        """
        alpha_m = 0.0
        alpha_f = 0.0
        super().__init__(M, f_int, f_ext, K, D, alpha_m, alpha_f, beta, gamma)


class WBZAlpha(GeneralizedAlpha):
    def __init__(self, M, f_int, f_ext, K, D, rho_inf=0.9):
        
        """
        Parametrize generalized-alpha time integration scheme as WBZ-alpha scheme.

        Parameters
        ----------
        M : function
            Mass Matrix function, signature M(q, dq, ddq, t)
        f_int : function
            Internal restoring force function, signature f_int(q, dq, ddq, t)
        f_ext : function
            External force function, signature, f_ext(q, dq, ddq, t)
        K : function
            Jacobian of f_int, signature K(q, dq, ddq, t)
        D : function
            Linear viscous damping matrix, signature D(q, dq, ddq, t)
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_m set rho_inf = (1 + alpha_m)/(1 - alpha_m) with -1 <= alpha_m <= 0.
        """
        alpha_m = (rho_inf - 1) / (rho_inf + 1)
        alpha_f = 0.0
        beta = 0.25 * (1 - alpha_m) ** 2
        gamma = 0.5 - alpha_m
        super().__init__(M, f_int, f_ext, K, D, alpha_m, alpha_f, beta, gamma)
        return   


class HHTAlpha(GeneralizedAlpha):
    def __init__(self, M, f_int, f_ext, K, D, rho_inf=0.9):
        """
        Parametrize generalized-alpha time integration scheme as HHT-alpha scheme.

        Parameters
        ----------
        M : function
            Mass Matrix function, signature M(q, dq, ddq, t)
        f_int : function
            Internal restoring force function, signature f_int(q, dq, ddq, t)
        f_ext : function
            External force function, signature, f_ext(q, dq, ddq, t)
        K : function
            Jacobian of f_int, signature K(q, dq, ddq, t)
        D : function
            Linear viscous damping matrix, signature D(q, dq, ddq, t)
        rho_inf : float
            High frequency spectral radius. 1/2 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_f set rho_inf = (1 - alpha_f)/(1 + alpha_f) with 0 <= alpha_m <= 1/3.
        """
        alpha_m = 0.0
        alpha_f = (1 - rho_inf) / (1 + rho_inf)
        beta = 0.25 * (1 + alpha_f) ** 2
        gamma = 0.5 + alpha_f
        super().__init__(M, f_int, f_ext, K, D, alpha_m, alpha_f, beta, gamma)
        return
