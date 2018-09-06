#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Generalized-alpha nonlinear dynamics solver.
"""

import numpy as np

from .nonlinear_dynamics_solver import NonlinearDynamicsSolver
from ..linalg.norms import vector_norm

__all__ = [
    'GeneralizedAlphaNonlinearDynamicsSolver'
]


class GeneralizedAlphaNonlinearDynamicsSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the generalized-alpha time
    integration scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystem
        Mechanical system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolver
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

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
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = False

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (2 * self.rho_inf - 1) / (self.rho_inf + 1)
        self.alpha_f = self.rho_inf / (self.rho_inf + 1)
        self.beta = 0.25 * (1 - self.alpha_m + self.alpha_f) ** 2
        self.gamma = 0.5 - self.alpha_m + self.alpha_f
        return

    def set_wbz_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as WBZ-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_m set rho_inf = (1 + alpha_m)/(1 - alpha_m) with -1 <= alpha_m <= 0.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = (rho_inf - 1) / (rho_inf + 1)
        self.alpha_f = 0.0
        self.beta = 0.25 * (1 - self.alpha_m) ** 2
        self.gamma = 0.5 - self.alpha_m
        return

    def set_hht_alpha_parameters(self, rho_inf=0.9):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as HHT-alpha scheme.

        Parameters
        ----------
        rho_inf : float
            High frequency spectral radius. 1/2 <= rho_inf <= 1. Default value rho_inf = 0.9. For alternative
            parametrization via alpha_f set rho_inf = (1 - alpha_f)/(1 + alpha_f) with 0 <= alpha_m <= 1/3.
        '''

        self.rho_inf = rho_inf
        self.alpha_m = 0.0
        self.alpha_f = (1 - rho_inf) / (1 + rho_inf)
        self.beta = 0.25 * (1 + self.alpha_f) ** 2
        self.gamma = 0.5 + self.alpha_f
        return

    def set_newmark_beta_parameters(self, beta=0.25, gamma=0.5):
        '''
        Parametrize nonlinear generalized-alpha time integration scheme as Newmark-beta scheme.

        Parameters
        ----------
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
        '''

        self.rho_inf = None
        self.alpha_m = 0.0
        self.alpha_f = 0.0
        self.beta = beta
        self.gamma = gamma
        return

    def set_parameters(self, alpha_m, alpha_f, beta, gamma):
        '''
        Overwrite standard parameters for the nonlinear generalized-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        beta : float

        gamma : float

        Second-order accuracy for gamma = 1/2 - alpha_m + alpha_f. Unconditional stability for
        alpha_m <= alpha_f <= 1/2 and beta >= 1/4 + 1/2*(alpha_f - alpha_m).
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.beta = beta
        self.gamma = gamma
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear generalized-alpha time integration scheme.
        '''

        q += self.dt * dq + self.dt ** 2 * (0.5 - self.beta) * ddq
        dq += self.dt * (1 - self.gamma) * ddq
        ddq *= 0
        return q, dq, v, ddq

    def newton_raphson(self, q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear generalized-alpha time integration scheme.
        '''

        ddq_m = (1 - self.alpha_m) * ddq + self.alpha_m * ddq_old
        q_f = (1 - self.alpha_f) * q + self.alpha_f * q_old
        dq_f = (1 - self.alpha_f) * dq + self.alpha_f * dq_old
        t_f = (1 - self.alpha_f) * t + self.alpha_f * t_old

        M = self.mechanical_system.M()
        D = self.mechanical_system.D()
        K_f, f_f = self.mechanical_system.K_and_f(q_f, t_f)
        f_ext_f = self.mechanical_system.f_ext(q_f, dq_f, t_f)

        Jac = -(1 - self.alpha_m) / (self.beta * self.dt ** 2) * M - (1 - self.alpha_f) * self.gamma / (
                self.beta * self.dt) * D - (1 - self.alpha_f) * K_f
        res = f_ext_f - M @ ddq_m - D @ dq_f - f_f
        return Jac, res, f_ext_f

    def correct(self, q, dq, v, ddq, delta_q):
        '''
        Correct variables for the nonlinear generalized-alpha time integration scheme.
        '''

        q += delta_q
        dq += self.gamma / (self.beta * self.dt) * delta_q
        ddq += 1 / (self.beta * self.dt ** 2) * delta_q
        return q, dq, v, ddq

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        '''
        Returns an estimate for the absolute local time discretization error for the nonlinear generalized-alpha time
        integration scheme.
        '''
        abs_dt_err = vector_norm((self.beta - 1. / 6.) * self.dt ** 2 * (ddq - ddq_old), 2) / np.sqrt(ddq.size)
        return abs_dt_err
