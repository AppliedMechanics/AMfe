#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
JWH-alpha nonlinear dynamics solver.
"""

from .nonlinear_dynamics_solver import NonlinearDynamicsSolver

__all__ = [
    'JWHAlphaNonlinearDynamicsSolver'
]


class JWHAlphaNonlinearDynamicsSolver(NonlinearDynamicsSolver):
    '''
    Class for solving the nonlinear dynamic problem of the mechanical system using the JWH-alpha time integration
    scheme.

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
       [1]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [2]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [3]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

        self.use_additional_variable_v = True

        # read options
        if 'rho_inf' in options:
            self.rho_inf = options['rho_inf']
        else:
            print('Attention: No value for high frequency spectral radius was given, setting rho_inf = 0.9.')
            self.rho_inf = 0.9

        # set parameters
        self.alpha_m = (3 - self.rho_inf) / (2 * (1 + self.rho_inf))
        self.alpha_f = 1 / (1 + self.rho_inf)
        self.gamma = 0.5 + self.alpha_m - self.alpha_f
        return

    def set_parameters(self, alpha_m, alpha_f, gamma):
        '''
        Overwrite standard parameters for the nonlinear JWH-alpha time integration scheme.

        Parameters
        ----------
        alpha_m : float

        alpha_f : float

        gamma : float

        Second-order accuracy for gamma = 1/2 + alpha_m - alpha_f. Unconditional stability for
        alpha_m >= alpha_f >= 1/2.
        '''

        self.rho_inf = None
        self.alpha_m = alpha_m
        self.alpha_f = alpha_f
        self.gamma = gamma
        return

    def predict(self, q, dq, v, ddq):
        '''
        Predict variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += self.dt * (
                self.alpha_m - self.gamma) / self.alpha_m * dq + self.dt * self.gamma / self.alpha_m * v + self.alpha_f * self.dt ** 2 * self.gamma * (
                     1 - self.gamma) / self.alpha_m * ddq
        dq += 1 / self.alpha_m * (v - dq) + self.alpha_f * self.dt * (1 - self.gamma) / self.alpha_m * ddq
        v += self.dt * (1 - self.gamma) * ddq
        ddq *= 0
        return q, dq, v, ddq

    def newton_raphson(self, q, dq, v, ddq, t, q_old, dq_old, v_old, ddq_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear JWH-alpha time integration scheme.
        '''
        ddq_m = self.alpha_m * ddq + (1 - self.alpha_m) * ddq_old
        q_f = self.alpha_f * q + (1 - self.alpha_f) * q_old
        v_f = self.alpha_f * v + (1 - self.alpha_f) * v_old
        t_f = self.alpha_f * t + (1 - self.alpha_f) * t_old

        D = self.mechanical_system.D()
        M = self.mechanical_system.M()
        K_f, f_f = self.mechanical_system.K_and_f(q_f, t_f)
        f_ext_f = self.mechanical_system.f_ext(q_f, v_f, t_f)

        Jac = -self.alpha_m ** 2 / (self.alpha_f * self.gamma ** 2 * self.dt ** 2) * M - self.alpha_m / (
                self.gamma * self.dt) * D - self.alpha_f * K_f
        res = f_ext_f - M @ ddq_m - D @ v_f - f_f
        return Jac, res, f_ext_f

    def correct(self, q, dq, v, ddq, delta_q):
        '''
        Correct variables for the nonlinear JWH-alpha time integration scheme.
        '''

        q += delta_q
        dq += 1 / (self.gamma * self.dt) * delta_q
        v += self.alpha_m / (self.alpha_f * self.gamma * self.dt) * delta_q
        ddq += self.alpha_m / (self.alpha_f * self.gamma ** 2 * self.dt ** 2) * delta_q
        return q, dq, v, ddq

    def estimate_local_time_discretization_error(self, ddq, ddq_old):
        '''
        Returns an estimate for the absolute local time discretization error for the nonlinear JWH-alpha time
        integration scheme.
        '''

        raise NotImplementedError('Error: Adaptive time stepping is not yet implemented for time integration with the '
                         + 'JWH-alpha scheme. Use the generalized-alpha scheme instead.')
        return
