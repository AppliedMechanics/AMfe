#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
JWH-alpha linear dynamics solver.
"""

from .linear_dynamics_solver import LinearDynamicsSolver

__all__ = [
    'JWHAlphaLinearDynamicsSolver'
]


class JWHAlphaLinearDynamicsSolver(LinearDynamicsSolver):
    '''
    Class for solving the linear dynamic problem of the mechanical system linearized around zero-displacement using the
    JWH-alpha time integration scheme.

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

    def effective_stiffness(self):
        '''
        Return effective stiffness matrix for linear JWH-alpha time integration scheme.
        '''

        K_eff = self.alpha_m ** 2 / (self.alpha_f * self.gamma ** 2 * self.dt ** 2) * self.M + self.alpha_m / (
                self.gamma * self.dt) * self.D + self.alpha_f * self.K
        return K_eff

    def effective_force(self, q_old, dq_old, v_old, ddq_old, t, t_old):
        '''
        Return actual effective force for linear JWH-alpha time integration scheme.
        '''

        t_f = self.alpha_f * t + (1 - self.alpha_f) * t_old

        f_ext_f = self.mechanical_system.f_ext(None, None, t_f)

        F_eff = (-(1 - self.alpha_f) * self.K + self.alpha_m / (self.gamma * self.dt) * self.D + self.alpha_m ** 2 / (
                self.alpha_f * self.gamma ** 2 * self.dt ** 2) * self.M) @ q_old + (
                        -(self.gamma - self.alpha_m) / self.gamma * self.D - self.alpha_m * (
                        self.gamma - self.alpha_m) / (
                                self.alpha_f * self.gamma ** 2 * self.dt) * self.M) @ dq_old + (
                        self.alpha_m / (self.alpha_f * self.gamma * self.dt) * self.M) @ v_old + (
                        -(self.gamma - self.alpha_m) / self.gamma * self.M) @ ddq_old + f_ext_f
        return F_eff

    def update(self, q, q_old, dq_old, v_old, ddq_old):
        '''
        Return actual velocity and acceleration for linear JWH-alpha time integration scheme.
        '''

        dq = 1 / (self.gamma * self.dt) * (q - q_old) + (self.gamma - 1) / self.gamma * dq_old
        v = self.alpha_m / self.alpha_f * dq + (1 - self.alpha_m) / self.alpha_f * dq_old + (
                self.alpha_f - 1) / self.alpha_f * v_old
        ddq = 1 / (self.gamma * self.dt) * (v - v_old) + (self.gamma - 1) / self.gamma * ddq_old
        return dq, v, ddq
