#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
JWH-alpha state-space nonlinear dynamics solver.
"""

from .nonlinear_dynamics_solver_state_space import NonlinearDynamicsSolverStateSpace

__all__ = [
    'JWHAlphaNonlinearDynamicsSolverStateSpace'
]


class JWHAlphaNonlinearDynamicsSolverStateSpace(NonlinearDynamicsSolverStateSpace):
    '''
    Class for solving the nonlinear dynamic problem of the state-space system using the JWH-alpha time integration
    scheme.

    Parameters
    ----------
    mechanical_system : Instance of MechanicalSystemStateSpace
        State-space system to be solved.
    options : Dictionary
        Options for solver:
        see class NonlinearDynamicsSolverStateSpace
        rho_inf : float
            High frequency spectral radius. 0 <= rho_inf <= 1. Default value rho_inf = 0.9.

    References
    ----------
       [1]  J. Chung and G. Hulbert (1993): A time integration algorithm for structural dynamics with improved
            numerical dissipation: the generalized-alpha method. Journal of Applied Mechanics 60(2) 371--375.
       [2]  K.E. Jansen, C.H. Whiting and G.M. Hulbert (2000): A generalized-alpha method for integrating the filtered
            Navier-Stokes equations with a stabilized finite element method. Computer Methods in Applied Mechanics and
            Engineering 190(3) 305--319. DOI 10.1016/S0045-7825(00)00203-6.
       [3]  C. Kadapa, W.G. Dettmer and D. Perić (2017): On the advantages of using the first-order generalised-alpha
            scheme for structural dynamic problems. Computers and Structures 193 226--238.
            DOI 10.1016/j.compstruc.2017.08.013.
       [4]  M. Géradin and D.J. Rixen (2015): Mechanical vibrations. Theory and application to structural dynamics.
            ISBN 978-1-118-90020-8.
    '''

    def __init__(self, mechanical_system, **options):
        super().__init__(mechanical_system, **options)

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

    def predict(self, x, dx):
        '''
        Predict variables for the nonlinear JWH-alpha time integration scheme.
        '''

        x += self.dt * (1 - self.gamma) * dx
        dx *= 0
        return x, dx

    def newton_raphson(self, x, dx, t, x_old, dx_old, t_old):
        '''
        Return actual Jacobian and residuum for the nonlinear JWH-alpha time integration scheme.
        '''

        dx_m = self.alpha_m * dx + (1 - self.alpha_m) * dx_old
        x_f = self.alpha_f * x + (1 - self.alpha_f) * x_old
        t_f = self.alpha_f * t + (1 - self.alpha_f) * t_old

        E = self.mechanical_system.E(x, t)
        A_f, F_f = self.mechanical_system.A_and_F(x_f, t_f)
        F_ext_f = self.mechanical_system.F_ext(x_f, t_f)

        Jac = self.alpha_f * A_f - self.alpha_m / (self.gamma * self.dt) * E
        Res = F_f + F_ext_f - E @ dx_m
        return Jac, Res, F_ext_f

    def correct(self, x, dx, delta_x):
        '''
        Correct variables for the nonlinear JWH-alpha time integration scheme.
        '''

        x += delta_x
        dx += 1 / (self.gamma * self.dt) * delta_x
        return x, dx
