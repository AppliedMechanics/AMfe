# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 18:24:13 2015

@author: johannesr

This file does not run at the moment!
Line 83 is causing an error! 
Something about a tuple out of range given to linalg.spsolve
"""

import numpy as np
import scipy as sp

import amfe
import matplotlib.pyplot as plt
l = 1.


class SimplePendulum(amfe.ConstrainedMechanicalSystem):
    '''
    Class for a constrained system that is a simple pendulum
    '''

    def __init__(self):
        self.ndof = 2
        self.ndof_const = 1
        self.m1 = 2.
        self.l = 1.
        self.g = 9.81

    def M(self, q, dq):
        return sp.array([[self.m1, 0], [0, self.m1]])

    def C(self, q, dq, t):
        x1, x2 = q
        return np.array([x1**2 + x2**2 - self.l**2])

    def B(self, q, dq, t):
        x1, x2 = q
        return np.array([[2*x1, 2*x2],])

    def f_ext(self, q, dq, t):
        return np.array([0, -self.m1*self.g])


my_pendulum = SimplePendulum()
my_constrained_solver = amfe.HHTConstrained(delta_t = 1E-3, alpha=0.01)
my_constrained_solver.set_constrained_system(my_pendulum)



q0 = np.array([l, 0.])
dq0 = np.array([0, 0.])
T = sp.arange(0, 10, 0.01)
q, dq, lambda_ = my_constrained_solver.integrate_nonlinear_system(q0, dq0, T)

#plt.plot(q)
plt.plot(q[:,0], q[:,1])

# cross checking with minimal representation
class SimplePendulumMinimal(amfe.ConstrainedMechanicalSystem):
    '''
    Class for the simple pendulum written in minimal coordinates
    '''

    def __init__(self):
        self.ndof = 1
        self.ndof_const = 0
        self.m1 = 2.
        self.l = 1.
        self.g = 9.81

    def M(self, q, dq):
        return sp.array([[self.m1*self.l**2]])

    def f_ext(self, q, dq, t):
        return sp.array([-sp.sin(q)*self.l*self.m1*self.g])

my_minimal_pendulum = SimplePendulumMinimal()
q0_min = sp.array([sp.pi/2,])
dq0_min = sp.array([0.,])
my_constrained_solver_min = amfe.HHTConstrained(delta_t = 1E-3, alpha=0.001)
my_constrained_solver_min.set_constrained_system(my_minimal_pendulum)
q_min, dq_min, lambda_min = my_constrained_solver_min.integrate_nonlinear_system(q0_min, dq0_min, T)

plt.plot(T, q)
plt.plot(T, sp.sin(q_min))
spectr = sp.fft(q[:,1])
plt.plot(abs(spectr))
plt.grid()


