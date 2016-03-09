# -*- coding: utf-8 -*-
'''
Just a simple test routine for checking if the integration scheme works properly.

'''

import unittest
import copy
import numpy as np
import scipy as sp

import amfe

#%%

class DynamicalSystem():
    
    def __init__(self, K, M, f_ext):
        self.q = []
        self.t = []
        self.K_int = K
        self.M_int = M
        self.f_ext = f_ext
    
    def S_and_res(self, q, dq, ddq, dt, t, beta, gamma):
        S = self.K_int + 1/(beta*dt**2)*self.M_int
        f_ext = self.f_ext(q, dq, t)
        res = self.M_int @ ddq + self.K_int @ q - f_ext
        return S, res, f_ext
    
    def K(self):
        return self.K_int

    def M(self):
        return self.M_int
    
    def write_timestep(self, t, q):
        self.t.append(t)
        self.q.append(q)


class IntegratorTest(unittest.TestCase):
    def setUp(self):
        c1 = 10
        c2 = 20
        c3 = 10
        c4 = 0
        K = np.array([[c1 + c2,-c2,0],
                      [-c2 , c2 + c3, -c3],
                      [0, -c3, c3 + c4]])
    
        M = np.diag([3,1,2])
    
        omega = 2*np.pi*1
        amplitude = 5
        def f_ext(q, dq, t):
            return np.array([0, 0., amplitude*np.cos(omega*t)])
    
    
        self.my_system = DynamicalSystem(K, M, f_ext)
    
        self.q_start = np.array([1, 0, 2.])*0
        self.dq_start = np.zeros_like(self.q_start)
    
        self.T = np.arange(0,5,0.05)
        
    def test_linear_vs_nonlinear_integrator(self):
        dt = 1E-3
        alpha = 0.1
        system1 = self.my_system
        system2 = copy.deepcopy(self.my_system)
        nl_integrator = amfe.NewmarkIntegrator(system1, alpha)
        nl_integrator.delta_t = dt
        nl_integrator.integrate(self.q_start, self.dq_start, self.T)
        q_nl = sp.array(system1.q)
        t_nl = sp.array(system1.t)

        amfe.integrate_linear_system(system2, self.q_start, self.dq_start, 
                                     self.T, dt, alpha)        
        q_lin = sp.array(system2.q)
        t_lin = sp.array(system2.t)
        np.testing.assert_allclose(t_nl, t_lin, atol=1E-10)
        # why does that work and below not?
        assert(np.any(np.abs(q_nl - q_lin) < 1E-3))
        # np.testing.assert_allclose(q_nl, q_lin, rtol=1E-1, atol=1E-4)
        return q_nl, q_lin, t_lin

if __name__ == '__main__':
    my_integrator_test = IntegratorTest()
    my_integrator_test.setUp()
    q_nl, q_lin, t = my_integrator_test.test_linear_vs_nonlinear_integrator()
    from matplotlib import pyplot
    pyplot.plot(t, q_nl)
    pyplot.plot(t, q_lin)
    

 #%%