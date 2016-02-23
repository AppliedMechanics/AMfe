# -*- coding: utf-8 -*-
'''
Just a simple test routine for checking if the integration scheme works properly.

'''

import numpy as np
import scipy as sp

import amfe

#%%

class DynamicalSystem():
    
    def __init__(self, K, M, f_ext):
        self.q = []
        self.t = []
        self.K = K
        self.M = M
        self.f_ext = f_ext
    
    def S_and_res(self, q, dq, ddq, dt, t, beta, gamma):
        S = self.K + 1/(beta*dt**2)*self.M
        f_ext = self.f_ext(q, dq, t)
        res = self.M @ ddq + self.K @ q - f_ext
        return S, res, f_ext
    
    def write_timestep(self, t, q):
        self.t.append(t)
        self.q.append(q)


def test_integrator():
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


    my_system = DynamicalSystem(K, M, f_ext)

    q_start = np.array([1, 0, 2.])*0
    dq_start = q_start*0

    T = np.arange(0,5,0.05)

    my_integrator = amfe.NewmarkIntegrator(my_system)
#    my_integrator.verbose = True
    my_integrator.integrate(q_start, dq_start, T)
    q = sp.array(my_system.q)
    t = sp.array(my_system.t)
    return q, t
    

if __name__ == '__main__':
    q, t = test_integrator()
    from matplotlib import pyplot
    pyplot.plot(t, q)

 #%%