# -*- coding: utf-8 -*-
'''
Just a simple test routine for checking if the integration scheme works properly.

'''

import numpy as np
import scipy as sp


# make amfe running
import sys
sys.path.insert(0,'..')

import amfe

#%%

def test_intgrator():
    c1 = 10
    c2 = 20
    c3 = 10
    c4 = 0
    K = np.array([[c1 + c2,-c2,0],
                  [-c2 , c2 + c3, -c3],
                  [0, -c3, c3 + c4]])

    def my_k(q):
        return K


    M = np.diag([3,1,2])

    def f_non(q):
        return K.dot(q) # + np.array([c1*q[0]**3, 0, 0])

    omega = 2*np.pi*1
    amplitude = 5
    def f_ext(q, dq, t):
        return np.array([0, 0., amplitude*np.cos(omega*t)])




    q_start = np.array([1, 0, 2.])*0
    dq_start = q_start*0

    T = np.arange(0,5,0.05)

    my_integrator = amfe.NewmarkIntegrator()
    my_integrator.set_nonlinear_model(f_non, my_k, M, f_ext)
    my_integrator.verbose = True
    q, dq = my_integrator.integrate_nonlinear_system(q_start, dq_start, T)

    from matplotlib import pyplot
    pyplot.plot(T, q)
    #pyplot.plot(T, dq)


 #%%