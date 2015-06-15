# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:14:11 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import time

from matplotlib import pyplot as plt
# make amfe running
import sys
sys.path.insert(0,'../..')
import amfe
from amfe import model_reduction as mor
from model_u import *

#%%
export_path = 'results/time_integration_1/time_integration'

# time integration: reference
my_integrator = amfe.NewmarkIntegrator(verbose=True)
#my_integrator.delta_t = 0.01
my_integrator.set_mechanical_system(my_system)

q0 = sp.zeros(ndof)
dq0 = sp.zeros(ndof)
T = sp.arange(0, 3, 0.001)
my_integrator.integrate_nonlinear_system(q0, dq0, T)

my_system.export_paraview('results/reference_time_integration_2/reference')


# POD
u_list = []
for u in my_system.u_output:
    u_tmp = my_system.b_constraints.T.dot(u)
    u_list.append(u_tmp.reshape(-1))

u_list = sp.array(u_list).T
U, sigma, V_pod = sp.linalg.svd(u_list)
plt.semilogy(sigma)
sp.save('u_reference_2', u_list)



# System matrices
K = my_system.K_global()
K = K.toarray()

M = my_system.M_global()
M = M.toarray()


# eigenvalue analysis
lambda_, V = sp.linalg.eigh(K, M)
omega = sp.sqrt(lambda_)

mor.principal_angles(V[:, :10], U[:,:10])





