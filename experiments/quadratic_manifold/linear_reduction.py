# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:38:15 2016

@author: rutzmoser
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

import amfe

from experiments.quadratic_manifold.benchmark_bar_arc import benchmark_system, \
    amfe_dir, alpha
#from benchmark_u import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/linear_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

no_of_modes = 10

omega, Phi = amfe.vibration_modes(benchmark_system, n=no_of_modes)
V = Phi
#%% 
# Reduce mechanical system using static correction derivatives
theta = amfe.static_correction_theta(Phi, benchmark_system.K, )
# Build new basis
ndof, n = Phi.shape
V = np.zeros((ndof, n*(n+3)//2))
V[:,:n] = Phi[:,:]
for i in range(n):
    for j in range(i+1):
        idx = n + i*(i+1)//2 + j
        V[:,idx] = theta[:,i,j] / np.sqrt(theta[:,i,j] @ theta[:,i,j])

# Deflation algorithm
U, s, V_svd = sp.linalg.svd(V, full_matrices=False)
deflating_tol = 1E-6
idx_defl = s > s[0]*deflating_tol
V = U[:,idx_defl]
ndof, no_of_modes = V.shape
#%%
my_reduced_system = amfe.reduce_mechanical_system(benchmark_system, V)
#%%

# check, if matrices are (almost) diagonal

# K = my_reduced_system.K()
#plt.matshow(K)
# M = my_reduced_system.M()
#plt.matshow(M)

#%%

# time integration

my_newmark = amfe.NewmarkIntegrator(my_reduced_system, alpha=alpha)
my_newmark.delta_t = 1E-4
my_newmark.atol = 1E-3
my_newmark.integrate(np.zeros(no_of_modes), np.zeros(no_of_modes), 
                     np.arange(0, 0.4, 1E-3))

out_file = amfe.append_to_filename(paraview_output_file)
my_reduced_system.export_paraview(out_file)
