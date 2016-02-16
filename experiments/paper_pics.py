# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:37:06 2016

@author: rutzmoser
"""

import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import amfe

# % cd experiments/quadratic_manifold/
from experiments.quadratic_manifold.benchmark_bar import benchmark_system, \
    amfe_dir, alpha

paraview_output_file = os.path.join(amfe_dir, 'results/print_paper' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

#%%
M = benchmark_system.M()
benchmark_system.u_output = []
benchmark_system.T_output = []
amfe.solve_linear_displacement(benchmark_system, t=1/80)

benchmark_system.T_output[-1] = 0.5
q = benchmark_system.constrain_vec(benchmark_system.u_output[-1])
# Make displacement mass orthogonal and save it... 
q /= np.sqrt(q @ M @ q)
benchmark_system.u_output[-1] = benchmark_system.unconstrain_vec(q)
benchmark_system.u_output = []
benchmark_system.T_output = []

#%%
omega, Phi = amfe.vibration_modes(benchmark_system)
V = Phi.copy()
V[:,-1] = q
Theta = amfe.static_correction_theta(V, benchmark_system.K)

#%%
# Restoring the force for mode 2
f = K @ V[:,-1]
benchmark_system.write_timestep(1, f)
benchmark_system.export_paraview(paraview_output_file)

#%%
# plot the modal derivatives of the system
no_of_dofs, no_of_modes = V.shape
for i in range(no_of_modes):
    for j in range(i + 1):
        benchmark_system.write_timestep(i*100 + j, Theta[:,i,j])

benchmark_system.export_paraview(paraview_output_file)

#%%
# plot the modes of the system

for t, phi in enumerate(V.T):
    benchmark_system.write_timestep(t, phi)
benchmark_system.export_paraview(paraview_output_file)
