# -*- coding: utf-8 -*-
"""
File with some code snippets for the creation of the pics needed in the paper
about the quadratic manifold. 

Created: February 2016
"""

#%% Load all the necessary stuff... 
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

#%% Set up the system and compute the static displacement field

M = benchmark_system.M()
K = benchmark_system.K()
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

#%% Compute the static correction Theta

omega, Phi = amfe.vibration_modes(benchmark_system)
V = Phi.copy()
V[:,-1] = q
Theta = amfe.static_correction_theta(V, benchmark_system.K)

#%% Restoring the force for mode 2

f = K @ V[:,-1]
benchmark_system.write_timestep(1, f)
benchmark_system.export_paraview(paraview_output_file)

#%% export the modal derivatives of the system to ParaView

no_of_dofs, no_of_modes = V.shape
for i in range(no_of_modes):
    for j in range(i + 1):
        benchmark_system.write_timestep(i*100 + j, Theta[:,i,j])

benchmark_system.export_paraview(paraview_output_file)

#%% export the modes of the system to ParaView

for t, phi in enumerate(V.T):
    benchmark_system.write_timestep(t, phi)
benchmark_system.export_paraview(paraview_output_file)

#%% 


