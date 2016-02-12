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

from experiments.quadratic_manifold.benchmark_bar import benchmark_system, \
    amfe_dir, alpha
#from benchmark_u import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/linear_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

no_of_modes = 10

omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)
my_reduced_system = amfe.reduce_mechanical_system(benchmark_system, V)
#%%

# check, if matrices are (almost) diagonal

K = my_reduced_system.K()
#plt.matshow(K)
M = my_reduced_system.M()
#plt.matshow(M)

#%%

# time integration

my_newmark = amfe.NewmarkIntegrator(my_reduced_system, alpha=alpha)
my_newmark.delta_t = 1E-3

my_newmark.integrate(np.zeros(no_of_modes), np.zeros(no_of_modes), 
                     np.arange(0, 0.4, 1E-3))

my_reduced_system.export_paraview(paraview_output_file)
