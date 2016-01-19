# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:38:15 2016

@author: rutzmoser
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

import amfe

from benchmark_example import benchmark_system, paraview_output_file


no_of_modes = 20

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

my_newmark = amfe.NewmarkIntegrator(my_reduced_system)
my_newmark.delta_t = 1E-4

t1 = time.time()

my_newmark.integrate(np.zeros(no_of_modes), 
                                      np.zeros(no_of_modes), np.arange(0, 0.4, 1E-4))

t2 = time.time()
print('Time for computation:', t2 - t1, 'seconds.')

my_reduced_system.export_paraview(paraview_output_file)

t3 = time.time()
print('Time for export:', t3 - t2, 'seconds.')
