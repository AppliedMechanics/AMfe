# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:57:24 2016

@author: rutzmoser
"""

import time
import os
import numpy as np

import amfe

from experiments.quadratic_manifold.benchmark_bar import benchmark_system, \
    amfe_dir, alpha
#from benchmark_u import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/no_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))


#%%

# time integration
ndof = benchmark_system.dirichlet_class.no_of_constrained_dofs

my_newmark = amfe.NewmarkIntegrator(benchmark_system, alpha=alpha)
my_newmark.delta_t = 1E-4

my_newmark.integrate(np.zeros(ndof), np.zeros(ndof), np.arange(0, 0.4, 1E-4))

benchmark_system.export_paraview(paraview_output_file)
