"""Reference run where nonlinear system is executed without reduction."""

import time
import os
import numpy as np

import amfe

from experiments.quadratic_manifold.benchmark_bar_arc import benchmark_system, \
    amfe_dir, alpha
#from benchmark_u import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/no_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

# time integration
ndof = benchmark_system.dirichlet_class.no_of_constrained_dofs


#%% Integrate the nonlinear system

my_newmark = amfe.NewmarkIntegrator(benchmark_system, alpha=alpha)
my_newmark.delta_t = 1E-4
my_newmark.atol = 1E-3

my_newmark.integrate(np.zeros(ndof), np.zeros(ndof), np.arange(0, 0.4, 1E-3))

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)
