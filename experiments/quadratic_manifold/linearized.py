"""Comparing run for showing the results of the linearized system"""

import time
import os
import numpy as np

import amfe

from experiments.quadratic_manifold.benchmark_bar_arc import benchmark_system, \
    amfe_dir, alpha
#from benchmark_u import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/linearized' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

# time integration
ndof = benchmark_system.dirichlet_class.no_of_constrained_dofs


#%%
# try the linear system:
q0, dq0 = np.zeros(ndof), np.zeros(ndof)
time_range = np.arange(0, 0.4, 1E-3)
amfe.integrate_linear_system(benchmark_system, q0, dq0, time_range, dt=1E-4, 
                             alpha=alpha)

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)
