# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:22:58 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import time

from matplotlib import pyplot as plt
# make amfe running
import sys
sys.path.insert(0,'..')
import amfe



gmsh_input_file = '../meshes/gmsh/bogen_grob.msh'
paraview_output_file = '../results/gmsh_bogen_grob' + time.strftime("_%Y%m%d_%H%M%S") + '/bogen_grob'

my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(gmsh_input_file)
# my_system.export_paraview(paraview_output_file)

my_element = amfe.ElementPlanar()
my_system.set_element(my_element)

start_index = amfe.node2total(54, 0)
end_index = amfe.node2total(55, 1)

# fixation
bottom_bounds_1 = [None, [0,1,2,3], None]
bottom_bounds_2 = [None, np.arange(start_index, end_index + 1), None]

other_side = [2, 3, 24, 25]
# top_bounds_x = [amfe.node2total(2,0), [amfe.node2total(i, 0) for i in other_side], None]
# top_bounds_y = [amfe.node2total(2,1), [amfe.node2total(i, 1) for i in other_side], None]

# my_dirichlet_bounds = [bottom_bounds_1, bottom_bounds_2, top_bounds_x, top_bounds_y]
my_dirichlet_bounds = [bottom_bounds_1, bottom_bounds_2]

my_system.apply_dirichlet_boundaries(my_dirichlet_bounds)

neumann_bounds = [[[amfe.node2total(i,1) for i in other_side], 'harmonic', (3E5, 100), None],
                    [[amfe.node2total(i,0) for i in other_side], 'harmonic', (1E5, 200), None]]
my_system.apply_neumann_boundaries(neumann_bounds)



# amfe.solve_linear_displacement(my_system)


ndof = my_system.ndof_global_constrained
K = my_system.K_global()
M = my_system.M_global()


lambda_, V = sp.linalg.eigh(K.toarray(), M.toarray())
omega = np.sqrt(lambda_)

# time integration
my_newmark = amfe.NewmarkIntegrator()
my_newmark.set_mechanical_system(my_system)
my_newmark.delta_t = 5E-5
my_newmark.integrate_nonlinear_system(np.zeros(ndof), np.zeros(ndof), np.arange(0,0.1,0.001))

# modal derivatives:




## plotting results
#plt.semilogy(omega)
#
#for i in range(40):
#    my_system.write_timestep(omega[i], V[:,i])



my_system.export_paraview(paraview_output_file)
