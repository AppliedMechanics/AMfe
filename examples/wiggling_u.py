# -*- coding: utf-8 -*-
"""
Created on Fri May 29 16:30:02 2015

@author: johannesr
"""


import numpy as np
import scipy as sp
import time

from matplotlib import pyplot as plt
# make amfe running
import sys
sys.path.insert(0, '..')
import amfe



gmsh_input_file = '../meshes/gmsh/bogen_grob.msh'
paraview_output_file = '../results/gmsh_bogen_grob' + time.strftime("_%Y%m%d_%H%M%S") + '/bogen_grob'

my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(gmsh_input_file)
# my_system.export_paraview(paraview_output_file)


start_index = amfe.node2total(54, 0)
end_index = amfe.node2total(55, 1)

# fixation
bottom_bounds_1 = [None, [0,1,2,3], None]
bottom_bounds_2 = [None, np.arange(start_index, end_index + 1), None]

my_dirichlet_bounds = [bottom_bounds_1, bottom_bounds_2]
my_system.apply_dirichlet_boundaries(my_dirichlet_bounds)


other_side = [2, 3, 24, 25]

neumann_bounds = [  [[amfe.node2total(i,1) for i in other_side], 'harmonic', (6E6, 20), None],
                    [[amfe.node2total(i,0) for i in other_side], 'harmonic', (2E6, 100), None]]
my_system.apply_neumann_boundaries(neumann_bounds)


ndof = my_system.ndof_global_constrained

###############################################################################
## time integration
###############################################################################

my_newmark = amfe.NewmarkIntegrator()
my_newmark.set_mechanical_system(my_system)
my_newmark.delta_t = 1E-4
my_newmark.integrate_nonlinear_system(np.zeros(ndof), np.zeros(ndof), np.arange(0,0.4,1E-4))


my_system.export_paraview(paraview_output_file)
