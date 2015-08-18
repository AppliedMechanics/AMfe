# -*- coding: utf-8 -*-
"""
Idea here is to run tests with all Elements and compare that to the results of analytical solutions
"""


import sys
import time

import numpy as np
import scipy as sp


# this is the way how to import the amfe-toolbox
sys.path.insert(0,'..')
import amfe


import matplotlib.pyplot as plt


# Mesh generation

# Building the mechanical system
my_mechanical_system = amfe.MechanicalSystem()
my_mechanical_system.load_mesh_from_gmsh('../meshes/test_meshes/bar_Quad4_simple.msh')


#%%

# Boundary handling
bottom_line_indices = my_mechanical_system.mesh_class.boundary_list[3]
top_line_indices = my_mechanical_system.mesh_class.boundary_line_list[1]
bottom_fixation_x = [None, [amfe.node2total(i, 0) for i in bottom_line_indices], None]
bottom_fixation_y = [None, [amfe.node2total(i, 1) for i in bottom_line_indices], None]
top_fixation_x = [amfe.node2total(top_line_indices[0], 0), [amfe.node2total(i, 0) for i in top_line_indices], None]
top_fixation_y = [amfe.node2total(top_line_indices[0], 1), [amfe.node2total(i, 1) for i in top_line_indices], None]

dirichlet_boundary_list = [bottom_fixation_x, bottom_fixation_y, top_fixation_x, top_fixation_y]
my_mechanical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)


# static force in y-direction
my_neumann_boundary_list = [[[amfe.node2total(top_line_indices[0], 0),], 'static', (5E11, ), None]]
my_mechanical_system.apply_neumann_boundaries(my_neumann_boundary_list)


##%%
## Investigation on the solution stuff...
#
K = my_mechanical_system.K_global()


#%%

# static solution
amfe.solve_nonlinear_displacement(my_mechanical_system, 40, smplfd_nwtn_itr=1)
#amfe.solve_linear_displacement(my_mechanical_system)
export_path = '../results/tests/bar_Quad4' + time.strftime("_%Y%m%d_%H%M%S") + '/bar_Quad4'
my_mechanical_system.export_paraview(export_path)



