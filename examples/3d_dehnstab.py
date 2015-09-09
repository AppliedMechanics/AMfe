# -*- coding: utf-8 -*-
"""




"""


import sys
import time

import numpy as np
import scipy as sp


# this is the way how to import the amfe-toolbox
sys.path.insert(0, '..')
import amfe


import matplotlib.pyplot as plt


# Building the mechanical system
my_mechanical_system = amfe.MechanicalSystem()
my_mechanical_system.load_mesh_from_gmsh('../meshes/test_meshes/bar_Tet10_fine.msh', mesh_3d=True)


#%%


# Boundary handling
my_mechanical_system.mesh_class.boundary_information()
index_top = 16
index_bottom = 14

top = my_mechanical_system.mesh_class.boundary_list[index_top]
bottom = my_mechanical_system.mesh_class.boundary_list[index_bottom]

bottom_fixation_x = [None, [amfe.node2total(i, 0, 3) for i in bottom], None]
bottom_fixation_y = [None, [amfe.node2total(i, 1, 3) for i in bottom], None]
bottom_fixation_z = [None, [amfe.node2total(i, 2, 3) for i in bottom], None]


top_fixation_x = [amfe.node2total(top[0], 0, 3), [amfe.node2total(i, 0, 3) for i in top], None]
top_fixation_y = [amfe.node2total(top[0], 1, 3), [amfe.node2total(i, 1, 3) for i in top], None]
top_fixation_z = [amfe.node2total(top[0], 2, 3), [amfe.node2total(i, 2, 3) for i in top], None]

dirichlet_boundary_list = [bottom_fixation_x, bottom_fixation_y, bottom_fixation_z,
                           top_fixation_x, top_fixation_y, top_fixation_z]
my_mechanical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)


# static force in y-direction
my_neumann_boundary_list = [[[amfe.node2total(top[0], 0, 3),], 'static', (1E12, ), None]]
my_mechanical_system.apply_neumann_boundaries(my_neumann_boundary_list)


##%%
## Investigation on the solution stuff...
#
K = my_mechanical_system.K_global()


#%%
my_tetra_element = amfe.Tet4(poisson_ratio=0.3, E_modul=210E9, density=1E4)
my_mechanical_system.element_class_dict['Tet4'] = my_tetra_element

# static solution
amfe.solve_nonlinear_displacement(my_mechanical_system, 10, smplfd_nwtn_itr=1)
#amfe.solve_linear_displacement(my_mechanical_system)
export_path = '../results/bar_tet10' + time.strftime("_%Y%m%d_%H%M%S") + '/bar_tet10'
my_mechanical_system.export_paraview(export_path)
