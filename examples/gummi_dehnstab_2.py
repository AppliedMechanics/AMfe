# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
This file is heavily deprecated.
"""

import time

import numpy as np
import scipy as sp

import amfe

warning = '''
###############################################################################
#############    This file is heavily deprecated. Don't use it.   #############
###############################################################################
'''
print(warning)

# Mesh generation
input_file = amfe.amfe_dir('meshes/gmsh/2D_Rectangle_tri6_dehnstab.msh')
# Building the mechanical system
my_materal = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4)
my_mechanical_system = amfe.MechanicalSystem()
my_mechanical_system.load_mesh_from_gmsh(input_file, phys_group=0,
                                         material=my_materal)




#%%


# Boundary handling
bottom_line_indices = my_mechanical_system.mesh_class.boundary_list[6]
top_line_indices = my_mechanical_system.mesh_class.boundary_list[4]

bottom_fixation_x = [None, [amfe.node2total(i, 0) for i in bottom_line_indices], None]
bottom_fixation_y = [None, [amfe.node2total(i, 1) for i in bottom_line_indices], None]
top_fixation_x = [amfe.node2total(top_line_indices[0], 0), [amfe.node2total(i, 0) for i in top_line_indices], None]
top_fixation_y = [amfe.node2total(top_line_indices[0], 1), [amfe.node2total(i, 1) for i in top_line_indices], None]

dirichlet_boundary_list = [bottom_fixation_x, bottom_fixation_y, top_fixation_x, top_fixation_y]
my_mechanical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)


# static force in y-direction
my_neumann_boundary_list = [[[amfe.node2total(top_line_indices[0], 1),], 'static', (2E11, ), None]]
my_mechanical_system.apply_neumann_boundaries(my_neumann_boundary_list)


# static solution
t1 = time.time()
amfe.solve_nonlinear_displacement(my_mechanical_system, 40, smplfd_nwtn_itr=1)
t2 = time.time()
print('Time for solving the static problem:', t2-t1)
export_path = '../results/gummi_mit_loch' + time.strftime("_%Y%m%d_%H%M%S") + '/gummi_mit_loch'
my_mechanical_system.export_paraview(export_path)
