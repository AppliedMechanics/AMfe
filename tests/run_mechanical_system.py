# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:37:20 2015

@author: johannesr
"""



import numpy as np
import scipy as sp
import time

# make amfe running
import sys
sys.path.insert(0,'..')
import amfe

# test gmsh input-output functionality

gmsh_input_file = '../meshes/test_meshes/bar_3d.msh'

#gmsh_input_file = 'meshes/test_meshes/bar_Tet4_finest_phys_group.msh'
paraview_output_file = '../results/gmsh_test/gmsh_import'

#%%


my_mechanical_system = amfe.MechanicalSystem()
my_material = amfe.KirchhoffMaterial()
my_mechanical_system.load_mesh_from_gmsh(gmsh_input_file, 29, my_material)

my_mechanical_system.apply_dirichlet_boundaries(31, 'xyz')

#%%
print('Choose now the area which will be pushed or pulled:')
#my_mechanical_system.apply_dirichlet_boundaries(30, 'yz')
def time_func(t):
    return t
my_mechanical_system.apply_neumann_boundaries(30, -1E11, 'normal', time_func=time_func)

# The old way
#nodes_list, dofs_list = my_mechanical_system.mesh_class.select_dirichlet_bc(30, 'x', output='external')
#NB = [dofs_list, 'static', (8E9, ), None]
#my_mechanical_system.apply_neumann_boundaries([NB, ])

K, f = my_mechanical_system.K_and_f(np.zeros(my_mechanical_system.dirichlet_class.no_of_constrained_dofs), t=1)
M = my_mechanical_system.M()

amfe.solve_nonlinear_displacement(my_mechanical_system, 10, smplfd_nwtn_itr=1, wrt_iter=False, rtol=1E-7)

# amfe.solve_linear_displacement(my_mechanical_system)
my_mechanical_system.export_paraview(paraview_output_file)

