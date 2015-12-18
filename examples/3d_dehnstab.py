# -*- coding: utf-8 -*-
"""
Running a 3D-tension bar
"""


import sys
import time

import numpy as np
import scipy as sp
import amfe


import matplotlib.pyplot as plt


# Building the mechanical system
my_material = amfe.KirchhoffMaterial()
my_mechanical_system = amfe.MechanicalSystem()
my_mechanical_system.load_mesh_from_gmsh('../meshes/test_meshes/bar_Tet4_fine.msh', 29, my_material)
# Fixations are simple to realize
my_mechanical_system.apply_dirichlet_boundaries(30, 'xyz')
my_mechanical_system.apply_dirichlet_boundaries(31, 'yz')

# make master-slave approach to fix x-direction
nodes, dofs = my_mechanical_system.mesh_class.select_dirichlet_bc(31, 'x', output='external')
my_mechanical_system.dirichlet_class.master_slave_list.append([dofs[0], dofs, None])
my_mechanical_system.dirichlet_class.update()

#%%

# static force in y-direction
# my_neumann_boundary_list = [[[dofs[0],], 'static', (1E10, ), None]]
my_mechanical_system.apply_neumann_boundaries(31, 1E12, 'x', lambda t: t)


# static solution
amfe.solve_nonlinear_displacement(my_mechanical_system, 10, smplfd_nwtn_itr=1)
#amfe.solve_linear_displacement(my_mechanical_system)
export_path = '../results/bar_tet10' + time.strftime("_%Y%m%d_%H%M%S") + '/bar_tet4'
my_mechanical_system.export_paraview(export_path)
