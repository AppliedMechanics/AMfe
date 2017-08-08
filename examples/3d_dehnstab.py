# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Running a 3D-tension bar
"""


import amfe


mesh_file = amfe.amfe_dir('meshes/test_meshes/bar_Tet4_fine.msh')
output_file = amfe.amfe_dir('results/bar_tet10/bar_tet4')

# Building the mechanical system
my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(mesh_file, 29, my_material)
# Fixations are simple to realize
my_system.apply_dirichlet_boundaries(30, 'xyz')
my_system.apply_dirichlet_boundaries(31, 'yz')

# make master-slave approach to add constraint that x-dofs are all the same at right end
nodes, dofs = my_system.mesh_class.set_dirichlet_bc(31, 'x', output='external')
my_system.dirichlet_class.apply_master_slave_list([[dofs[0], dofs[1:], None],])
my_system.dirichlet_class.update()

# %%

# static force in x-direction
# my_neumann_boundary_list = [[[dofs[0],], 'static', (1E10, ), None]]
my_system.apply_neumann_boundaries(31, 1E12, (1, 0, 0), lambda t: t)


# static solution
amfe.solve_nonlinear_displacement(my_system)
# amfe.solve_linear_displacement(my_system)

my_system.export_paraview(output_file)
