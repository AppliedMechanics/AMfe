# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
An example describing the mesh tying capability of amfe.
"""
import amfe
#%%

input_file = amfe.amfe_dir('meshes/gmsh/plate_mesh_tying.msh')
output_file = amfe.amfe_dir('results/mesh_tying/plate_mesh_tying')

my_system = amfe.MechanicalSystem()
my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)

my_system.load_mesh_from_gmsh(input_file, 1, my_material)
my_system.mesh_class.load_group_to_mesh(2, my_material)
my_system.assembly_class.preallocate_csr()

my_system.tie_mesh(5, 6, tying_type='slide', verbose=True,
                   conform_slave_mesh=True, fix_mesh_dist=1E-1)

my_system.apply_dirichlet_boundaries(3, 'xyz')
my_system.apply_neumann_boundaries(4, 1E10, (1, 1, 1), lambda t: t)

#%%
solver = amfe.LinearStaticsSolver(my_system)
solver.solve()

my_system.export_paraview(output_file + '_full_linear')

#%% Modal analysis
amfe.vibration_modes(my_system, save=True)
my_system.export_paraview(output_file + '_modes')
