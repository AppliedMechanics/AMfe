# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Rubber boot example
"""

import amfe



input_file = amfe.amfe_dir('meshes/gmsh/rubber_boot.msh')
output_file = amfe.amfe_dir('results/rubber_boot/boot')

# PE-LD; better material would be Mooney-Rivlin
my_material = amfe.KirchhoffMaterial(E=200E6, nu=0.3, rho=1E3)

my_system = amfe.MechanicalSystem(stress_recovery=False)
my_system.load_mesh_from_gmsh(input_file, 977, my_material, scale_factor=1E-3)
my_system.apply_dirichlet_boundaries(978, 'xyz')
my_system.apply_neumann_boundaries(979, 1E7, (0,1,0), lambda t:t)

#%%

amfe.vibration_modes(my_system, save=True)

my_system.export_paraview(output_file + '_modes')

#%%