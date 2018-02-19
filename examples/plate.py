# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
File running the hexahedron elements
"""

import amfe
import numpy as np


gmsh_input_file = amfe.amfe_dir('meshes/gmsh/plate_transfinite.msh')
paraview_output_file = amfe.amfe_dir('results/plate/plate_transfinite')

my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 30, my_material)
my_system.apply_dirichlet_boundaries(31, 'xyz')

#my_system.apply_neumann_boundaries(32, 1E8, np.array([0,0,1]),
#                                   time_func=lambda t: t)
# pulling the system
my_system.apply_neumann_boundaries(32, -1E10, 'normal',
                                   time_func=lambda t: t)

# Test the paraview basic output
# my_system.export_paraview(paraview_output_file)

#%%
nlsolver = amfe.NonlinearStaticsSolver(my_system, number_of_load_steps=20)
nlsolver.solve()

my_system.export_paraview(paraview_output_file)

#%%
omega, V = amfe.vibration_modes(my_system, save=True)
my_system.export_paraview(paraview_output_file)

#%%
Theta = amfe.static_correction_theta(V, my_system.K)
my_system.clear_timesteps()

no_of_modes = V.shape[1]
for i in range(no_of_modes):
    for j in range(i + 1):
        my_system.write_timestep(u=Theta[:,i,j], t=i*100 + j)

my_system.export_paraview(paraview_output_file + 'mds')
#%%
my_system.K_and_f()
