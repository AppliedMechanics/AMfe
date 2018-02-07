# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Dynamic Pipe test case
"""

import numpy as np
import amfe


input_file = amfe.amfe_dir('meshes/gmsh/pipe.msh')
output_file = amfe.amfe_dir('results/pipe/pipe')

# Steel
#my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
# PE-LD
my_material = amfe.KirchhoffMaterial(E=200E6, nu=0.3, rho=1E3, plane_stress=True)

my_system = amfe.MechanicalSystem(stress_recovery=True)
my_system.load_mesh_from_gmsh(input_file, 84, my_material)
my_system.apply_dirichlet_boundaries(83, 'xyz')
my_system.apply_neumann_boundaries(85, 1E6, (0,1,0), lambda t:t)


#%%
#amfe.solve_linear_displacement(my_system)
solvernl = amfe.NonlinearStaticsSolver(my_system, number_of_load_steps=50, track_iterations=True, verbose=True)
solvernl.solve()
my_system.export_paraview(output_file + '_10')


