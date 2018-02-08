# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Example showing a cantilever beam which is loaded on the tip with a force
showing nonlinear displacements.
"""


import amfe


input_file = amfe.amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe.amfe_dir('results/beam_nonlinear/beam')


my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(input_file, 7, my_material)
my_system.apply_dirichlet_boundaries(8, 'xy') # fixature of the left side
my_system.apply_neumann_boundaries(key=9, val=1E8, direct=(0,-1),
                                   time_func=lambda t: t)


solverlin = amfe.LinearStaticsSolver(my_system)
solverlin.solve()
my_system.export_paraview(output_file + '_linear')

my_system.clear_timesteps()
solvernl = amfe.NonlinearStaticsSolver(my_system, number_of_load_steps=50)
solvernl.solve()
my_system.export_paraview(output_file + '_nonlinear')



#%% Modal analysis

omega, V = amfe.vibration_modes(my_system, save=True)
my_system.export_paraview(output_file + '_modes')
