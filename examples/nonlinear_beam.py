"""
Example showing a cantilever beam which is loaded on the tip with a force 
showing nonlinear displacements. 
"""


import amfe


input_file = '../meshes/gmsh/bar.msh'
output_file = '../results/beam_nonlinear/beam_nonlinear'


my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(input_file, 7, my_material)
my_system.apply_dirichlet_boundaries(8, 'xy') # fixature of the left side
my_system.apply_neumann_boundaries(key=9, val=1E8, direct=(0,-1), 
                                   time_func=lambda t: t)


amfe.solve_linear_displacement(my_system)
#amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=50)

my_system.export_paraview(output_file + '_linear')
#my_system.export_paraview(output_file)



#%% Modal analysis

omega, V = amfe.vibration_modes(my_system, save=True)
my_system.export_paraview(output_file + '_modes')