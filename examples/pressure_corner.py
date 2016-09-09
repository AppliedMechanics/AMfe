"""
Example showing a corner with pressure
"""

import amfe

input_file = amfe.amfe_dir('meshes/gmsh/pressure_corner.msh')
output_file = amfe.amfe_dir('results/pressure_corner/pressure_corner')


my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_system = amfe.MechanicalSystem(stress_recovery=True)
my_system.load_mesh_from_gmsh(input_file, 11, my_material)
my_system.apply_dirichlet_boundaries(9, 'x')
my_system.apply_dirichlet_boundaries(10, 'y')
my_system.apply_neumann_boundaries(12, 1E10, 'normal', lambda t: t)


#amfe.solve_linear_displacement(my_system)
amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=50, 
                                  track_niter=True)

my_system.export_paraview(output_file)



#%% Modal analysis

# omega, V = amfe.vibration_modes(my_system, save=True)

