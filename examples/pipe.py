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
my_system.apply_neumann_boundaries(85, 1E7, (0,1,0), lambda t:t)


#%%
#amfe.solve_linear_displacement(my_system)
amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=100,
                                  track_niter=True)

my_system.export_paraview(output_file + '_10')



#%%
dt = 0.01
T = np.arange(0, 1, dt)
ndof = my_system.K().shape[0]
q0 = np.zeros(ndof)
dq0 = np.zeros(ndof)

amfe.integrate_nonlinear_system(my_system, q0, dq0, T, dt, alpha=0.1)

