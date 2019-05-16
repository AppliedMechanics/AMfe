# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Cantilever beam loaded at tip solved with adaptive time stepping.
"""


# load packages
import amfe
import numpy as np


# define in- and output files
input_file = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad8.msh')
output_file = amfe.amfe_dir('results/beam/Beam10x1Quad8_nonlinear_dynamics_adaptive_time_stepping')


# define system
material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(5, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1.)
system.apply_rayleigh_damping(1e0, 1e-6)


# vibration modes
amfe.vibration_modes(mechanical_system=system, n=20, save=True)
system.export_paraview(output_file + '_vibration_modes')
system.clear_timesteps()


# define simulation parameters
options = {
    't0': 0.,
    't_end': 10.,
    'dt': 1.e-4,
    'output_frequency': 1,
    'rho_inf': .95,
    'initial_conditions': {
        'q0': np.zeros(ndof),
        'dq0': np.zeros(ndof)},
    'relative_tolerance': 1.e-6,
    'absolute_tolerance': 1.e-9,
    'verbose': True,
    'max_number_of_iterations': 7,
    'convergence_abort': True,
    'write_iterations': False,
    'track_iterations': True,
    'save_solution': True}


solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
# solver.solve()
solver.solve_with_adaptive_time_stepping(rel_dt_err_tol=1.e-3)


# write output
system.export_paraview(output_file)

end = len(system.T_output)
file = open(output_file + '.dat', 'w')
for i in range(end):
    # file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][2]) + ' ' + str(system.u_output[i][3]) + '\n')
    file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][2]) + ' ' + str(system.u_output[i][3]) \
               + ' ' + str(solver.dt_info[i]) + '\n')
file.close()
