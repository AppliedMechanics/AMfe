# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Cantilever beam loaded at tip.
"""


# load packages
import amfe
from scipy import linalg
import numpy as np


###############################################################################
# set simulation type
###############################################################################
#  > static (True) or dynamic (False) analysis
statics = True
###############################################################################
#  > linear (True) or nonlinear (False) analysis
linear = False
###############################################################################
#  > time integration scheme ('GeneralizedAlpha', 'WBZAlpha', 'HHTAlpha',
#    'NewmarkBeta', 'JWHAlpha' or 'JWHAlphaStateSpace') for dynamic analysis
scheme = 'GeneralizedAlpha'
###############################################################################


# define in- and output files
input_file = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad8.msh')
output_file = amfe.amfe_dir('results/beam/Beam10x1Quad8')
if not linear:
    output_file += '_nonlinear'
else:
    output_file += '_linear'
if not statics:
    output_file += '_dynamics'
else:
    output_file += '_statics'
if scheme is 'GeneralizedAlpha':
    output_file += '_generalizedalpha'
elif scheme is 'WBZAlpha':
    output_file += '_wbzalpha'
elif scheme is 'HHTAlpha':
    output_file += '_hhtalpha'
elif scheme is 'NewmarkBeta':
    output_file += '_newmarkbeta'
elif scheme is 'JWHAlpha':
    output_file += '_jwhalpha'
elif scheme is 'JWHAlphaStateSpace':
    output_file += '_jwhalphastatespace'


# define system
material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(5, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
if not statics:  # dynamics
    system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1)
else:  # statics
    system.apply_neumann_boundaries(key=3, val=5e8, direct=(0, -1), time_func=lambda t: t)
# system.apply_rayleigh_damping(1e0, 1e-5)


# define simulation parameters
options = {
    'linear_solver': amfe.linalg.PardisoSolver,  # amfe.linalg.ScipySparseSolver
    'number_of_load_steps': 10,
    'newton_damping': 1.0,
    'simplified_newton_iterations': 1,
    't': 1.0,
    't0': 0.0,
    't_end': 0.4,
    'dt': 5e-4,
    'output_frequency': 1,
    'rho_inf': 0.95,
    'initial_conditions': {
        'x0': np.zeros(2*ndof),
        'q0': np.zeros(ndof),
        'dq0': np.zeros(ndof)},
    'relative_tolerance': 1.0E-6,
    'absolute_tolerance': 1.0E-9,
    'verbose': True,
    'max_number_of_iterations': 30,
    'convergence_abort': True,
    'write_iterations': False,
    'track_iterations': False,
    'save_solution': True}
rho_inf = 0.95
alpha = 0.0005


# represent mechanical system as state-space system
#  > solve for non-linear static displacement of system still in mechanical form
if not statics:  # dynamics
    system.apply_neumann_boundaries(key=3, val=-2.5e8, direct=(0, -1), time_func=lambda t: 1)  # deleted dynamic NBCs
    system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: t)  # set static NBCs
solver = amfe.NonlinearStaticsSolver(mechanical_system=system, **options)
solver.solve()
q_static = system.constrain_vec(system.u_output[-1][:])
system.clear_timesteps()
if not statics:  # dynamics
    system.apply_neumann_boundaries(key=3, val=-2.5e8, direct=(0, -1), time_func=lambda t: t)  # delete static NBCs
    system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1)  # reset dynamic NBCs
#  > convert system to state-space form
state_space_system = amfe.mechanical_system.convert_mechanical_system_to_state_space(
    system, regular_matrix=system.K(q_static), overwrite=False)


# solve system
if not statics:
    if not linear:  # non-linear dynamics
        if scheme is 'GeneralizedAlpha':
            solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'WBZAlpha':
            solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_wbz_alpha_parameters(rho_inf=rho_inf)
        elif scheme is 'HHTAlpha':
            solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_hht_alpha_parameters(rho_inf=rho_inf)
        elif scheme is 'NewmarkBeta':
            solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_newmark_beta_parameters(beta=0.25*(1 + alpha)**2, gamma=0.5 + alpha)
        elif scheme is 'JWHAlpha':
            solver = amfe.JWHAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'JWHAlphaStateSpace':
            system = state_space_system
            solver = amfe.JWHAlphaNonlinearDynamicsSolverStateSpace(mechanical_system=system, **options)
        else:
            raise ValueError('Time integration scheme not supported!')
    else:  # linear dynamics
        if scheme is 'GeneralizedAlpha':
            solver = amfe.GeneralizedAlphaLinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'WBZAlpha':
            solver = amfe.GeneralizedAlphaLinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_wbz_alpha_parameters(rho_inf=rho_inf)
        elif scheme is 'HHTAlpha':
            solver = amfe.GeneralizedAlphaLinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_hht_alpha_parameters(rho_inf=rho_inf)
        elif scheme is 'NewmarkBeta':
            solver = amfe.GeneralizedAlphaLinearDynamicsSolver(mechanical_system=system, **options)
            solver.set_newmark_beta_parameters(beta=0.25*(1 + alpha)**2, gamma=0.5 + alpha)
        elif scheme is 'JWHAlpha':
            solver = amfe.JWHAlphaLinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'JWHAlphaStateSpace':
            system = state_space_system
            solver = amfe.JWHAlphaLinearDynamicsSolverStateSpace(mechanical_system=system, **options)
        else:
            raise ValueError('Time integration scheme not supported!')
else:
    if not linear:  # non-linear statics
        solver = amfe.NonlinearStaticsSolver(mechanical_system=system, **options)
    else:  # linear statics
        solver = amfe.LinearStaticsSolver(mechanical_system=system, **options)

print('\n System solver = ')
print(solver)
print('\n Linear solver =')
print(solver.linear_solver)
print('\n Solving...')
solver.solve()

# alternative solver with adaptive time stepping (available only for nonlinear mechanical forms)
# solver.solve_with_adaptive_time_step(dt_start=1.0e-4, dt_min=1.0e-6, dt_max=1.0e-1, change_factor_min=0.5,
#                                      change_factor_max=2.0, savety_factor=0.9, trust_in_new_increased_dt = 0.01,
#                                      relative_dt_tolerance=1.0e-2, max_dt_iterations=10,
#                                      new_dt_for_failing_newton_convergence=0.8)

# alternative for solving system
# system.set_solver(solver=amfe.GeneralizedAlphaNonlinearDynamicsSolver, **options)
# system.solve()


# write output
system.export_paraview(output_file)

end = len(system.T_output)
file = open(output_file + '.dat', 'w')
for i in range(end):
    file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][2]) + ' ' + str(system.u_output[i][3]) + '\n')
    # file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][2]) + ' ' + str(system.u_output[i][3]) \
    #            + ' ' + str(solver.dt_info[i]) + '\n')  # for time step adapting solvers only
file.close()

