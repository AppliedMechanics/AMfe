# Copyright (c) 2017, Lehrstuhl fuer Regelungstechnik, Technische Universitaet Muenchen.
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


# define in- and output files
input_file = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad8.msh')
output_file = amfe.amfe_dir('results/beam/Beam10x1Quad8')


# define system
material = amfe.KirchhoffMaterial(E=2.1E11, nu=0.3, rho=7.867E3, plane_stress=False)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(5, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
system.apply_neumann_boundaries(key=3, val=1e8, direct=(0, -1), time_func=lambda t: 1)
system.apply_rayleigh_damping(1e0, 1e-5)


# define simulation parameters
options = {
    'linear_solver': amfe.linalg.PardisoSolver,  # amfe.linalg.ScipySparseSolver
    'number_of_load_steps': 10,
    'newton_damping': 1.0,
    'simplified_newton_iterations': 1,
    't0': 0.0,
    't_end': 0.4,
    'dt': 1e-3,
    'dt_output': 1e-3,
    'rho_inf': 0.9,
    'initial_conditions': {
        'q0': np.zeros(ndof),
        'dq0': np.zeros(ndof)},
    'relative_tolerance': 1.0E-6,
    'absolute_tolerance': 1.0E-9,
    'verbose': False,
    'max_number_of_iterations': 99,
    'convergence_abort': True,
    'write_iterations': False,
    'track_number_of_iterations': True,
    'save_solution': False}

linear = False
# linear = True
statics = False
# statics = False
scheme = 'GeneralizedAlpha'
# scheme = 'JWHAlpha'
if not linear:
    filename = '_nonlinear_'
else:
    filename = '_linear_'
if not statics:
    filename += 'dynamics_'
else:
    filename += 'statics_'


# solve system
if not statics:
    if not linear:  # non-linear dynamics
        if scheme is 'GeneralizedAlpha':
            solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'JWHAlpha':
            solver = amfe.JWHAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
        else:
            raise ValueError('Time integration scheme not supported!')
    else:  # linear dynamics
        if scheme is 'GeneralizedAlpha':
            solver = amfe.GeneralizedAlphaLinearDynamicsSolver(mechanical_system=system, **options)
        elif scheme is 'JWHAlpha':
            solver = amfe.JWHAlphaLinearDynamicsSolver(mechanical_system=system, **options)
        else:
            raise ValueError('Time integration scheme not supported!')
else:
    if not linear:  # non-linear statics
        solver = amfe.NonlinearStaticsSolver(mechanical_system=system, **options)
    else:  # linear statics
        solver = amfe.LinearStaticsSolver(mechanical_system=system, **options)

print(solver)
print(solver.linear_solver)
solver.solve()


# write output
system.export_paraview(output_file + filename)

end = len(system.u_output)
file = open(output_file + filename + '.dat', 'w')
for i in range(end):
    file.write(str(system.T_output[i]) + ' ' + str(system.u_output[i][2]) + ' ' + str(system.u_output[i][3]) + '\n')
file.close()

