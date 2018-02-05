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
input_file = amfe.amfe_dir('meshes/gmsh/Beam/Beam10x1Quad8.msh')
output_file = amfe.amfe_dir('results/Beam/Beam10x1Quad8')


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
    'linear_solver': amfe.linalg.PardisoSolver,
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
static = False
reduce = False
if not linear:
    filename = '_nonlinear_'
else:
    filename = '_linear_'
if not static:
    filename += 'dynamics_'
else:
    filename += 'statics_'


# solve system
if not static:
    if not linear:# non-linear dynamics
        pass
    else:# linear dynamics
        pass
else:
    if not linear:# non-linear statics
        pass
    else:# linear statics
        pass


# write output
# end = len(my_system.u_output)
# my_file = open(output_file + filename + 'mechForm.dat', 'w')
# my_time = []
# my_displX = []
# my_displY = []
# for i in range(end):
#     my_time.append(my_system.T_output[i])
#     my_displX.append(my_system.u_output[i][2])
#     my_displY.append(my_system.u_output[i][3])
#     my_file.write(str(my_time[i]) + ' ' + str(my_displX[i]) + ' ' + str(my_displY[i]) + '\n')
# my_file.close()

#my_system.export_paraview(output_file + filename + 'mechForm')

