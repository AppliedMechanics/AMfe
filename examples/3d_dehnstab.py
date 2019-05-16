# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Running a 3D-tension bar
"""
#%%
import numpy as np

from amfe.ui import *
from amfe.io import amfe_dir
from amfe.material import KirchhoffMaterial
from amfe.solver import *

mesh_file = amfe_dir('meshes/test_meshes/bar_Tet4_fine.msh')
output_file = amfe_dir('results/bar_tet10/bar_tet4_pardiso')

# Building the structural component
mesh = import_mesh_from_file(mesh_file)
component = create_structural_component(mesh)
my_material = KirchhoffMaterial()
component.assign_material(my_material, [29], 'S')

# Fixations are simple to realize
set_dirichlet_by_group(component, [30], ('ux', 'uy', 'uz'))
set_dirichlet_by_group(component, [31], ('uy', 'uz'))

# Special boundary condition: let all x coordinates have equal displacement
constraint = component.constraints.create_equal_displacement_constraint()
nodeids = mesh.get_nodeids_by_groups([31])
dofs = component.mapping.get_dofs_by_nodeids(nodeids, ('ux',))
dofs = dofs.reshape(-1)
masterdof = dofs[0]
slavedofs = dofs[1:]

for slavedof in slavedofs:
    component.assign_constraint('Equal Displacement Right', constraint, np.array([masterdof, slavedof], dtype=int),
                                np.array([], dtype=int))

# %%

# static force in x-direction
# my_neumann_boundary_list = [[[dofs[0],], 'static', (1E10, ), None]]
neumann_obj = component.neumann.create_fixed_direction_neumann(np.array([1.0, 0.0, 0.0], dtype=float), lambda t: 1E12*t)
component.assign_neumann('Right Force', neumann_obj, [31])

system, formulation = create_constrained_mechanical_system_from_component(component, constant_mass=True, constant_damping=True,
                                                             constraint_formulation='lagrange')

solfac = SolverFactory()

solfac.set_system(system)
solfac.set_analysis_type('static')
solfac.set_large_deflection(True)
solfac.set_dt_initial(0.05)
solfac.set_analysis_type('zero')
solfac.set_linear_solver('pardiso')
solfac.set_linear_solver_option('scaling', 1)
solfac.set_linear_solver_option('maximum_weighted_matching', 1)
solfac.set_newton_verbose(True)
solfac.set_nonlinear_solver('newton')
solfac.set_newton_atol(1e-5)
solfac.set_newton_rtol(1e-7)
solver = solfac.create_solver()

solution = AmfeSolution()


def write_callback(t, x, dx, ddx):
    u, du, ddu = formulation.recover(x, dx, ddx, t)
    solution.write_timestep(t, u, du, ddu)


x0 = np.zeros(system.dimension)
dx0 = x0.copy()
t0 = 0.0
t_end = 1.0

solver.solve(write_callback, t0, x0, dx0, t_end)

write_results_to_paraview(solution, component, output_file)

