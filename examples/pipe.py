#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Static Pipe example
"""

import numpy as np

from amfe.io import amfe_dir
from amfe.ui import *
from amfe.material import KirchhoffMaterial
from amfe.solver import AmfeSolution, SolverFactory
from amfe.solver.translators import create_constrained_mechanical_system_from_component

input_file = amfe_dir('meshes/gmsh/pipe.msh')
output_file = amfe_dir('results/pipe/pipe')

# PE-LD
my_material = KirchhoffMaterial(E=200E6, nu=0.3, rho=1E3, plane_stress=True)

mesh = import_mesh_from_file(input_file)

my_component = create_structural_component(mesh)

assign_material_by_group(my_component, my_material, 84)

set_dirichlet_by_group(my_component, 83, ('ux', 'uy', 'uz'))

set_neumann_by_group(my_component, 85, np.array([0.0, 1.0, 0.0]), F=lambda t: 1E6*t)


#%%
system, formulation = create_constrained_mechanical_system_from_component(my_component, constant_mass=True, constant_damping=True,
                                                             constraint_formulation='boolean')

solfac = SolverFactory()

solfac.set_system(system)
solfac.set_analysis_type('static')
solfac.set_dt_initial(0.05)
solfac.set_analysis_type('zero')
solfac.set_linear_solver('pardiso')
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

write_results_to_paraview(solution, my_component, output_file)


