# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Created on Fri Dec 18 15:31:45 2015

@author: johannesr
"""


import numpy as np
from amfe.io import amfe_dir
from amfe.ui import *
from amfe.material import KirchhoffMaterial
from amfe.component import StructuralComponent
from amfe.solver import SolverFactory, AmfeSolution
from amfe.solver.translators import create_constrained_mechanical_system_from_component
from amfe.structural_dynamics import vibration_modes

meshfile = amfe_dir('meshes/gmsh/AMFE_logo.msh')
output_file = amfe_dir('results/AMFE_logo/logo_5_v2.xdmf')


material_1 = KirchhoffMaterial(E=5E6, rho=1E4)
material_2 = KirchhoffMaterial(E=5E7, rho=1E4)

mesh = import_mesh_from_file(meshfile)

my_component = StructuralComponent(mesh)
my_component.assign_material(material_1, [299, 300], 'S')
my_component.assign_material(material_2, [301, 302], 'S')

set_dirichlet_by_group(my_component, 298, ('ux', 'uy', 'uz'))

no_of_dofs = my_component.mapping.no_of_dofs
q0 = np.zeros(no_of_dofs)
nodeids = my_component.mesh.nodes_df.index.to_list()
dofnumbers = my_component.mapping.get_dofs_by_nodeids(nodeids, ('uy', ))
M_unconstr = my_component.M(q0, q0, 0.0)

g = np.zeros_like(q0)
for dofnumber in dofnumbers:
    g[dofnumber] = -9.81


f_gravity = M_unconstr @ g


def my_fext(u, du, t):
    return f_gravity*t


my_component.f_ext = my_fext

my_system, my_formulation = create_constrained_mechanical_system_from_component(my_component, True, True)

x0 = np.zeros(my_system.dimension)

om, V = vibration_modes(my_system.K(x0, x0, 0.0), my_system.M(x0, x0, 0.0), 10)

print('The vibration modes eigenfrequencies (Hz) are:\n', om/(2*np.pi))

solfac = SolverFactory()

solfac.set_system(my_system)
solfac.set_analysis_type('static')
solfac.set_dt_initial(0.01)
solfac.set_analysis_type('zero')
solfac.set_linear_solver('pardiso')
solfac.set_newton_verbose(True)
solfac.set_nonlinear_solver('newton')
solfac.set_newton_atol(1e-5)
solfac.set_newton_rtol(1e-7)
solver = solfac.create_solver()

solution = AmfeSolution()


def write_callback(t, x, dx, ddx):
    u, du, ddu = my_formulation.recover(x, dx, ddx, t)
    solution.write_timestep(t, u, du, ddu)


dx0 = x0.copy()
t0 = 0.0
t_end = 1.0

solver.solve(write_callback, t0, x0, dx0, t_end)

write_results_to_paraview(solution, my_component, output_file)
