# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Example showing a corner with pressure following the Tutorial 1
"""

# --- Preparation ---
from amfe.io import amfe_dir
import amfe.ui

input_file = amfe_dir('meshes/gmsh/pressure_corner.msh')
output_file_deformation = amfe_dir('results/pressure_corner/pressure_corner_nonlinear_deformation')
output_file_modes = amfe_dir('results/pressure_corner/pressure_corner_linear_modes')

# --- Load Mesh ---
my_mesh = amfe.ui.import_mesh_from_file(input_file)

# --- Setting up new component ---
my_component = amfe.ui.create_structural_component(my_mesh)

# --- Define materials and assign it to component ---
my_material = amfe.ui.create_material('Kirchhoff', E=210E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=0.1)
amfe.ui.assign_material_by_group(my_component, my_material, 11)

# --- Apply boundary conditions ---
amfe.ui.set_dirichlet_by_group(my_component, 9, 'ux', 'Dirichlet0')
amfe.ui.set_dirichlet_by_group(my_component, 10, 'uy', 'Dirichlet1')

amfe.ui.set_neumann_by_group(my_component, 12, 'normal', following=True, neumann_name='Neumann0', f=lambda t: 1E7*t)

# --- Translate the Component to a MechanicalSystem ---
my_system, my_formulation = amfe.ui.create_mechanical_system(my_component)

# --- Solve and write deformation analysis ---
my_solution = amfe.ui.solve_nonlinear_static(my_system, my_formulation, my_component, 50)
amfe.ui.write_results_to_paraview(my_solution, my_component, output_file_deformation)

# --- Solve and write modal analysis ---
modes = amfe.ui.solve_modes(my_system, my_formulation, no_of_modes=10, hertz=True)
amfe.ui.write_results_to_paraview(modes, my_component, output_file_modes)

print(modes.t)
