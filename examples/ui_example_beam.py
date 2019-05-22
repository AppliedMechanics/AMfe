import numpy as np
import amfe.ui as ui
from amfe.io.tools import amfe_dir
from amfe.forces import constant_force, triangular_force

###################################################
###     Simple linear-elastic cantilever beam   ###
###################################################

mesh_file = amfe_dir('meshes/gmsh/simple_beam_gmsh/simple_beam_gmsh.msh')
mesh = ui.import_mesh_from_file(mesh_file)

model = ui.create_structural_component(mesh)

material = ui.create_material('Kirchhoff', E=210E9, nu=0.3, rho=1E4, plane_stress=True)

ui.assign_material_by_group(model, material, 'material')

ui.set_dirichlet_by_group(model, 'dirichlet', ('ux'), 'Dirichlet_x')
ui.set_dirichlet_by_nodeids(model, [1], ('uy'), 'Dirichlet_y')

F = constant_force(5E7)
ui.set_neumann_by_group(model, 'neumann', np.array([0.0, -1.0]), 'Load', F)

solution_writer = ui.solve_linear_static(model)

ui.write_results_to_paraview(solution_writer, model, amfe_dir('results/gmsh/ui_example_beam_linear'))


###################################################
###  Nonlinear heterogeneous cantilever beam    ###
###################################################

mesh_file = amfe_dir('meshes/gmsh/compositeBeam_50x10.msh')
mesh = ui.import_mesh_from_file(mesh_file)

model = ui.create_structural_component(mesh)

aluminium = ui.create_material('Kirchhoff', E=70E9, nu=0.3, rho=1E4, plane_stress=True)
rubber = ui.create_material('MooneyRivlin', A10=0.4E3, A01=0.1E3, kappa=1E5, rho=1E4, plane_stress=False)

ui.assign_material_by_group(model, aluminium, 'surface_material_steel')
ui.assign_material_by_group(model, rubber, 'surface_material_elastomer')

ui.set_dirichlet_by_group(model, 'x_dirichlet_line', ('ux'), 'Dirichlet_x')
ui.set_dirichlet_by_group(model, 'xy_dirichlet_point', ('uy'), 'Dirichlet_y')

F = constant_force(5E3)
ui.set_neumann_by_group(model, 'z_neumann', np.array([0.0, -1.0]), 'Load', F)

solution_writer = ui.solve_nonlinear_static(model)

ui.write_results_to_paraview(solution_writer, model, amfe_dir('results/gmsh/ui_example_beam_nonlinear'))


###########################################################
###  Dynamic nonlinear heterogeneous cantilever beam    ###
###########################################################

F = triangular_force(0, 0.01, 0.02, 5E6)
ui.set_neumann_by_group(model, 'z_neumann', np.array([0.0, -1.0]), 'Load', F)

solution_writer = ui.solve_nonlinear_dynamic(model, 0.0, 0.2)

ui.write_results_to_paraview(solution_writer, model, amfe_dir('results/gmsh/ui_example_beam_nonlinear_dynamic'))