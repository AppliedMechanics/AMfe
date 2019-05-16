import numpy as np
import amfe.ui as ui
from amfe.io.tools import amfe_dir
from amfe.forces import constant_force

mesh_file = amfe_dir('meshes/gmsh/simple_beam_gmsh/simple_beam_gmsh.msh')
mesh = ui.import_mesh_from_file(mesh_file)

model = ui.create_structural_component(mesh)

material = ui.create_material('Kirchhoff', E=210E9, nu=0.3, rho=1E4, plane_stress=True)

ui.assign_material_by_group(model, material, 'material')

ui.set_dirichlet_by_group(model, ['dirichlet'], ('ux'), 'Dirichlet_x')
ui.set_dirichlet_by_nodeids(model, [1], ('uy'), 'Dirichlet_y')

F = constant_force(3.0)
ui.set_neumann_by_group(model, 'neumann', np.array([0.0, 1.0]), 'Load', F)

solution_writer = ui.solve_linear_static(model)

ui.write_results_to_paraview(solution_writer, model, amfe_dir('results/gmsh/simple_beam_gmsh'))
