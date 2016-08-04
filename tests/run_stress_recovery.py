"""
Test run for the stress recovery implementation. 
"""

import numpy as np
import scipy as sp
import amfe


#%% Setting up the mesh

# test gmsh input-output functionality
# gmsh_input_file = amfe.amfe_dir() + '/meshes/test_meshes/bar_3d.msh'
gmsh_input_file = amfe.amfe_dir() + '/meshes/gmsh/bar_tri6.msh'

paraview_output_file = amfe.amfe_dir() + '/results/stress_recovery/stress_recovery'

my_mesh = amfe.Mesh()
my_mesh.import_msh(gmsh_input_file)


#%% Setting up the system

my_material = amfe.material.KirchhoffMaterial()
my_mesh.load_group_to_mesh(7, my_material)
my_assembly = amfe.Assembly(my_mesh)
my_mesh.set_dirichlet_bc(8, 'xy')
my_mesh.set_neumann_bc(9, 1E7, (0,1))
my_assembly.preallocate_csr()

#%% run with mechanical system
my_system = amfe.MechanicalSystem(stress_recovery=True)
my_system.load_mesh_from_gmsh(gmsh_input_file, 7, my_material)
my_system.apply_dirichlet_boundaries(8, 'xy')
my_system.apply_neumann_boundaries(9, 1E7, (0,1), time_func=lambda t: t)

my_system.K()
amfe.solve_nonlinear_displacement(my_system)
my_system.export_paraview(paraview_output_file)
u0 = my_system.u_output[-1]

#%% testing, if the system can assemble the stresses

#u0 = np.zeros(my_mesh.no_of_dofs)
K, f, S, E = my_assembly.assemble_k_f_S_E(u0, t=0)


#%% Exporting the stresses

field_list = [(S.reshape((-1,1)), {'ParaView':True,
                                   'Name':'stress',
                                   'AttributeType':'Tensor6',
                                   'Center':'Node',
                                   'NoOfComponents':6})]
my_mesh.save_mesh_xdmf(paraview_output_file, field_list)


