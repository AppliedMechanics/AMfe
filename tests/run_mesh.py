# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:44:13 2015

Test the mesh-module

@author: johannesr
"""

import time


import numpy as np
import scipy as sp

import amfe


# test gmsh input-output functionality
gmsh_input_file = amfe.amfe_dir('meshes/test_meshes/bar_3d.msh')

#gmsh_input_file = 'meshes/test_meshes/bar_Tet4_finest_phys_group.msh'
paraview_output_file = amfe.amfe_dir('results/gmsh_test/gmsh_import')

my_mesh = amfe.Mesh()
my_mesh.import_msh(gmsh_input_file)


#%%
my_mesh.mesh_information('Sepp')

my_material = amfe.material.KirchhoffMaterial()
my_mesh.load_group_to_mesh(29, my_material)
my_assembly = amfe.Assembly(my_mesh)
my_mesh.set_dirichlet_bc(30, 'xyz')
my_mesh.set_neumann_bc(31, -1E10, 'normal')
my_assembly.preallocate_csr()

#%%

#%%
t1 = time.clock()
u0 = np.zeros(my_mesh.no_of_dofs)
K_unconstr, f_unconstr = my_assembly.assemble_k_and_f(u=u0, t=1)
K_nm_unconstr, f_nm_unconstr = my_assembly.assemble_k_and_f_neumann(u0, t=1)
t2 = time.clock()
print('Time for assembly:', t2-t1, 's.')
#%%

# my_boundary = amfe.DirichletBoundary(my_mesh.no_of_dofs, [[None, my_mesh.dofs_dirichlet, None],])
my_boundary = amfe.DirichletBoundary(my_mesh.no_of_dofs)
my_boundary.constrain_dofs(my_mesh.dofs_dirichlet)
B = my_boundary.b_matrix()
K = B.T @ K_unconstr @ B
f = B.T @ f_unconstr
f_ext = B.T @ f_nm_unconstr
u = sp.sparse.linalg.spsolve(K, f_ext)
#%%
# THIS SECTION IS DEPRECATED
#my_mesh.set_displacement(B.dot(u))
# my_mesh.save_mesh_for_paraview(paraview_output_file)
# my_mesh.save_mesh_xdmf(paraview_output_file)
#%%
#
## test mesh generator and mesh functionality
#node_file = '../meshes/selbstgebaut/curved_mesh_nodes.csv'
#element_file = '../meshes/selbstgebaut/curved_mesh_elements.csv'
#my_mesh_creator = amfe.MeshGenerator(1, 5, 10, 50, 0.3, x_curve=True)
#my_mesh_creator.build_mesh()
#my_mesh_creator.save_mesh(node_file, element_file)
#
#my_mesh.read_nodes_from_csv(node_file, node_dof=3)
#my_mesh.read_elements_from_csv(element_file)
#
#my_mesh.save_mesh_for_paraview('../results/selbstgebaut/selbstgebaut')
#
#
#print('List boundary nodes sorted by the boundary number. \
# \nTake care: The lines start indexing with 0, gmsh does this with 1.\n')
#for i, line in enumerate(my_mesh.boundary_list):
#    print('Boundary', i, '(gmsh-Key:', my_mesh.amfe2gmsh_boundary_dict[i], ')')
#    print(line, '\n')
