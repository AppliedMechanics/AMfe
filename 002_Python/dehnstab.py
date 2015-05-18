# -*- coding: utf-8 -*-
"""
Beispiel Dehnstab
"""

# Standard-Module
import numpy as np
import scipy as sp
import os
import time

# Eigene Module
import element
import mesh
import assembly
import boundary

# Output-Ordner
output_dir = os.path.splitext(os.path.basename(__file__))[0] + time.strftime("_%Y%m%d_%H%M%S")
t1 = time.clock()

# Netzgenerator
knotenfile = output_dir + '/Vernetzung/nodes.csv'
elementfile = output_dir + '/Vernetzung/elements.csv'
my_meshgenerator = mesh.MeshGenerator(x_len=3, y_len=3*3*3, x_no_elements=3*3, y_no_elements=3*3*3*3)
my_meshgenerator.build_mesh()
my_meshgenerator.save_mesh(knotenfile, elementfile)

# Netz
my_mesh = mesh.Mesh(node_dof=2)  # 2 Freiheitsgrade pro Knoten
my_mesh.read_nodes_from_csv(knotenfile)
my_mesh.read_elements_from_csv(elementfile)


t2 = time.clock()
print('Netz mit', my_mesh.no_of_dofs, 'Freiheitsgraden und', my_mesh.no_of_elements, 'Elementen erstellt')

# Element
my_element = element.ElementPlanar(poisson_ratio=0.3)

multiproc = False
if multiproc:
    no_of_proc= 6
    element_object_list = [element.ElementPlanar() for i in range(no_of_proc)]
    element_function_list_k = [j.k_int for j in element_object_list]
    element_function_list_m = [j.m_int for j in element_object_list]
    my_multiprocessing = Assembly.MultiprocessAssembly(Assembly.PrimitiveAssembly, element_function_list_k, nodes_array, element_array)
    K_coo = my_multiprocessing.assemble()
else:
    my_assembler = assembly.PrimitiveAssembly(my_mesh.nodes, my_mesh.elements, my_element.k_int)
    K_coo = my_assembler.assemble_matrix()
K = K_coo.tocsr()
print('Matrix K assembliert')

t3 = time.clock()
# update der Matrix-Funktion des Elements
if multiproc:
    my_multiprocessing = assembly.MultiprocessAssembly(assembly.PrimitiveAssembly, element_function_list_m, nodes_array, element_array)
    M_coo = my_multiprocessing.assemble()
else:
    my_assembler.matrix_function = my_element.m_int
    M_coo = my_assembler.assemble_matrix()
M = M_coo.tocsr()
print('Matrix M assembliert')

t4 = time.clock()
print('Zeit zum Generieren des Netzes:', t2 - t1)
print('Zeit zum Assemblieren der Steifigkeitsmatrix:', t3 - t2)
print('Zeit zum Assemblieren der Massenmatrix:', t4 - t3)

t5 = time.clock()
# Randbedingungen
bottom_fixation = [None, range(20), None]
#bottom_fixation = [None, [1 + 2*x for x in range(10)], None]
#bottom_fixation2 = [None, [0, ], None]
conv = assembly.ConvertIndices(2)
master_node = conv.node2total(810, 1)
top_fixation = [master_node, [master_node + 2*x for x in range(10)], None]
dirichlet_boundary_list = [bottom_fixation, top_fixation]
my_dirichlet_bcs = boundary.DirichletBoundary(M.shape[0], dirichlet_boundary_list)
B = my_dirichlet_bcs.b_matrix()

t6 = time.clock()
print('Zeit zum Aufbringen der Randbedingungen:', t6-t5)

# Statische Analyse:
K_bound = B.T.dot(K.dot(B))

F = np.zeros(my_mesh.no_of_dofs)
F[master_node] = 1E10
F_bound = B.T.dot(F)

u_bound = sp.linalg.solve(K_bound.toarray(), F_bound)
u_full = B.dot(u_bound)
my_mesh.set_displacement(u_full)
my_mesh.save_mesh_for_paraview(output_dir + '/Paraview/Dehnstab')
print("Verschiebung oberer Rand:",   u_full[my_mesh.no_of_dofs -1])



