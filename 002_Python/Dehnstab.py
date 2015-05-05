# -*- coding: utf-8 -*-



# Eigene Module
import Element
import Mesh
import Assembly
import ImportMesh

# Standard-Module
import numpy as np
import scipy as sp



# Zeitmessung:
import time


t1 = time.clock()

# Netz
my_meshgenerator = Mesh.Mesh_generator(x_len=3, y_len=3*3*3, x_no_elements=3*3, y_no_elements=3*3*3*3)
my_meshgenerator.build_mesh()
ndof = len(my_meshgenerator.nodes)*2
nelements = len(my_meshgenerator.elements)
nodes_array = np.array(my_meshgenerator.nodes)[:,1:]
element_array = np.array(my_meshgenerator.elements)[:,1:]
t2 = time.clock()
print('Netz mit', ndof, 'Freiheitsgraden und', nelements, 'Elementen erstellt')

# Element
my_element = Element.ElementPlanar(poisson_ratio=0.3)
multiproc = False
if multiproc:
    no_of_proc= 6
    element_object_list = [Element.ElementPlanar() for i in range(no_of_proc)]
    element_function_list_k = [j.k_int for j in element_object_list]
    element_function_list_m = [j.m_int for j in element_object_list]
    my_multiprocessing = Assembly.MultiprocessAssembly(Assembly.PrimitiveAssembly, element_function_list_k, nodes_array, element_array)
    K_coo = my_multiprocessing.assemble()
else:
    my_assembly = Assembly.PrimitiveAssembly(np.array(my_meshgenerator.nodes)[:,1:], np.array(my_meshgenerator.elements)[:,1:], my_element.k_int)
    K_coo = my_assembly.assemble()
K = K_coo.tocsr()
print('Matrix K assembliert')

t3 = time.clock()
# update der Matrix-Funktion
if multiproc:
    my_multiprocessing = Assembly.MultiprocessAssembly(Assembly.PrimitiveAssembly, element_function_list_m, nodes_array, element_array)
    M_coo = my_multiprocessing.assemble()
else:
    my_assembly.matrix_function = my_element.m_int
    M_coo = my_assembly.assemble()
M = M_coo.tocsr()
print('Matrix M assembliert')

t4 = time.clock()
print('Das System hat', ndof, 'Freiheitsgrade und', nelements, 'Elemente.')
print('Zeit zum generieren des Netzes:', t2 - t1)
print('Zeit zum Assemblieren der Steifigkeitsmatrix:', t3 - t2)
print('Zeit zum Assemblieren der Massenmatrix:', t4 - t3)


# Graphische Analyse des Netzes
knotenfile = 'Vernetzungen/nodes.csv'
elementfile = 'Vernetzungen/elements.csv'
my_meshgenerator.save_mesh(knotenfile, elementfile)

my_mesh = Mesh.Mesh()
my_mesh.read_nodes(knotenfile)
my_mesh.read_elements(elementfile)
my_mesh.set_displacement(np.zeros(ndof))
my_mesh.save_mesh_for_paraview('Versuche/Dehnstab')


# Randbedingungen
bottom_fixation = [None, range(20), None]
#bottom_fixation = [None, [1 + 2*x for x in range(10)], None]
#bottom_fixation2 = [None, [0, ], None]
conv = Assembly.ConvertIndices(2)
master_node = conv.node2total(810, 1)
top_fixation = [master_node, [master_node + 2*x for x in range(10)], None]
dirichlet_boundary_list = [bottom_fixation, top_fixation]
my_dirichlet_bcs = Assembly.Boundary(M.shape[0], dirichlet_boundary_list)
B = my_dirichlet_bcs.b_matrix()


# Statische Analyse:
K_bound = B.T.dot(K.dot(B))

F = np.zeros(ndof)
F[master_node] = 1E10
F_bound = B.T.dot(F)

u_bound = sp.linalg.solve(K_bound.toarray(), F_bound)
u_full = B.dot(u_bound)
my_mesh.set_displacement(u_full)
my_mesh.save_mesh_for_paraview('Versuche/Dehnstab')



