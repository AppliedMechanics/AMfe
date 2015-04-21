# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:13:52 2015

Löschen aller Variablen in IPython:
%reset

Darstellung von Matrizen:

pylab.matshow(A)
@author: johannesr
"""


import numpy as np
import scipy as sp
import Element
import Mesh
# from scipy import sparse

# Zeitmessung:
import time


my_mesh = Mesh.Mesh_generator(x_len=3*3, y_len=4*3, x_no_elements=3*3, y_no_elements=3*3)
my_mesh.build_mesh()


t1 = time.clock()

# Hier kommt das Assembly-Zeugs:
nodes = np.array(my_mesh.nodes)[:,1:]
elements = np.array(my_mesh.elements, dtype=int)[:,1:]

ndof_global = nodes.shape[0]
row_global = []
col_global = []
vals_global = []


# Erzeugung eines Dreieckselements
my_element = Element.ElementPlanar(E_modul=10)



for element in elements:
    # vielleicht a bisserl ineffizient... ;-)
    x = np.array([nodes[i] for i in element]).reshape(-1)
    ndof_local = x.shape[0]
    # Nur Test-Zeugs, um die Performance zu checken
    # k_element = np.zeros((ndof_local, ndof_local))
    k_element = my_element.k_int(x, x)
    counter = 0
    row = np.zeros(ndof_local**2)
    col = np.zeros(ndof_local**2)
    vals = np.zeros(ndof_local**2)
    # Double-Loop for indexing
    for i_loc, i_glob in enumerate(element):
        for j_loc, j_glob in enumerate(element):
            row[counter] = i_glob
            col[counter] = j_glob
            vals[counter] = k_element[i_loc, j_loc]
            counter += 1
    row_global.append(row)
    col_global.append(col)
    vals_global.append(vals)

t2 = time.clock()
row_global = np.array(row_global).reshape(-1)
col_global = np.array(col_global).reshape(-1)
vals_global = np.array(vals_global).reshape(-1)


K_coo = sp.sparse.coo_matrix((vals_global, (row_global, col_global)), dtype=float)
K_array = K_coo.toarray()
K_csr = K_coo.tocsr()
K_csr.eliminate_zeros()
t3 = time.clock()
print('Rechenzeit für ', ndof_global, 'Freiheitsgrade für Assembly:', t2-t1, 's')
print('und die Rechenzeit für Aufbau der Matrix:', t3-t2)

# Timeit
#import timeit
#t = timeit.Timer('K_array = K_coo.toarray()', time.time)
#zeit = t.timeit()


# Analyse der Assemblierten Matrix
import matplotlib.pylab as pylab
pylab.matshow(K_array)

# Test der
force_vector = np.zeros(ndof_global)
force_index = 55
force_vector[55] = 100



def apply_boundaries(K, boundary_indices):
    '''
    Funktion zum Aufbringen von Randbedingungen an den Vektor bzw. die Matrix K,
    dessen Zeilen (und Spalten) gelöscht werden, wenn diese in der Liste oder dem Array boundary_indices auftreten.
    '''
    ndof = K.shape[0]
    positive_index_list = [i for i in range(ndof) if not i in boundary_indices]
    if len(K.shape) == 1:
        return K[positive_index_list]
    elif len(K.shape) ==2:
        return K[np.ix_(positive_index_list, positive_index_list)]
    else:
        print('Kein Vektor oder keine Matrix übergeben')


boundary_indices = np.arange(10)
K_bound = apply_boundaries(K_array, boundary_indices)
f_bound = apply_boundaries(force_vector, boundary_indices)
u = np.linalg.solve(K_bound, f_bound)








## Test
#my_meshgenerator = Mesh_generator(x_len=3*3, y_len=4*3, x_no_elements=3*3*3, y_no_elements=3*3*3, height = 1.5, x_curve=True, y_curve=False)
#my_meshgenerator.build_mesh()
#my_meshgenerator.save_mesh('saved_nodes.csv', 'saved_elements.csv')
#
#my_mesh = Mesh()
#my_mesh.read_elements('saved_elements.csv')
#my_mesh.read_nodes('saved_nodes.csv', node_dof=3)
#my_mesh.save_mesh_for_paraview('myfilename')
#

#%%