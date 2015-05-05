#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:13:52 2015

Löschen aller Variablen in IPython:
%reset

Darstellung von Matrizen:
pylab.matshow(A)



@author: Johannes Rutzmoser
"""


import numpy as np
import scipy as sp
import Element
import Mesh
from scipy import sparse
from scipy import linalg

import multiprocessing as mp
from multiprocessing import Pool

# Zeitmessung:
import time

class PrimitiveAssembly():
    '''
    Assemblierungsklasse, die für gegebene Tableaus von Knotenkoordinaten und Assemblierungsknoten eine Matrix assembliert
    '''

    def __init__(self, node_coordinates_array = None, element_assembly_array=None, matrix_function=None, ndof_node=2):
        '''
        Verlangt ein dreispaltiges Koordinatenarray, indem die Koordinaten in x, y, und z-Koordinaten angegeben sind
        Anzahl der Freiheitsgrade für einen Knotenfreiheitsgrad: ndof_node gibt an, welche Koordinaten verwendet werden sollen;
        Wenn mehr Koordinaten pro Knoten nötig sind (z.B. finite Rotationen), werden Nullen hinzugefügt
        '''
        self.nodes = node_coordinates_array
        self.elements = element_assembly_array
        self.matrix_function = matrix_function
        self.ndof_node = ndof_node
        self.ndof_global = self.nodes.size*self.ndof_node
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        pass


    def assemble(self, u=None):
        '''
        assembliert die matrix_function für eine gegebene aktuelle Konfiguration x und die Ursprungskonfiguration X
        möglicherweise ist eine verschiebungsbasierte Darstellung mit X und u geschickter! Wäre näher dran an der Realität und man könnte einfacher lineare Elemente beschreiben, weil per Default u zu null gesetzt wird!
        '''
        # Löschen der alten Variablen
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        # Anzahl der lokalen Freiheitsgrade
        ndof_local = len(self.elements[0])*self.ndof_node
        # Wenn keine Verschiebung gegeben ist, wird u_local gleich 0 gesetzt
        u_local = np.zeros(ndof_local)
        for element in self.elements:
            # Koordinaten des elements
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            # element_indices have to be corrected in order respect the dimensions
            element_indices = np.array([[2*i + j for j in range(self.ndof_node)] for i in element]).reshape(-1)
            if u:
                u_local = u(element_indices)
            element_matrix = self.matrix_function(X, u_local)
            row = np.zeros((ndof_local, ndof_local))
            row[:,:] = element_indices
            self.row_global.append(row.reshape(-1))
            self.col_global.append((row.T).reshape(-1))
            self.vals_global.append(element_matrix.reshape(-1))
            pass
        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo


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

def remove_boundaries(K, boundary_indices):
    '''
    Ist die inverse Funktion zu apply_boundaries; Rekonstruiert den vollen Verschiebungsvektor bzw. die volle Matrix mit leeren Zeilen und Spalten
    '''
    ndof_red = K.shape[0]
    ndof = ndof_red + len(boundary_indices)
    positive_index_list = [i for i in range(ndof) if not i in boundary_indices]
    if len(K.shape) == 1:
        K_return = np.zeros(ndof)
        K_return[positive_index_list] = K
    elif len(K.shape) == 2:
        K_return = np.zeros((ndof, ndof))
        K_return[np.ix_(positive_index_list, positive_index_list)] = K
    return K_return



class MultiprocessAssembly():
    '''
    Klasse um schnell im Multiprozess zu assemblieren; Verteilt die Assemblierung auf alle Assemblierungsklassen und summiert die anschließend alles auf
    - funktioniert nicht so schnell, wie ich es erwartet hätte; großer Aufwand scheint nach wie vor zu sein,

    '''
    def __init__(self, assembly_class, list_of_matrix_functions, nodes_array, element_array):
        '''
        ???
        '''
        self.no_of_processes = len(list_of_matrix_functions)
        self.nodes_array = nodes_array
        self.element_array = element_array
        self.list_of_matrix_functions = list_of_matrix_functions
        domain_size = self.nodes_array.shape[0]//self.no_of_processes
        element_domain_list = []
        for i in range(self.no_of_processes - 1):
            element_domain_list.append(self.element_array[i*domain_size:(i+1)*domain_size,:])
        element_domain_list.append(self.element_array[(i+1)*domain_size:,:]) # assemble last domain to the end in order to consider flooring above
        self.assembly_class_list = [assembly_class(self.nodes_array, element_domain_list[i], matrix_function) for i, matrix_function in enumerate(list_of_matrix_functions)]
        pass

    def assemble(self):
        '''
        assembles the mesh with a multiprocessing routine
        '''
        pool = mp.Pool(processes=self.no_of_processes)
        results = [pool.apply_async(assembly_class.assemble) for assembly_class in self.assembly_class_list]
        matrix_coo_list = [j.get() for j in results]
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        data_global = np.array([], dtype=float)
        for matrix_coo in matrix_coo_list:
            row_global = np.append(row_global, matrix_coo.row)
            col_global = np.append(col_global, matrix_coo.col)
            data_global = np.append(data_global, matrix_coo.data)
        matrix_coo = sp.sparse.coo_matrix((data_global, (row_global, col_global)), dtype=float)
        return matrix_coo


class ConvertIndices():
    '''
    Klasse, die sich um die Konversion von Indizes Kümmert. Hauptfunktion ist, dass die Indizes von Knotendarstellungen und Koordinatenrichtung in Voigt-Notation übertragen werden können und umgekehrt.     
    '''
    def __init__(self, no_of_dofs_per_node=2):
        self.node_dof = no_of_dofs_per_node        
        pass
    
    def node2total(self, node_index, coordinate_index):
        '''
        Konvertiert den Knotenindex, wie er im Mesh-Programm dargestellt wird 
        zusammen mit dem Index der Koordinatenrichtung zum globalen Freiheitsgrad        
        '''
        return node_index*self.node_dof + coordinate_index
    
    def total2node(self, total_dof):
        '''
        Konvertiert den globalen Freiheitsgrad, wie er von den Berechnungsroutinen 
        verwendet wird, zu Knoten- und Indexfreiheitsgrad
        '''
        return total_dof // self.node_dof, total_dof%self.node_dof


class Boundary():
    '''
    Randbedingungen-Klasse: 
    
    Mit ihr können die Randbedingungen auf eine Struktur aufgebracht werden. Es werden generell Master-Slave-Randbedingungen unterstützt... 
    
    Generell sind zwei Lösungsmöglichkeiten da: Streichen der Slave-Koordinaten und Elimination der Slave-Koordinaten,     
    
    Die B-Matrix kann zur Verfügung gestellt werden
    '''
    def __init__(self, ndof_full_system, master_slave_list=None):
        self.ndof_full_system = ndof_full_system
        self.master_slave_list = master_slave_list
        pass

    
    def b_matrix(self):
        '''
        Erzeugt die B-Matrix, die die globalen (u_global) Freiheitsgrade mit den beschränkten Freiheitsgraden (u_bound) verbindet: 
        
        u_global = B*u_bound
        
        Die globalen 
        '''
        B = sp.sparse.eye(self.ndof_full_system).tocsr()
        global_slave_node_list = np.array([], dtype=int)
        for master_node, slave_node_list, b_matrix in self.master_slave_list:
            if (type(b_matrix) != type(np.zeros(1))): # a little hack in order to get the types right
                # Make a B-Matrix, if it's not there
                b_matrix = np.ones(len(slave_node_list))
            if master_node != None:
                # check, if the master node is existent; otherwise the columns will only be deleted
                for i in range(len(slave_node_list)):
                    # Hier werden die Sklaven-Knoten auf die Master-Knoten aufaddiert
                    col = B[:,slave_node_list[i]]*b_matrix[i]
                    row_indices = col.nonzero()[0]
                    no_of_nonzero_entries = row_indices.shape[0]
                    col_indices = np.ones(no_of_nonzero_entries)*master_node
                    B = B + sp.sparse.csr_matrix((col.data, (row_indices, col_indices)), shape=(self.ndof_full_system, self.ndof_full_system))
                # Der Master-Knoten wird nochmals subtrahiert:
                B[master_node, master_node] -= 1
            # Lösche den Master-Knoten aus der slave_node_list, um ihn zu erhalten...
            cleaned_slave_node_list = [x for x in slave_node_list if x != master_node]
            global_slave_node_list = np.append(global_slave_node_list, cleaned_slave_node_list)
        # delete the rows which are slave-nodes with a boolean mask:
        mask = np.ones(self.ndof_full_system, dtype=bool)
        mask[global_slave_node_list] = False
        B = B[:,mask]
        return B
    
if False:
    # Test of the Boundary-Class-Routine; Should work more or less now!    
    master_slave_list = [[0, [0, 1, 2, 5], np.array([0.5, 1, 1.5, 3])], ]
    #master_slave_list = [[4, [0, 1, 2], None, ]]
    my_boundary = Boundary(10, master_slave_list)
    B = my_boundary.b_matrix()
    B_array = B.toarray()


   
#%%

#
#t1 = time.clock()
#
## Netz
#my_meshgenerator = Mesh.Mesh_generator(x_len=3, y_len=3*3*3, x_no_elements=3*3*3*3*3, y_no_elements=3*3*3*3*3*3)
#my_meshgenerator.build_mesh()
#ndof = len(my_meshgenerator.nodes)*2
#nelements = len(my_meshgenerator.elements)
#nodes_array = np.array(my_meshgenerator.nodes)[:,1:]
#element_array = np.array(my_meshgenerator.elements)[:,1:]
#t2 = time.clock()
#print('Netz mit', ndof, 'Freiheitsgraden und', nelements, 'Elementen erstellt')
#
## Element
#my_element = Element.ElementPlanar()
#multiproc = False
#if multiproc:
#    no_of_proc= 6
#    element_object_list = [Element.ElementPlanar() for i in range(no_of_proc)]
#    element_function_list_k = [j.k_int for j in element_object_list]
#    element_function_list_m = [j.m_int for j in element_object_list]
#    my_multiprocessing = MultiprocessAssembly(PrimitiveAssembly, element_function_list_k, nodes_array, element_array)
#    K_coo = my_multiprocessing.assemble()
#else:
#    my_assembly = PrimitiveAssembly(np.array(my_meshgenerator.nodes)[:,1:], np.array(my_meshgenerator.elements)[:,1:], my_element.k_int)
#    K_coo = my_assembly.assemble()
#K = K_coo.tocsr()
#print('Matrix K assembliert')
#
#t3 = time.clock()
## update der Matrix-Funktion
#if multiproc:
#    my_multiprocessing = MultiprocessAssembly(PrimitiveAssembly, element_function_list_m, nodes_array, element_array)
#    M_coo = my_multiprocessing.assemble()
#else:
#    my_assembly.matrix_function = my_element.m_int
#    M_coo = my_assembly.assemble()
#M = M_coo.tocsr()
#print('Matrix M assembliert')
#
#t4 = time.clock()
#print('Das System hat', ndof, 'Freiheitsgrade und', nelements, 'Elemente.')
#print('Zeit zum generieren des Netzes:', t2 - t1)
#print('Zeit zum Assemblieren der Steifigkeitsmatrix:', t3 - t2)
#print('Zeit zum Assemblieren der Massenmatrix:', t4 - t3)
#
#
#shit_product = M.T.dot(K.dot(M))
#%%
#
#M = M_coo.toarray()
#K = K_coo.toarray()
#boundaries = [0, 1, 2, 3, 4, 5, 6]
#M_bound = apply_boundaries(M, boundaries)
#K_bound = apply_boundaries(K, boundaries)
#print('Randbedingungen aufgebracht')
#
## Hermit'sches generalisiertes Eigenwertproblem
#eigvals, eigvecs = sp.linalg.eigh(K, M)
#print('Eigenwertproblem gelöst')
#
##eigvecs = M*0
##for i in range(M_bound.shape[0]):
##    eigvecs[:,i] = remove_boundaries(eigvecs_bound[:,i], boundaries)
#
#print('Eigenformen geupdated')
#
## Mesh handling
#knotenfile = 'Vernetzungen/nodes.csv'
#elementfile = 'Vernetzungen/elements.csv'
#my_meshgenerator.save_mesh(knotenfile, elementfile)
#
#my_mesh = Mesh.Mesh()
#my_mesh.read_nodes(knotenfile)
#my_mesh.read_elements(elementfile)
##my_mesh.set_displacement(eigvecs[:,-2])
#my_mesh.set_displacement_with_time(eigvecs, len(eigvals))
#my_mesh.save_mesh_for_paraview('Versuche/Balken')
#
#
#import matplotlib.pylab as pylab
#pylab.plot(eigvals)


#%%
## Validierung der Elemente anhand von analytischer Balken-Eigenfrequenz:
#E_modul=210E9
#poisson_ratio=0.3
#element_thickness=1.
#density=10
#
#A = 3.
#I = 3**3/12
#
#l = 3*3*3*3
#m = A*l*density
#analytic_omega= 1.875**2*np.sqrt(E_modul*I/(m*l**4))
#omega = np.sqrt(eigvals_bound[0])
#

#%%
#
#t1 = time.clock()
#
#
## Hier kommt das Assembly-Zeugs:
#nodes = np.array(my_meshgenerator.nodes)[:,1:]
#elements = np.array(my_meshgenerator.elements, dtype=int)[:,1:]
#
#
#
#ndof_global = nodes.size
#row_global = []
#col_global = []
#vals_global = []
#
#
## Erzeugung eines Dreieckselements
#my_element = Element.ElementPlanar(E_modul=10)
#
#
## Schleife über alle Elemente
#for element in elements:
#    # vielleicht a bisserl ineffizient... ;-)
#    # Knotenpositionen x des Elements
#    x = np.array([nodes[i] for i in element]).reshape(-1)
#    ndof_local = len(x)
#    # Nur Test-Zeugs, um die Performance zu checken
#    # k_element = np.zeros((ndof_local, ndof_local))
#    k_element = my_element.k_int(x, x)
#    counter = 0
#    row = np.zeros(ndof_local**2)
#    col = np.zeros(ndof_local**2)
#    vals = np.zeros(ndof_local**2)
#    # element_indices have to be corrected in order respect the dimensions
#    # Attention: This only works for planar scenarios. Otherwise the list comprehension hast to be changed!
#    element_indices = np.array([(2*i, 2*i+1) for i in element]).reshape(-1)
#    # Double-Loop for indexing
#    for i_loc, i_glob in enumerate(element_indices):
#        for j_loc, j_glob in enumerate(element_indices):
#            row[counter] = i_glob
#            col[counter] = j_glob
#            vals[counter] = k_element[i_loc, j_loc]
#            counter += 1
#    row_global.append(row)
#    col_global.append(col)
#    vals_global.append(vals)
#
#t2 = time.clock()
#row_global = np.array(row_global).reshape(-1)
#col_global = np.array(col_global).reshape(-1)
#vals_global = np.array(vals_global).reshape(-1)
#
#
#K_coo = sp.sparse.coo_matrix((vals_global, (row_global, col_global)), dtype=float)
#K_array = K_coo.toarray()
#K_csr = K_coo.tocsr()
#K_csr.eliminate_zeros()
#t3 = time.clock()
#print('Rechenzeit für', ndof_global, 'Knotenfreiheitsgrade und ', elements.shape[0], 'Elementen für Assembly:', t2-t1, 's')
#print('und die Rechenzeit für Aufbau der Matrix:', t3-t2)
#
## Timeit
##import timeit
##t = timeit.Timer('K_array = K_coo.toarray()', time.time)
##zeit = t.timeit()
#
#
## Analyse der Assemblierten Matrix
#import matplotlib.pylab as pylab
#pylab.matshow(K_array)
#
## Kraft-Randbedingungen
#force_vector = np.zeros(ndof_global)
#force_index = -1
#force_vector[force_index] = 1.
#
#
#
#def apply_boundaries(K, boundary_indices):
#    '''
#    Funktion zum Aufbringen von Randbedingungen an den Vektor bzw. die Matrix K,
#    dessen Zeilen (und Spalten) gelöscht werden, wenn diese in der Liste oder dem Array boundary_indices auftreten.
#    '''
#    ndof = K.shape[0]
#    positive_index_list = [i for i in range(ndof) if not i in boundary_indices]
#    if len(K.shape) == 1:
#        return K[positive_index_list]
#    elif len(K.shape) ==2:
#        return K[np.ix_(positive_index_list, positive_index_list)]
#    else:
#        print('Kein Vektor oder keine Matrix übergeben')
#
#def remove_boundaries(K, boundary_indices):
#    '''
#    Ist die inverse Funktion zu apply_boundaries; Rekonstruiert den vollen Verschiebungsvektor bzw. die volle Matrix mit leeren Zeilen und Spalten
#    '''
#    ndof_red = K.shape[0]
#    ndof = ndof_red + len(boundary_indices)
#    positive_index_list = [i for i in range(ndof) if not i in boundary_indices]
#    if len(K.shape) == 1:
#        K_return = np.zeros(ndof)
#        K_return[positive_index_list] = K
#    elif len(K.shape) == 2:
#        K_return = np.zeros((ndof, ndof))
#        K_return[np.ix_(positive_index_list, positive_index_list)] = K
#    return K_return
#
#boundary_indices = np.arange(0,2*27,3)
#K_bound = apply_boundaries(K_array, boundary_indices)
#f_bound = apply_boundaries(force_vector, boundary_indices)
#u_bound = np.linalg.solve(K_bound, f_bound)
#
#u = remove_boundaries(u_bound, boundary_indices)
#K_new = remove_boundaries(K_bound, boundary_indices)
#
#t4 = time.clock()
#print('und die Rechenzeit fürs Lösen des Systems:', t4-t3)
#
#
#
## Mesh handling
#knotenfile = 'Vernetzungen/nodes.csv'
#elementfile = 'Vernetzungen/elements.csv'
#my_meshgenerator.save_mesh(knotenfile, elementfile)
#
#my_mesh = Mesh.Mesh()
#my_mesh.read_nodes(knotenfile)
#my_mesh.read_elements(elementfile)
#my_mesh.set_displacement(u)
#my_mesh.save_mesh_for_paraview('Sepp')
#
#
#

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
