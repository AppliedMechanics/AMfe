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
from scipy import sparse
from scipy import linalg

import multiprocessing as mp
from multiprocessing import Pool


class PrimitiveAssembly():
    '''
    Assemblierungsklasse, die für gegebene Tableaus von Knotenkoordinaten und Assemblierungsknoten eine Matrix assembliert
    '''

    # Hier muessen wir uns mal genau ueberlegen, was alles dem assembly uebergeben werden soll
    # ob das ganze Mesh, oder nur ein paar Attribute
    def __init__(self, mesh = None, matrix_function=None, node_dof=2, vector_function=None):
        '''
        Verlangt ein dreispaltiges Koordinatenarray, indem die Koordinaten in x, y, und z-Koordinaten angegeben sind
        Anzahl der Freiheitsgrade für einen Knotenfreiheitsgrad: node_dof gibt an, welche Koordinaten verwendet werden sollen;
        Wenn mehr Koordinaten pro Knoten nötig sind (z.B. finite Rotationen), werden Nullen hinzugefügt
        '''
        self.nodes = mesh.nodes
        self.elements = mesh.elements
        self.matrix_function = matrix_function
        self.vector_function = vector_function
        self.node_dof = mesh.node_dof
        self.ndof_global = mesh.no_of_dofs
        self.no_of_element_nodes = mesh.no_of_element_nodes
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        pass


    def assemble_matrix(self, u=None):
        '''
        assembliert die matrix_function für die Ursprungskonfiguration X und die Verschiebung u.
        '''
        # deletion of former variables
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        # number of dofs per element (6 for triangle since no_of_element_nodes = 3 and node_dof = 2)
        ndof_local = self.no_of_element_nodes*self.node_dof
        # preset for u_local; necessary, when u=None       
        u_local = np.zeros(ndof_local)

        for element in self.elements:
            # Koordinaten des elements
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            # element_indices have to be corrected in order respect the dimensions
            element_indices = np.array([[self.node_dof*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            if u is not None:
                u_local = u[element_indices]
            element_matrix = self.matrix_function(X, u_local)
            row = np.zeros((ndof_local, ndof_local))
            row[:,:] = element_indices
            self.row_global.append(row.reshape(-1))
            self.col_global.append((row.T).reshape(-1))
            self.vals_global.append(element_matrix.reshape(-1))

        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo

    def assemble_vector(self, u):
        '''
        Assembliert die Force-Function für die Usprungskonfiguration X und die Verschiebung u
        '''
        global_force = np.zeros(self.ndof_global)
        for element in self.elements:
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            element_indices = np.array([[2*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            global_force[element_indices] += self.vector_function(X, u[element_indices])
        return global_force







class MultiprocessAssembly():
    '''
    Klasse um schnell im Multiprozess zu assemblieren; Verteilt die Assemblierung auf alle Assemblierungsklassen und summiert die anschließend alles auf
    - funktioniert nicht so schnell, wie ich es erwartet hätte; genauere Analysen bisher noch nicht vorhanden, da profile-Tool nich zuverlässig für multiprocessing-Probleme zu funktionieren scheint.
    - ACHTUNG: Diese Klasse ist derzeit nicht in aktiver Nutzung. Möglicherweise macht es Sinn, diese Klasse zu überarbeiten, da sich die gesamte Programmstruktur gerade noch ändert.
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



#%%

################################################################################
### Ist eher veraltetes Zeugs; soll durch die Boundary-Klasse besser gemacht und ersetzt werden.
################################################################################
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

