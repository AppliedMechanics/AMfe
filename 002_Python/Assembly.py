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
        assembliert die matrix_function für die Ursprungskonfiguration X und die Verschiebung u. 
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
        
        Die globalen Freiheitsgrade differieren daher von den Freiheitsgraden des beschränkten Systems; 
        Eingabeparameter ist eine Liste mit Dirichlet-Randbedingungen: 
        [Master-DOF, [Liste_von_Sklaven-DOFs], Gewichtungsvektor]
        
        Master-DOF: (typ: int) Der DOF, auf den die Sklaven-DOFs projiziert werden. Der Master-DOF wird am ende eliminiert, d.h. er sollte üblicherweise auch in den Sklaven-DOFs auftauchen
        
        [Liste_von_Sklaven-DOFs]: (typ: liste mit ints) Die DOFs, die auf den Master-DOF projiziert werden. Zur Gewichtung wird der Gewichtungsvektor angewendet, der genauso viele Einträge haben muss wie die Sklaven-DOF-Liste
        
        Gewichtungsvektor: (typ: np.array oder None) TODO Beschreibung
        
        
        Wichtig: Für die Dirichlet-Randbedingungen werden Freiheitsgrade des globalen Systems und nicht die Knotenfreiheitsgrade berücksichtigt. Die Indexwerte der Knoten müssen stets in DOFs des globalen Sytems umgerechnet werden
        '''
        B = sp.sparse.eye(self.ndof_full_system).tocsr()
        global_slave_node_list = np.array([], dtype=int)
        for master_node, slave_node_list, b_matrix in self.master_slave_list:
            if (type(b_matrix) != type(np.zeros(1))): # a little hack in order to get the types right
                # Make a B-Matrix, if it's not there
                b_matrix = np.ones(len(slave_node_list))
            if len(slave_node_list) != len(b_matrix):
                raise ValueError('Die Dimension der Sklaven-Knotenliste entspricht nicht der Dimension des Gewichtungsvektors!')
            if master_node != None: # check, if the master node is existent; otherwise the columns will only be deleted
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


###############################################################################
## Ist eher veraletetes Zeugs; soll durch die Boundary-Klasse besser gemacht und ersetzt werden. 
###############################################################################

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

