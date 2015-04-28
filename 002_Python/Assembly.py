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
        self.ndof_global = self.nodes.size
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
#            print('Element-Matrix:', element_matrix)
#            print('X, u_local', X, u_local)
            counter = 0
            row = np.zeros(ndof_local**2)
            col = np.zeros(ndof_local**2)
            vals = np.zeros(ndof_local**2)
            # Double-Loop for indexing
            for i_loc, i_glob in enumerate(element_indices):
                for j_loc, j_glob in enumerate(element_indices):
                    row[counter] = i_glob
                    col[counter] = j_glob
                    vals[counter] = element_matrix[i_loc, j_loc]
                    counter += 1
            self.row_global.append(row)
            self.col_global.append(col)
            self.vals_global.append(vals)
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



#%%
t1 = time.clock()

# Netz
my_meshgenerator = Mesh.Mesh_generator(x_len=3, y_len=3*3, x_no_elements=3*3*3, y_no_elements=3*3*3)
my_meshgenerator.build_mesh()
ndof = len(my_meshgenerator.nodes)*2
nelements = len(my_meshgenerator.elements)
t2 = time.clock()

# Element
my_element = Element.ElementPlanar()
my_assembly = PrimitiveAssembly(np.array(my_meshgenerator.nodes)[:,1:], np.array(my_meshgenerator.elements)[:,1:], my_element.k_int)
K_coo = my_assembly.assemble()
#K = K_coo.toarray()
print('Matrix K assembliert')

t3 = time.clock()
# update der Matrix-Funktion
my_assembly.matrix_function = my_element.m_int
M_coo = my_assembly.assemble()
#M = M_coo.toarray()
print('Matrix M assembliert')

t4 = time.clock()
print('Das System hat', ndof, 'Freiheitsgrade und', nelements, 'Elemente.')
print('Zeit zum generieren des Netzes:', t2 - t1)
print('Zeit zum Assemblieren der Steifigkeitsmatrix:', t3 - t2)
print('Zeit zum Assemblieren der Massenmatrix:', t4 - t3)

#%%
boundaries = [0, 1, 2, 3, 4, 5, 6]
M_bound = apply_boundaries(M, boundaries)
K_bound = apply_boundaries(K, boundaries)
print('Randbedingungen aufgebracht')

# Hermit'sches generalisiertes Eigenwertproblem
eigvals_bound, eigvecs_bound = sp.linalg.eigh(K_bound, M_bound)
print('Eigenwertproblem gelöst')

eigvecs = M*0
for i in range(M_bound.shape[0]):
    eigvecs[:,i] = remove_boundaries(eigvecs_bound[:,i], boundaries)

print('Eigenformen geupdated')

# Mesh handling
knotenfile = 'Vernetzungen/nodes.csv'
elementfile = 'Vernetzungen/elements.csv'
my_meshgenerator.save_mesh(knotenfile, elementfile)

my_mesh = Mesh.Mesh()
my_mesh.read_nodes(knotenfile)
my_mesh.read_elements(elementfile)
#my_mesh.set_displacement(eigvecs[:,-2])
my_mesh.set_displacement_with_time(eigvecs, len(eigvals_bound))
my_mesh.save_mesh_for_paraview('Versuche/Balken')


import matplotlib.pylab as pylab
pylab.plot(eigvals_bound[:30])


#%%
# Validierung der Elemente anhand von analytischer Balken-Eigenfrequenz:
E_modul=210E9
poisson_ratio=0.3
element_thickness=1.
density=10

A = 3.
I = 3**3/12

l = 3*3*3*3
m = A*l*density
analytic_omega= 1.875**2*np.sqrt(E_modul*I/(m*l**4))
omega = np.sqrt(eigvals_bound[0])


#%%

t1 = time.clock()


# Hier kommt das Assembly-Zeugs:
nodes = np.array(my_meshgenerator.nodes)[:,1:]
elements = np.array(my_meshgenerator.elements, dtype=int)[:,1:]



ndof_global = nodes.size
row_global = []
col_global = []
vals_global = []


# Erzeugung eines Dreieckselements
my_element = Element.ElementPlanar(E_modul=10)


# Schleife über alle Elemente
for element in elements:
    # vielleicht a bisserl ineffizient... ;-)
    # Knotenpositionen x des Elements
    x = np.array([nodes[i] for i in element]).reshape(-1)
    ndof_local = len(x)
    # Nur Test-Zeugs, um die Performance zu checken
    # k_element = np.zeros((ndof_local, ndof_local))
    k_element = my_element.k_int(x, x)
    counter = 0
    row = np.zeros(ndof_local**2)
    col = np.zeros(ndof_local**2)
    vals = np.zeros(ndof_local**2)
    # element_indices have to be corrected in order respect the dimensions
    # Attention: This only works for planar scenarios. Otherwise the list comprehension hast to be changed!
    element_indices = np.array([(2*i, 2*i+1) for i in element]).reshape(-1)
    # Double-Loop for indexing
    for i_loc, i_glob in enumerate(element_indices):
        for j_loc, j_glob in enumerate(element_indices):
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
print('Rechenzeit für', ndof_global, 'Knotenfreiheitsgrade und ', elements.shape[0], 'Elementen für Assembly:', t2-t1, 's')
print('und die Rechenzeit für Aufbau der Matrix:', t3-t2)

# Timeit
#import timeit
#t = timeit.Timer('K_array = K_coo.toarray()', time.time)
#zeit = t.timeit()


# Analyse der Assemblierten Matrix
import matplotlib.pylab as pylab
pylab.matshow(K_array)

# Kraft-Randbedingungen
force_vector = np.zeros(ndof_global)
force_index = -1
force_vector[force_index] = 1.



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

boundary_indices = np.arange(0,2*27,3)
K_bound = apply_boundaries(K_array, boundary_indices)
f_bound = apply_boundaries(force_vector, boundary_indices)
u_bound = np.linalg.solve(K_bound, f_bound)

u = remove_boundaries(u_bound, boundary_indices)
K_new = remove_boundaries(K_bound, boundary_indices)

t4 = time.clock()
print('und die Rechenzeit fürs Lösen des Systems:', t4-t3)



# Mesh handling
knotenfile = 'Vernetzungen/nodes.csv'
elementfile = 'Vernetzungen/elements.csv'
my_meshgenerator.save_mesh(knotenfile, elementfile)

my_mesh = Mesh.Mesh()
my_mesh.read_nodes(knotenfile)
my_mesh.read_elements(elementfile)
my_mesh.set_displacement(u)
my_mesh.save_mesh_for_paraview('Sepp')




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
