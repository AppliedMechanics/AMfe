# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:14:28 2015

@author: fabian
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# make amfe running
import sys
sys.path.insert(0, '../..')
import amfe
import plot_bar

#%% Geometric parameters
l1 = 4
l2 = 3

E_modul=1.0
density=1.0
A = 1.0

# %%Nodes
nodes = np.array([[0, 0]])
nodes = np.append(nodes,[[0,l2]],axis=0)
nodes = np.append(nodes,[[0,2*l2]],axis=0)
for i in range(nodes.shape[0],30):  
    nodes = np.append(nodes,nodes[i-3,:]+[[l1,0]],axis=0)
    
#%% Elements
# Vertical bars
elements = np.array([[0,1]])
elements = np.append(elements,[[1,2]],axis=0)
for i in range(2,20):
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0) 
    
# Horizontal bars
elements = np.append(elements,[[0,3]],axis=0)
elements = np.append(elements,[[1,4]],axis=0)
elements = np.append(elements,[[2,5]],axis=0)
for i in range(23,47):
    elements = np.append(elements,elements[i-3,:]+[[3,3]],axis=0)    

# Diagonal bars
elements = np.append(elements,[[3,1]],axis=0)
elements = np.append(elements,[[1,5]],axis=0)
for i in range(49,65):
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0)


my_mesh_generator = amfe.MeshGenerator()
my_mesh_generator.nodes = nodes
my_mesh_generator.elements = elements
my_mesh_generator.save_mesh('./mesh/full_system/nodes.csv',
                            './mesh/full_system/elements.csv')


#%% Building the mechanical system

# Initialize system
my_system = amfe.MechanicalSystem()
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0,
                                  mesh_style='Bar')
# Load mesh
my_system.load_mesh_from_csv('./mesh/full_system/nodes.csv',
                             './mesh/full_system/elements.csv')


# Plot mesh of bars
pos_of_nodes = nodes.reshape((-1, 1))    
plot_bar.plt_mesh(elements, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='b', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')   

plt.show()
