# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:22:51 2015

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
l1 = 4.0
l2 = 3.0

E_modul=1.0
density=1.0
A = 1.0

#%%Nodes
nodes = np.array([[0, 0]])
nodes = np.append(nodes,[[0,l2]],axis=0)
nodes = np.append(nodes,[[0,2*l2]],axis=0)
for i in range(nodes.shape[0],30):  
    nodes = np.append(nodes,nodes[i-3,:]+[[l1,0]],axis=0)
    
#%% Elements left substructure
# Vertical bars
elements = np.array([[0,1]])
elements = np.append(elements,[[1,2]],axis=0)
for i in range(2,12): #12 statt 20
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0) 
    
# Horizontal bars
elements = np.append(elements,[[0,3]],axis=0)
elements = np.append(elements,[[1,4]],axis=0)
elements = np.append(elements,[[2,5]],axis=0)
for i in range(15,27): 
    elements = np.append(elements,elements[i-3,:]+[[3,3]],axis=0)    

# Diagonal bars
elements = np.append(elements,[[3,1]],axis=0)
elements = np.append(elements,[[1,5]],axis=0)
for i in range(29,37): 
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0)

# Determine active nodes
active_nodes = np.unique(elements)
nodes_loc = nodes[active_nodes,:]

# Plot mesh of bars
pos_of_nodes = nodes.reshape((-1, 1))    
plot_bar.plt_mesh(elements, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')  




# Call mesh_generator to save the mesh as csv-files
my_mesh_generator = amfe.MeshGenerator(mesh_style='Bar2D')
my_mesh_generator.nodes = nodes
my_mesh_generator.elements = elements
my_mesh_generator.save_mesh('./mesh/substructures/nodes-sub1-glob.csv',
                            './mesh/substructures/elements-sub1.csv')
                            
                            
                            
                            
                        
#%% Elements right substructure
# Vertical bars
elements = np.array([[18,19]])
elements = np.append(elements,[[19,20]],axis=0)
for i in range(2,8):
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0) 
    
# Horizontal bars
elements = np.append(elements,[[15,18]],axis=0)
elements = np.append(elements,[[16,19]],axis=0)
elements = np.append(elements,[[17,20]],axis=0)
for i in range(11,20):
    elements = np.append(elements,elements[i-3,:]+[[3,3]],axis=0)    

# Diagonal bars
elements = np.append(elements,[[18,16]],axis=0)
elements = np.append(elements,[[16,20]],axis=0)
for i in range(22,28):
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0)


# Plot mesh of bars
pos_of_nodes = nodes.reshape((-1, 1))    
plot_bar.plt_mesh(elements, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=2, p_title='Mesh',
                  p_col_node='r')  



# Call mesh_generator to save the mesh as csv-files
my_mesh_generator = amfe.MeshGenerator(mesh_style='Bar2D')
my_mesh_generator.nodes = nodes
my_mesh_generator.elements = elements
my_mesh_generator.save_mesh('./mesh/substructures/nodes-sub2-glob.csv',
                            './mesh/substructures/elements-sub2.csv')
