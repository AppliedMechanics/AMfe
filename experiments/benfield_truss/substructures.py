#!/usr/bin/env python3
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

#%% Left substructure
# Nodes
nodes = np.array([[0, 0]])
nodes = np.append(nodes,[[0,l2]],axis=0)
nodes = np.append(nodes,[[0,2*l2]],axis=0)
for i in range(nodes.shape[0],18):  
    nodes = np.append(nodes,nodes[i-3,:]+[[l1,0]],axis=0)
    
# Elements 
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


# Plot mesh of bars
pos_of_nodes = nodes.reshape((-1, 1))    
plot_bar.plt_mesh(elements, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')  

# Call mesh_generator to save the mesh as csv-files
my_mesh_generator = amfe.MeshGenerator(mesh_style='Bar2D')
my_mesh_generator.nodes = nodes
my_mesh_generator.elements = elements
my_mesh_generator.save_mesh('./mesh/substructures/nodes-sub1.csv',
                            './mesh/substructures/elements-sub1.csv')
                           

# Building the mechanical system
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)

my_system.load_mesh_from_csv('./mesh/substructures/nodes-sub1.csv',
                             './mesh/substructures/elements-sub1.csv',
                             ele_type = 'Bar2Dlumped')

# Build mass and stiffness matrix
M1, K1 = amfe.give_mass_and_stiffness(my_system)

# Determine boundary and internal dofs
dof_1 = np.arange(my_system.node_list.shape[0])
nodes_b1 = np.where(my_system.node_list[:, 0] == 5*l1)[0]
dof_b1 = np.sort(np.concatenate((nodes_b1*2, nodes_b1*2+1), axis=1))
dof_i1 = np.setdiff1d(dof_1,dof_b1)                           
                            
#%% Right substructure
                            
# Nodes
nodes = np.array([[l1*5, 0]])
nodes = np.append(nodes,[[l1*5,l2]],axis=0)
nodes = np.append(nodes,[[l1*5,2*l2]],axis=0)
for i in range(nodes.shape[0],15):  
    nodes = np.append(nodes,nodes[i-3,:]+[[l1,0]],axis=0)                            
                        
# Elements right substructure
# Vertical bars
elements = np.array([[3,4]])
elements = np.append(elements,[[4,5]],axis=0)
for i in range(2,8):
    elements = np.append(elements,elements[i-2,:]+[[3,3]],axis=0) 
    
# Horizontal bars
elements = np.append(elements,[[0,3]],axis=0)
elements = np.append(elements,[[1,4]],axis=0)
elements = np.append(elements,[[2,5]],axis=0)
for i in range(11,20):
    elements = np.append(elements,elements[i-3,:]+[[3,3]],axis=0)    

# Diagonal bars
elements = np.append(elements,[[3,1]],axis=0)
elements = np.append(elements,[[1,5]],axis=0)
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
my_mesh_generator.save_mesh('./mesh/substructures/nodes-sub2.csv',
                            './mesh/substructures/elements-sub2.csv')

#%% Building the mechanical system
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)

my_system.load_mesh_from_csv('./mesh/substructures/nodes-sub2.csv',
                            './mesh/substructures/elements-sub2.csv',
                             ele_type = 'Bar2Dlumped')
                             
M2, K2 = amfe.give_mass_and_stiffness(my_system)             

# Determine interface dofs
dof_2 = np.arange(my_system.node_list.shape[0]*2)
nodes_b2 = np.where(my_system.node_list[:, 0] == 5*l1)[0]
dof_b2 = np.sort(np.concatenate((nodes_b2*2, nodes_b2*2+1), axis=1))                
dof_i2 = np.setdiff1d(dof_2,dof_b2)  

