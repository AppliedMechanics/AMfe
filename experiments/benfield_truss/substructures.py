#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:22:51 2015

@author: fabian
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

# make amfe running
import sys
sys.path.insert(0, '../..')
import amfe
import plot_bar
import create_sub

l1 = 4.0

#%% Left substructure
# Check if data is already existing
path_nod = './mesh/substructures/nodes-sub1.csv'
path_ele = './mesh/substructures/elements-sub1.csv'
if not os.path.exists(path_nod) or not os.path.exists(path_ele):
    create_sub.create_substructure()
    print("Had to create Benfield truss since", path_nod, "or", path_ele,
        "did not exist\n")
    
# Building the mechanical system
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M1, K1 = amfe.give_mass_and_stiffness(my_system)

# Plot mesh of bars
pos_of_nodes = my_system.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system.element_list, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')  


# Determine boundary and internal dofs
dof_1 = np.arange(my_system.node_list.shape[0])
nodes_b1 = np.where(my_system.node_list[:, 0] == 5*l1)[0]
dof_b1 = np.sort(np.concatenate((nodes_b1*2, nodes_b1*2+1), axis=1))
dof_i1 = np.setdiff1d(dof_1,dof_b1)                           


                           
#%% Right substructure
# Check if data is already existing
path_nod = './mesh/substructures/nodes-sub2.csv'
path_ele = './mesh/substructures/elements-sub2.csv'
if not os.path.exists(path_nod) or not os.path.exists(path_ele):
    create_sub.create_substructure()
    print("Had to create Benfield truss since", path_nod, "or", path_ele,
        "did not exist\n")                           

# Building the mechanical system
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M2, K2 = amfe.give_mass_and_stiffness(my_system)             

# Plot mesh of bars
pos_of_nodes = my_system.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system.element_list, pos_of_nodes, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=2, p_title='Mesh',
                  p_col_node='r')  
                             

# Determine interface dofs
dof_2 = np.arange(my_system.node_list.shape[0]*2)
nodes_b2 = np.where(my_system.node_list[:, 0] == 5*l1)[0]
dof_b2 = np.sort(np.concatenate((nodes_b2*2, nodes_b2*2+1), axis=1))                
dof_i2 = np.setdiff1d(dof_2,dof_b2)  

