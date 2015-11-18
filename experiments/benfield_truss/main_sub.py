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
my_system1 = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system1.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M1, K1 = amfe.give_mass_and_stiffness(my_system1)

# Plot mesh of bars
pos_of_nodes1 = my_system1.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system1.element_list, pos_of_nodes1, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')  


# Determine boundary and internal dofs
dof_1 = np.arange(my_system1.node_list.shape[0]*2)
nodes_b1 = np.where(my_system1.node_list[:, 0] == 5*l1)[0]
dof_b1 = np.sort(np.concatenate((nodes_b1*2, nodes_b1*2+1), axis=1))
dof_i1 = np.setdiff1d(dof_1,dof_b1)                           
B1 = np.zeros((6,dof_1.shape[0]))
B1[:,dof_b1] = np.identity((6))
L1 = np.identity((6))
                           
#%% Right substructure
# Check if data is already existing
path_nod = './mesh/substructures/nodes-sub2.csv'
path_ele = './mesh/substructures/elements-sub2.csv'
if not os.path.exists(path_nod) or not os.path.exists(path_ele):
    create_sub.create_substructure()
    print("Had to create Benfield truss since", path_nod, "or", path_ele,
        "did not exist\n")                           

# Building the mechanical system
my_system2 = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
my_system2.load_mesh_from_csv(path_nod, path_ele, ele_type = 'Bar2Dlumped')
M2, K2 = amfe.give_mass_and_stiffness(my_system2)             

# Plot mesh of bars
pos_of_nodes2 = my_system2.node_list.reshape((-1, 1))    
plot_bar.plt_mesh(my_system2.element_list, pos_of_nodes2, plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=2, p_title='Mesh',
                  p_col_node='r')  
                             

# Determine interface dofs
dof_2 = np.arange(my_system2.node_list.shape[0]*2)
nodes_b2 = np.where(my_system2.node_list[:, 0] == 5*l1)[0]
dof_b2 = np.sort(np.concatenate((nodes_b2*2, nodes_b2*2+1), axis=1))                
dof_i2 = np.setdiff1d(dof_2,dof_b2)  
B2 = np.zeros((6,dof_2.shape[0]))
B2[:,dof_b2] = np.identity((6))
L2 = np.identity((6))





#%% IN CONSTRUCTION: CRAIG BAMPTON METHODE
n_modes1 = 6
n_modes2 = 6


def CraigBampton(K, M, dof_i, dof_b, n_modes=6):
    # Eigenmodes
    Om2, phi = sp.sparse.linalg.eigsh(K[np.ix_(dof_i,dof_i)], k=n_modes, 
                                      M=M[np.ix_(dof_i,dof_i)], which='SM')
    # Static mode shapes
    psi = sp.sparse.linalg.spsolve(K[np.ix_(dof_i,dof_i)], -K[np.ix_(dof_i,dof_b)])   
    
    return Om2, phi, psi


def assemble_CBsys():
    
    pass


class Substructure():
    '''
    Class for substructures
    '''

    def __init__(self, K=[], M=[], dof_i=[], dof_b=[]):
        self.K = K
        self.M = M
        self.dof_i = dof_i
        self.dof_b = dof_b
        self.B = []

sub1 = Substructure(K1,M1,dof_i1,dof_b1)
sub2 = Substructure(K2,M2,dof_i2,dof_b2)

substructure_list = [sub1, sub2]


Om2_1, phi_1, psi_1 = CraigBampton(K1,M1,dof_i1,dof_b1, n_modes=n_modes1)
Om2_2, phi_2, psi_2 = CraigBampton(K2,M2,dof_i2,dof_b2, n_modes=n_modes2)
print(Om2_1)
print(Om2_2)















pos_of_nodes1 = my_system1.node_list.reshape((-1, 1))
disp_fix1 = np.zeros((dof_1.shape[0],n_modes1))
disp_fix1[dof_i1,:] = phi_1
scale = 10
disp1 = pos_of_nodes1 + scale*disp_fix1
 
plot_bar.plt_mesh(my_system1.element_list, disp1[:,3], plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=3, p_title='Shape',
                  p_col_node='r') 


pos_of_nodes2 = my_system2.node_list.reshape((-1, 1))
disp_fix2 = np.zeros((dof_2.shape[0],n_modes2))
disp_fix2[dof_i2,:] = phi_2
scale = 10
disp2 = pos_of_nodes2 + scale*disp_fix2
  
plot_bar.plt_mesh(my_system2.element_list, disp2[:,3], plot_no_of_ele=True, 
                  plot_nodes=True, p_col='0.25', no_of_fig=4, p_title='Shape',
                  p_col_node='r')



