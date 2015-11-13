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
l1 = 4.0
l2 = 3.0

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


my_mesh_generator = amfe.MeshGenerator(mesh_style='Bar2D')
my_mesh_generator.nodes = nodes
my_mesh_generator.elements = elements
my_mesh_generator.save_mesh('./mesh/full_system/nodes.csv',
                            './mesh/full_system/elements.csv')


#%% Building the mechanical system

# Initialize system
my_system = amfe.MechanicalSystem(E_modul=1.0, crosssec=1.0, density=1.0)
# Load mesh
my_system.load_mesh_from_csv('./mesh/full_system/nodes.csv',
                             './mesh/full_system/elements.csv',
                             ele_type = 'Bar2Dlumped')


# Dirichlet_boundary conditions
#my_system.apply_dirichlet_boundaries()


# Build mass and stiffness matrix
M, K = amfe.give_mass_and_stiffness(my_system)
lam, phi_i = sp.sparse.linalg.eigsh(K, k=20, M=M, which='SM')

# Plot mesh of bars
pos_of_nodes = nodes.reshape((-1, 1))    
plot_bar.plt_mesh(elements, pos_of_nodes, plot_no_of_ele=False, 
                  plot_nodes=False, p_col='0.25', no_of_fig=1, p_title='Mesh',
                  p_col_node='r')   
scale = 20
disp_eigenmodes = pos_of_nodes + phi_i*scale
plot_bar.plt_mesh(elements, disp_eigenmodes[:,6], plot_no_of_ele=True, 
                  plot_nodes=True, p_col='b', no_of_fig=1, p_title='Eigenmode',
                  p_col_node='r')

plt.show()
