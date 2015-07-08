#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:53:18 2015

@author: gruber
"""

import numpy as np
import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import time

# make amfe running
import sys
sys.path.insert(0,'..')
import amfe


#%% Mesh generation
x_len,y_len, x_no_elements, y_no_elements = 1,1,10,10 
pos_x0, pos_y0 = 0,0 
my_mesh_generator = amfe.MeshGenerator(x_len=x_len, y_len=y_len, 
                                       x_no_elements=x_no_elements, 
                                       y_no_elements=y_no_elements, 
                                       pos_x0 = pos_x0, pos_y0 = pos_y0,
                                       mesh_style = 'Quad4')
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('..meshes/selbstgebaut_quad/nodes.csv', 
                            '../meshes/selbstgebaut_quad/elements.csv')


#%% Building the mechanical system

# Initialize system
my_system = amfe.MechanicalSystem()

# Load mesh
my_system.load_mesh_from_csv('../meshes/selbstgebaut_quad/nodes.csv', 
                             '../meshes/selbstgebaut_quad/elements.csv')


#dirichlet_boundary conditions
#top_fixation_2 = [None, [master_node - 1 + 2*x for x in range(11)], None]
nodes_to_fix = np.where(my_system.node_list[:,0] == pos_x0)[0]
dofs_to_fix = np.concatenate((nodes_to_fix*2,nodes_to_fix*2+1),axis=1)
fixation_left = [None, dofs_to_fix, None]
dirichlet_boundary_list = [fixation_left]
my_system.apply_dirichlet_boundaries(dirichlet_boundary_list)    
       
M, K = amfe.give_mass_and_stiffness(my_system)  









# Copy lists for plotting
node_list = np.copy(my_system.node_list)
element_list = np.copy(my_system.element_list)



# Plot nodes
plt.figure(5)
print('Plotting nodes',end='')
plt.plot(node_list[:,0],node_list[:,1],'gx')
for i_nod in range(len(node_list)):
    plt.text(node_list[i_nod,0],node_list[i_nod,1],i_nod,
             verticalalignment='bottom',color='g')
plt.axis('equal')
#plt.axis([pos_x0-0.1, pos_x0+x_len+0.1, pos_y0-0.1, pos_y0+y_len+0.1])
plt.title('Nodes of the mesh')
print(' successful!')




# Create a list of elements that every row is one closed loop of nodes
ele_list_plot = np.concatenate((element_list,
                                element_list[:,0].reshape(len(element_list),1)),
                                axis=1)
                                
plt.figure(6)
print('Plotting elements',end='')
plot_no_of_ele = True
for i_ele in range(len(ele_list_plot)):
    nodes = ele_list_plot[i_ele,:]
    plt.plot(node_list[nodes,0], node_list[nodes,1],color='b')
    if plot_no_of_ele:
        coor = 0.5*node_list[nodes[0],:] + 0.5*node_list[nodes[2],:]
        plt.text(coor[0],coor[1],i_ele,color='b',horizontalalignment='center',
                 verticalalignment='center')
plt.axis('equal')
#plt.axis([pos_x0-0.1, pos_x0+x_len+0.1, pos_y0-0.1, pos_y0+y_len+0.1])
plt.title('Mesh')
print(' successful!')




#%% Compute eigenvalues
lam, phi = sp.sparse.linalg.eigsh(K,k=6,M=M,which='SM')

no_of_eigenm = 5
scale = -0.05

disp_glob = my_system.b_constraints * phi
pos = node_list.reshape((-1,1)) + disp_glob*scale 


t1 = time.time()
plt.figure(7)
print('Plotting mode {0} (= eigenmodes number {1}, including rigid body modes)'
      .format(no_of_eigenm,no_of_eigenm+1),end='')

for i_ele in range(len(ele_list_plot)):
    nodes = ele_list_plot[i_ele,:]
    dof_x = nodes*2
    dof_y = nodes*2+1
    plt.plot(pos[dof_x,no_of_eigenm], pos[dof_y,no_of_eigenm],color='r')
plt.axis('equal')    
#plt.axis([pos_x0-0.1, pos_x0+x_len+0.1, pos_y0-0.1, pos_y0+y_len+0.1])
plt.title('Mesh')
print(' successful!')

delta_t = time.time() - t1
print('Benötigte Zeit: {0}'.format(delta_t))




plt.show()


