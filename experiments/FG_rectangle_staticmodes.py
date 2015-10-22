#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 14:53:18 2015

@author: gruber
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# make amfe running
import sys
sys.path.insert(0, '..')
import amfe


#%% Mesh generation
x_len, y_len, x_no_elements, y_no_elements = 1.5, 0.5, 12, 4
pos_x0, pos_y0 = 0, 0
my_mesh_generator = amfe.MeshGenerator(x_len=x_len, y_len=y_len,
                                       x_no_elements=x_no_elements,
                                       y_no_elements=y_no_elements,
                                       pos_x0=pos_x0, pos_y0=pos_y0,
                                       mesh_style='Quad4')
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('./meshes/selbstgebaut_quad/nodes.csv',
                            './meshes/selbstgebaut_quad/elements.csv')


#%% Building the mechanical system

# Initialize system
my_system = amfe.MechanicalSystem()
my_system = amfe.MechanicalSystem(E_modul=1.0, poisson_ratio=0.0, 
                                  element_thickness=0.1, density=1.0)



# Load mesh
my_system.load_mesh_from_csv('./meshes/selbstgebaut_quad/nodes.csv',
                             './meshes/selbstgebaut_quad/elements.csv')


#dirichlet_boundary conditions
#top_fixation_2 = [None, [master_node - 1 + 2*x for x in range(11)], None]
nodes_to_fix = np.where(my_system.node_list[:, 0] == pos_x0)[0]
dofs_to_fix = np.concatenate((nodes_to_fix*2, nodes_to_fix*2+1), axis=1)
fixation_left = [None, dofs_to_fix, None]
dirichlet_boundary_list = [fixation_left]
my_system.apply_dirichlet_boundaries(dirichlet_boundary_list)

# Boundary degrees of freedom (on right side), where structure may be fixed to 
# another neighbouring structure
nodes_boundary = np.where(my_system.node_list[:, 0] == pos_x0+x_len)[0]
dofs_boundary = np.concatenate((nodes_boundary*2, nodes_boundary*2+1), axis=1)

# Build mass and stiffness matrix
M, K = amfe.give_mass_and_stiffness(my_system)


#%% Plot mesh
# Copy lists for plotting
node_list = np.copy(my_system.node_list)
element_list = np.copy(my_system.element_list)
pos_of_nodes = node_list.reshape((-1, 1))


def plot_nodes_Quad4(coord, p_col='b', no_of_fig=1,
                     p_title='Nodes of the mesh'):
    print('Plotting nodes...', end='')
    plt.figure(no_of_fig)
    plt.plot(coord[0::2], coord[1::2], color=p_col, marker='o', linestyle='')
    for i_nod in range(int(len(coord)/2)):
        plt.text(coord[2*i_nod], coord[2*i_nod+1], i_nod,
                 verticalalignment='bottom', color=p_col)
    plt.axis('equal')
    #plt.axis([pos_x0-0.1, pos_x0+x_len+0.1, pos_y0-0.1, pos_y0+y_len+0.1])
    plt.title('Nodes of the mesh')
    plt.hold(True)
    print(' successful!')


def plot_mesh_Quad4(elements, coord, plot_no_of_ele=False, plot_nodes=False,
                    p_col='r', no_of_fig=1, p_title='Mesh',
                    p_col_node = 'b'):

    '''
    Plot mesh of plane Quad4 elements

    Parameters
    ----------
    elements: array
        array containing the node number of each element
    coord: array
        array containing the coordinates of each node
    plot_no_of_ele: bool, optional
        flag stating if numbers of elements are plotted
    plot_nodes: bool, optional
        flag stating if nodes are plotted
    p_col: str, optional
        color of plot

    Returns
    -------
    None
    '''
    print('Plotting elements...', end='')
    plt.figure(no_of_fig)

    # Duplicate first node of each element to get closed circle
    ele_list_plot = np.concatenate((element_list,
                                    element_list[:, 0]
                                    .reshape(len(element_list), 1)), axis=1)

    # Loop over elements
    for i_ele in range(len(elements)):

        nodes = ele_list_plot[i_ele, :]  # Node number of one element
        plt.plot(coord[nodes*2], coord[nodes*2+1], color=p_col)  # Plot ele

        # Plot number of element in each element if plot_no_of_ele == True
        if plot_no_of_ele:
            dof1 = nodes[0]*2+np.array([0, 1])
            dof2 = nodes[2]*2+np.array([0, 1])
            pos_of_text = 0.5*coord[dof1] + 0.5*coord[dof2]
            plt.text(pos_of_text[0], pos_of_text[1], i_ele, color=p_col,
                     horizontalalignment='center', verticalalignment='center')

    # Plot the nodes of the mesh if plot_nodes == True
    if plot_nodes:
        plot_nodes_Quad4(coord, no_of_fig=no_of_fig, p_title=p_title,
                         p_col = p_col_node)

    # Further properties of plot
    plt.axis('equal')
    plt.title(p_title)
    plt.hold(True)
    print(' successful!')


# Plot mesh and nodes of Quad4-elements
plot_nodes_Quad4(pos_of_nodes, p_col='b', no_of_fig=1)
plot_mesh_Quad4(element_list, pos_of_nodes, plot_no_of_ele=True, p_col='g',
                no_of_fig=2, p_title='Reine Vernetzung')


control_eigenvalue = 0


#%% Compute static modes
f = sp.sparse.csr_matrix((my_system.b_constraints.shape[0],dofs_boundary.shape[0]))
f[dofs_boundary,:] = sp.sparse.eye(dofs_boundary.shape[0])
f_c = my_system.b_constraints.T*f
disp = sp.sparse.linalg.spsolve(K, f_c)

# Extend computed eigenmodes to all dofs (since some dofs are fixed)
# each eigenmodes is a 1D-array
disp_glob = my_system.b_constraints * disp
# Add eigenmode displacement (multiplied by scaling factor) to positions node
scale = -0.003
pos = pos_of_nodes + disp_glob*scale 


# Plot undeformed mesh in grey
plot_mesh_Quad4(element_list, pos_of_nodes, p_col='0.5',
                plot_nodes=True,no_of_fig=7, p_col_node = '0.5')

# Plot deformed mesh of Quad4-elements
no_of_eigenm =2    # = true eigenmodes number is no_of_eigenmode + 1!
p_title = 'Static Mode number {0} '.format(no_of_eigenm+1)
plot_mesh_Quad4(element_list, pos[:, no_of_eigenm], plot_no_of_ele=False,
                plot_nodes=True, p_col='r', no_of_fig=7, p_title=p_title)





plt.show()

#%% How to save file for further processing with Tikz
#np.savetxt('data_elements.dat', element_list, fmt='%i')
#np.savetxt('data_coordinates.dat', pos[:, no_of_eigenm].reshape((-1,2)), fmt='%f')
