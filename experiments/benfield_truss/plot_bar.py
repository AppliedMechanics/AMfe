# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:49:08 2015

@author: fabian
"""
import numpy as np
import matplotlib.pyplot as plt



def plt_mesh(elements, coord, plot_no_of_ele=False, plot_nodes=False,
                    p_col='b', no_of_fig=1, p_title='Mesh',
                    p_col_node='r'):

    '''
    Plot mesh of plane Quad4 elements

    Parameters
    ----------
    elements: array
        array containing the node number of each element
    coord: array
        1D-array containing the coordinates of each node
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
    print('Plotting elements in figure number %d ... '%no_of_fig, end='')
    plt.figure(no_of_fig)

    # Loop over elements
    for i_ele in range(elements.shape[0]):
    #for i_ele in range(1,2):
        ele_nodes = elements[i_ele,:]
        plt.plot(coord[ele_nodes*2],coord[ele_nodes*2+1], color=p_col)  # Plot ele

        # Plot number of element in each element if plot_no_of_ele == True
        if plot_no_of_ele:
            dof1 = ele_nodes[0]*2+np.array([0, 1])
            dof2 = ele_nodes[1]*2+np.array([0, 1])
            pos_of_text = 0.5*coord[dof1] + 0.5*coord[dof2]
            plt.text(pos_of_text[0], pos_of_text[1], i_ele, color=p_col,
                     horizontalalignment='center', verticalalignment='center')

    # Plot the nodes of the mesh if plot_nodes == True
    if plot_nodes:
        plt_nodes(coord, no_of_fig=no_of_fig, p_title=p_title,
                  p_col=p_col_node)

    # Further properties of plot
    plt.axis('equal')
    plt.title(p_title)
    plt.hold(True)
    print('Plot of elements in figure number %d successful!'%no_of_fig)


    
    
def plt_nodes(coord, p_col='b', no_of_fig=1,
                     p_title='Nodes of the mesh'):
    print('Plotting nodes in figure number %d ... '%no_of_fig, end='')
    plt.figure(no_of_fig)
    plt.plot(coord[0::2], coord[1::2], color=p_col, marker='o', linestyle='')
    for i_nod in range(int(len(coord)/2)):
        plt.text(coord[2*i_nod], coord[2*i_nod+1], i_nod,
                 verticalalignment='bottom', color=p_col)
    plt.axis('equal')
    plt.title('Nodes of the mesh')
    plt.hold(True)
    print('Plot of nodes in figure number %d successful!'%no_of_fig)    
    