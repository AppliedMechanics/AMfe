# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:05:50 2015

@author: johannesr
"""


def node2total(node_index, coordinate_index, ndof_node=2):
    '''
    Converts the node index and the corresponding coordinate index to the index
    of the total dof.

    Parameters
    ----------
    node_index : int
        Index of the node as shown in tools like paraview
    coordinate_index: int
        Index of the coordinate; 0 if it's x, 1 if it's y etc.
    ndof_node: int, optional
        Number of degrees of freedom per node

    Returns
    -------
    total_index : int
        Index of the total dof

    '''
    if coordinate_index >= ndof_node:
        raise ValueError('coordinate index is greater than dof per node.')
    return node_index*ndof_node + coordinate_index

def total2node(total_index, ndof_node=2):
    '''
    Converts the total index in the global dofs to the coordinate index and the
    index fo the coordinate.

    Parameters
    ----------
    total_index : int
        Index of the total dof
    ndof_node: int, optional
        Number of degrees of freedom per node

    Returns
    -------
    node_index : int
        Index of the node as shown in tools like paraview
    coordinate_index : int
        Index of the coordinate; 0 if it's x, 1 if it's y etc.

    '''
    return total_index // ndof_node, total_index % ndof_node

