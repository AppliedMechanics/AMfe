#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:46:10 2015

@author: fabian
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# make amfe running
import sys
sys.path.insert(0, '..')
import amfe


#%% Generate mesh for substructures
'''Generate global mesh with respect to global node ID's'''
x_len, y_len = 1, 1
x_no_sub, y_no_sub = 4, 3
x_no_ele_p_sub, y_no_ele_p_sub = 4, 3

pos_x0, pos_y0 = 0, 0

x_no_ele = x_no_sub*x_no_ele_p_sub
y_no_ele = y_no_sub*y_no_ele_p_sub
no_ele = x_no_ele*y_no_ele

x_no_nod = x_no_ele+1
y_no_nod = y_no_ele+1
no_nod = x_no_nod*y_no_nod

no_sub_total = x_no_sub*y_no_sub

x_delta = x_len/x_no_ele
y_delta = y_len/y_no_ele

# Create node table
nodes = []
for j_nod in range(y_no_ele+1):
    for i_nod in range(x_no_ele+1):
        nodes.append([pos_x0+x_delta*i_nod,pos_y0+y_delta*j_nod])

# Create element table
elements = [0]*no_ele
ele_type = 3
for j_sub in range(1,y_no_sub+1):
    for i_sub in range(1,x_no_sub+1):
        # Compute the number of the current subdomain (starting with 1)
        no_of_curr_sub = i_sub + x_no_sub*(j_sub-1)
        # Compute the number of the node in south-east corner of subdomain
        start_node_sub = (i_sub-1)*x_no_ele_p_sub \
                        +(j_sub-1)*(y_no_ele_p_sub)*(x_no_ele+1)               
        
        for j_ele in range(0,y_no_ele_p_sub):
            for i_ele in range(0,x_no_ele_p_sub):
                # Compute the global element number (starting with 0)
                dom_ele_num = i_ele + (j_ele-1+1)*x_no_ele \
                            + (i_sub-1)*x_no_ele_p_sub \
                            + (j_sub-1)*y_no_ele_p_sub*x_no_ele
                
                # Default values                     
                no_tags = 4
                no_of_part = 1                       
                
                # Compute node number corresponding to the elements
                node1 = i_ele   + j_ele*x_no_nod    +start_node_sub
                node2 = i_ele+1 + j_ele*x_no_nod    +start_node_sub
                node3 = i_ele+1 +(j_ele+1)*x_no_nod +start_node_sub
                node4 = i_ele   +(j_ele+1)*x_no_nod +start_node_sub               

                elements[dom_ele_num] = [ele_type, no_tags, 0, 0, \
                        no_of_part, no_of_curr_sub, node1, node2, node3, node4]

