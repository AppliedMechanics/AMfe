# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:44:13 2015

Test the mesh-module

@author: johannesr
"""


import numpy as np
import scipy as sp
import time

# make amfe running
import sys
sys.path.insert(0,'..')
import amfe


# test gmsh input-output functionality

gmsh_input_file = '../meshes/gmsh/2D_Rectangle_partition1.msh'
paraview_output_file = '../results/gmsh' + time.strftime("_%Y%m%d_%H%M%S") + '/gmsh_import'

my_mesh = amfe.Mesh()
my_mesh.import_msh(gmsh_input_file)

my_mesh.save_mesh_for_paraview(paraview_output_file)



# test mesh generator and mesh functionality
node_file = '../meshes/selbstgebaut/curved_mesh_nodes.csv'
element_file = '../meshes/selbstgebaut/curved_mesh_elements.csv'
my_mesh_creator = amfe.MeshGenerator(1, 5, 10, 50, 0.3, x_curve=True)
my_mesh_creator.build_mesh()
my_mesh_creator.save_mesh(node_file, element_file)

my_mesh.read_nodes_from_csv(node_file, node_dof=3)
my_mesh.read_elements_from_csv(element_file)

my_mesh.save_mesh_for_paraview('../results/selbstgebaut/selbstgebaut')

