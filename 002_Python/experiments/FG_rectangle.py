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
sys.path.insert(0,'..')
import amfe


# Mesh generation
my_mesh_generator = amfe.MeshGenerator(x_len=10, y_len=10, x_no_elements=3, 
                                       y_no_elements=3, mesh_style = 'Quad')
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('../meshes/selbstgebaut_quad/nodes.csv', 
                            '../meshes/selbstgebaut_quad/elements.csv')



# Building the mechanical system
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_csv('../meshes/selbstgebaut_quad/nodes.csv', 
                             '../meshes/selbstgebaut_quad/elements.csv' )
