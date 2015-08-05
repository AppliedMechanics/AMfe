# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 16:53:16 2015

@author: gruber
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# make amfe running
import sys
sys.path.insert(0, '..')
import amfe


gmsh_input_file = '../meshes/gmsh/2D_Quad4_3Parts.msh'
gmsh_input_file = '../meshes/gmsh/2D_Rectangle_partition2.msh'

my_system = amfe.MechanicalSystem()
#my_system.load_mesh_from_gmsh(gmsh_input_file)

