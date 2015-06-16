# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:13:17 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import time

from matplotlib import pyplot as plt
# make amfe running
import sys
sys.path.insert(0,'../..')
import amfe




# für den Bogen
gmsh_input_file = '../../meshes/gmsh/bogen_grob.msh'
start_index = amfe.node2total(54, 0)
end_index = amfe.node2total(55, 1)

# fixation
bottom_bounds_1 = [None, [0,1,2,3], None]
bottom_bounds_2 = [None, np.arange(start_index, end_index + 1), None]


# Default values;
kwargs = {'E_modul' : 210E9, 'poisson_ratio' : 0.3, 'element_thickness' : 1, 'density' : 1E4}
element_class_dict = {'Tri3' : amfe.Tri3(**kwargs), 'Tri6' : amfe.Tri6(**kwargs)}


my_system = amfe.MechanicalSystem()
my_system.element_class_dict = element_class_dict
my_system.load_mesh_from_gmsh(gmsh_input_file)
# my_system.export_paraview(paraview_output_file)

nodes_to_fix = my_system.mesh_class.boundary_line_list[4]
bottom_bounds_1 = [None, [amfe.node2total(i, 0) for i in nodes_to_fix], None]
bottom_bounds_2 = [None, [amfe.node2total(i, 1) for i in nodes_to_fix], None]
my_dirichlet_bounds = [bottom_bounds_1, bottom_bounds_2]
my_system.apply_dirichlet_boundaries(my_dirichlet_bounds)


top_bounds= my_system.mesh_class.boundary_line_list[1]

neumann_bounds = [  [[amfe.node2total(i,0) for i in top_bounds], 'harmonic', (6E6, 3), None],
                    [[amfe.node2total(i,1) for i in top_bounds], 'harmonic', (2E6, 6), None]]
my_system.apply_neumann_boundaries(neumann_bounds)


ndof = my_system.ndof_global_constrained

