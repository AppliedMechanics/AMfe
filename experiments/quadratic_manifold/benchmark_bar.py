# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:40:10 2016

@author: rutzmoser
"""

import amfe
import numpy as np
import time
import os

amfe_dir = amfe.__path__[0]
amfe_dir = os.path.dirname(amfe_dir) # move one folder up

rel_gmsh_input_file_dir = 'meshes/gmsh/bar.msh'
rel_paraview_output_file = 'results/test' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/test'
gmsh_input_file = os.path.join(amfe_dir, rel_gmsh_input_file_dir)
paraview_output_file = os.path.join(amfe_dir, rel_paraview_output_file)

my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 7, my_material)
# Test the paraview basic output 
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(8, 'xy')

harmonic_y = lambda t: np.sin(2*np.pi*t*20)

my_system.apply_neumann_boundaries(9, 2E7, 'y', harmonic_y)

benchmark_system = my_system

alpha = 0.1
