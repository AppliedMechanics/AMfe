# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 16:24:11 2016

@author: rutzmoser
"""

import amfe
import numpy as np
import time
import os

amfe_dir = amfe.__path__[0]
amfe_dir = os.path.dirname(amfe_dir) # move one folder up

gmsh_input_file_dir_relative = 'meshes/gmsh/fancy_arc.msh'
paraview_output_file_relative = 'results/test' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/test'
gmsh_input_file = os.path.join(amfe_dir, gmsh_input_file_dir_relative)
paraview_output_file = os.path.join(amfe_dir, paraview_output_file_relative)

my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 24, my_material)
# Test the paraview basic output 
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(19, 'xy')
my_system.apply_dirichlet_boundaries(20, 'xy')

def harmonic_ext(t):
    return np.sin(2*np.pi*t*10)

my_system.apply_neumann_boundaries(21, 6E7, 'normal', harmonic_ext)

benchmark_system = my_system

# Time integration parameters
alpha = 0.1