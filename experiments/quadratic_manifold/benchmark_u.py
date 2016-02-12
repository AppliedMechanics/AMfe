# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 17:58:48 2016

@author: rutzmoser
"""
import amfe
import numpy as np
import time
import os

amfe_dir = amfe.__path__[0]
amfe_dir = os.path.dirname(amfe_dir) # move one folder up

gmsh_input_file_dir_relative = 'meshes/gmsh/bogen_grob.msh'
paraview_output_file_relative = 'results/test' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/test'
gmsh_input_file = os.path.join(amfe_dir, gmsh_input_file_dir_relative)
paraview_output_file = os.path.join(amfe_dir, paraview_output_file_relative)

my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 15, my_material)
# Test the paraview basic output 
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(13, 'xy')

harmonic_x = lambda t: np.sin(2*np.pi*t*30)
harmonic_y = lambda t: np.sin(2*np.pi*t*50)

my_system.apply_neumann_boundaries(14, 6E7, 'x', harmonic_x)
my_system.apply_neumann_boundaries(14, 6E7, 'y', harmonic_y)

benchmark_system = my_system

# Time integration parameters
alpha = 0.1