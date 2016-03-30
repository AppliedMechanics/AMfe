"""Clamped clamped beam arch example
"""

import amfe
import numpy as np
import time
import os

amfe_dir = amfe.__path__[0]
amfe_dir = os.path.dirname(amfe_dir) # move one folder up

rel_gmsh_input_file_dir = 'meshes/gmsh/beam_arc.msh'
rel_paraview_output_file = 'results/test' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/test'
gmsh_input_file = os.path.join(amfe_dir, rel_gmsh_input_file_dir)
paraview_output_file = os.path.join(amfe_dir, rel_paraview_output_file)

my_material = amfe.KirchhoffMaterial(E=3E9)
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 10, my_material)
# Test the paraview basic output 
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(5, 'xy')
my_system.apply_dirichlet_boundaries(6, 'xy')

neumann_domain = 7

benchmark_system = my_system

alpha = 0.1


def harmonic_y(t):
    return np.sin(2*np.pi*t*8.65) + np.sin(2*np.pi*t*28)

benchmark_system.apply_neumann_boundaries(key=neumann_domain, val=4E5,
                                          direct=(0,1),
                                          time_func=harmonic_y)


