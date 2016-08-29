# -*- coding: utf-8 -*-
"""
"""


import numpy as np
import scipy as sp
import time

import amfe



gmsh_input_file = amfe.amfe_dir('meshes/gmsh/c_bow_coarse.msh')
paraview_output_file = amfe.amfe_dir('results/c_bow_coarse' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/bogen_grob')


my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 15, my_material)
# Test the paraview basic output 
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(13, 'xy')

harmonic_x = lambda t: np.sin(2*np.pi*t*30)
harmonic_y = lambda t: np.sin(2*np.pi*t*50)

my_system.apply_neumann_boundaries(14, 6E7, (1,0), harmonic_x)
my_system.apply_neumann_boundaries(14, 6E7, (0,1), harmonic_y)


###############################################################################
## time integration
###############################################################################

ndof = my_system.dirichlet_class.no_of_constrained_dofs

my_newmark = amfe.NewmarkIntegrator(my_system)
my_newmark.delta_t = 2E-4

my_newmark.integrate(np.zeros(ndof), np.zeros(ndof), np.arange(0,0.1,1E-3))
my_system.export_paraview(paraview_output_file)
