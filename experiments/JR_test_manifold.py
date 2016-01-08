# -*- coding: utf-8 -*-
# pylint: disable=trailing-whitespace, C0103, E1101, E0611
"""
Created on Thu Jan  7 16:38:15 2016

@author: rutzmoser
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

import amfe


no_of_modes = 20

gmsh_input_file = '../meshes/gmsh/bogen_grob.msh'
paraview_output_file = '../results/test' + \
                        time.strftime("_%Y%m%d_%H%M%S") + '/test'


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


omega, V = amfe.vibration_modes(my_system, n=no_of_modes)

my_reduced_system = amfe.reduce_mechanical_system(my_system, V)
#%%

# check, if matrices are (almost) diagonal

K = my_reduced_system.K()
#plt.matshow(K)
M = my_reduced_system.M()
#plt.matshow(M)

#%%

# time integration

my_newmark = amfe.NewmarkIntegrator()
my_newmark.set_mechanical_system(my_reduced_system)
my_newmark.delta_t = 1E-4

t1 = time.time()

my_newmark.integrate_nonlinear_system(np.zeros(no_of_modes), 
                                      np.zeros(no_of_modes), np.arange(0, 0.4, 1E-4))

t2 = time.time()
print('Time for computation:', t2 - t1, 'seconds.')

my_reduced_system.export_paraview(paraview_output_file)

t3 = time.time()
print('Time for export:', t3 - t2, 'seconds.')
