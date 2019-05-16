# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
"""


import numpy as np
import scipy as sp
import time

import amfe



gmsh_input_file = amfe.amfe_dir('meshes/gmsh/c_bow_coarse.msh')
paraview_output_file = amfe.amfe_dir('results/c_bow_coarse/c_bow_coarse')


my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()

my_system.load_mesh_from_gmsh(gmsh_input_file, 15, my_material)
# Test the paraview basic output
# my_system.export_paraview(paraview_output_file)

my_system.apply_dirichlet_boundaries(13, 'xy')

harmonic_x = lambda t: np.sin(2*np.pi*t*30)
harmonic_y = lambda t: np.sin(2*np.pi*t*50)

my_system.apply_neumann_boundaries(14, 1E8, (1,0), harmonic_x)
my_system.apply_neumann_boundaries(14, 1E8, (0,1), harmonic_y)


###############################################################################
## time integration
###############################################################################

ndof = my_system.dirichlet_class.no_of_constrained_dofs
q0 = np.zeros(ndof)
dq0 = np.zeros(ndof)
initial_conditions = {'q0': q0, 'dq0': dq0}

dt = 5e-4
t_end = 2

#%%
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_system,dt=dt, t_end=t_end, initial_conditions=initial_conditions)
solver.solve()
my_system.export_paraview(paraview_output_file + '_full_model')

my_system.clear_timesteps()
solver = amfe.GeneralizedAlphaLinearDynamicsSolver(my_system, dt=dt, t_end=t_end, initial_conditions=initial_conditions)
solver.solve()
my_system.export_paraview(paraview_output_file + '_linear_model')

my_system.clear_timesteps()
omega, V = amfe.vibration_modes(my_system, 7, save=True)
my_system.export_paraview(paraview_output_file + '_modes')

Theta, Theta_tilde = amfe.shifted_modal_derivatives(V, my_system.K, my_system.M(), omega)
my_system.clear_timesteps()
V_temp = amfe.augment_with_derivatives(None, Theta, deflate=False)
for i in np.arange(V_temp.shape[1]):
    my_system.write_timestep(i, V_temp[:,i])
my_system.export_paraview(paraview_output_file + '_theta_shifted')

my_system.clear_timesteps()
V_temp = amfe.augment_with_derivatives(None, Theta_tilde, deflate=False)
for i in np.arange(V_temp.shape[1]):
    my_system.write_timestep(i, V_temp[:,i])
my_system.export_paraview(paraview_output_file + '_theta_shifted_tilde')

static_derivatives = amfe.static_derivatives(V, my_system.K,my_system.M())
V_temp = amfe.augment_with_derivatives(None, static_derivatives, deflate=False)
for i in np.arange(V_temp.shape[1]):
    my_system.write_timestep(i, V_temp[:,i])
my_system.export_paraview(paraview_output_file + '_static_derivatives')



V_extended = amfe.augment_with_derivatives(V, Theta)
V_extended = amfe.augment_with_derivatives(V_extended, Theta_tilde)
my_system.clear_timesteps()
for i in np.arange(V_extended.shape[1]):
    my_system.write_timestep(i, V_extended[:,i])

my_system.export_paraview(paraview_output_file + '_basis_theta_theta_tilde_deflated')

V_extended_sd = amfe.augment_with_derivatives(V, static_derivatives)
my_system.clear_timesteps()
for i in np.arange(V_extended_sd.shape[1]):
    my_system.write_timestep(i, V_extended_sd[:,i])
my_system.export_paraview(paraview_output_file + '_static_derivatives_deflated')


system_red_sd = amfe.reduce_mechanical_system(my_system, V_extended_sd[:,0:20])
system_red_theta =amfe.reduce_mechanical_system(my_system, V_extended[:,0:20])


q0_r = np.zeros(20)
dq0_r = np.zeros(20)
initial_conditions = {'q0': q0_r, 'dq0': dq0_r}
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(system_red_sd,dt=dt, t_end=t_end, initial_conditions=initial_conditions)
solver.solve()
system_red_sd.export_paraview(paraview_output_file + '_red_sd_20')

solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(system_red_theta,dt=dt, t_end=t_end, initial_conditions=initial_conditions)
solver.solve()
system_red_theta.export_paraview(paraview_output_file + '_red_theta_20')