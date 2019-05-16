# -*- coding: utf-8 -*-
# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Created on Fri Dec 18 15:31:45 2015

@author: johannesr
"""


import numpy as np
import scipy as sp
import amfe


input_file = amfe.amfe_dir('meshes/gmsh/AMFE_logo.msh')
output_file = amfe.amfe_dir('results/AMFE_logo/logo_5')


material_1 = amfe.KirchhoffMaterial(E=5E6, rho=1E4)
material_2 = amfe.KirchhoffMaterial(E=5E7, rho=1E4)
my_system = amfe.MechanicalSystem()


my_system.load_mesh_from_gmsh(input_file, 299, material_1)
my_system.load_mesh_from_gmsh(input_file, 300, material_1)
my_system.load_mesh_from_gmsh(input_file, 301, material_2)
my_system.load_mesh_from_gmsh(input_file, 302, material_2)


my_system.apply_dirichlet_boundaries(298, 'xyz')
M_unconstr = my_system.assembly_class.assemble_m()
no_of_nodes = my_system.mesh_class.no_of_nodes
#g = [(0, -9.81, 0) for i in range(no_of_nodes)]
g = [(0, -9.81, 0) for i in range(no_of_nodes)]
g = np.array(g).flatten()
f_gravity = M_unconstr @ g

def f_ext_unconstr(u=None, t=None):
#    if t > 0.5:
#        return f_gravity*0
#    else:
        return f_gravity*np.sin(t*2*np.pi*4.2)

my_system._f_ext_unconstr = f_ext_unconstr

om, V = amfe.vibration_modes(my_system, 40)

print('The vibration modes eigenfrequencies (1/s) are:\n', om/(2*np.pi))

#amfe.solve_linear_displacement(my_system)
#amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=50)

# time integration
dt = 1E-2
T = np.arange(0,4,dt)
ndof = my_system.dirichlet_class.no_of_constrained_dofs
q0 = dq0 = np.zeros(ndof)
solver = amfe.NonlinearDynamicsSolver(my_system, initial_conditions={'q0': q0, 'dq0': dq0}, verbose=True, t_end=4, dt=dt)
solver.solve()
# old api: amfe.integrate_nonlinear_system(my_system, q0, dq0, T, dt, alpha=0.01)


my_system.export_paraview(output_file)

