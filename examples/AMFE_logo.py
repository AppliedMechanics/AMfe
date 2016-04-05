# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:31:45 2015

@author: johannesr
"""


import numpy as np
import scipy as sp
import amfe


input_file = '../meshes/gmsh/AMFE_logo.msh'
output_file = '../results/AMFE_logo/logo_5'


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
#
ndof = my_system.dirichlet_class.no_of_constrained_dofs
my_integrator = amfe.NewmarkIntegrator(my_system, alpha=0.01)
my_integrator.delta_t = 0.5E-3
my_integrator.verbose = True
my_integrator.integrate(np.zeros(ndof), np.zeros(ndof), 
                                         np.arange(0,4,1E-2))


my_system.export_paraview(output_file)

