# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:44:04 2015

@author: johannesr
"""


import numpy as np
import scipy as sp

import mesh
import element
import assembly
import boundary
import integrator
import dynamical_system


my_mesh_generator = mesh.MeshGenerator(x_len=1., y_len=10, x_no_elements=10, y_no_elements=100)
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('Vernetzungen/nodes.csv', 'Vernetzungen/elements.csv')

my_dynamical_system = dynamical_system.DynamicalSystem()
my_dynamical_system.load_mesh_from_csv('Vernetzungen/nodes.csv', 'Vernetzungen/elements.csv' )
my_dynamical_system.export_paraview('Versuche/dynamisches_system_4')

my_element = element.ElementPlanar(E_modul=210E9)
my_dynamical_system.set_element(my_element)

bottom_fixation = [None, range(20), None]
#bottom_fixation = [None, [1 + 2*x for x in range(10)], None]
#bottom_fixation2 = [None, [0, ], None]
conv = assembly.ConvertIndices(2)
master_node = conv.node2total(1100, 1)
top_fixation = [master_node, [master_node + 2*x for x in range(10)], None]
dirichlet_boundary_list = [bottom_fixation, top_fixation]


# Test der assembly-funktion:

# my_dirichlet_boundary_list = [[None, np.arange(40), None], [200, [200 + 2*i for i in range(40)], None]]
my_neumann_boundary_list = [[[master_node,], 'ramp', (8E10, 0.0), None]]
my_dynamical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)
my_dynamical_system.apply_neumann_boundaries(my_neumann_boundary_list)

ndof_bc = my_dynamical_system.f_ext_global(None, None, 0).shape[0]
# Test
# my_dynamical_system.neumann_bc_class.function_list[0](3)

## Zuerst das statische Problem:
#u_bc = sp.sparse.linalg.spsolve(my_dynamical_system.K_global(), my_dynamical_system.f_ext_global(None, None, 0))
#u_global = my_dynamical_system.b_constraints.dot(u_bc)
#my_dynamical_system.write_timestep(1, u_bc)
#my_dynamical_system.export_paraview('Versuche/dynamisches_system')

# Dann das dynamische Problem:

my_integrator = integrator.NewmarkIntegrator(alpha=0)
#my_integrator.delta_t = 5E-3
my_integrator.set_dynamical_system(my_dynamical_system)
# my_integrator.f_ext = None
my_integrator.verbose = True
my_integrator.integrate_nonlinear_system(np.zeros(ndof_bc), np.zeros(ndof_bc),  np.arange(0,0.05,0.01))
my_dynamical_system.export_paraview('Versuche/dynamisches_system_4')

#my_dynamical_system.export_paraview('ParaView/linear_static_testcase')
#my_dynamical_system.export_paraview('Para_View/nonlinear_dynamic_testcase')

