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


my_mesh_generator = mesh.MeshGenerator(x_len=1., y_len=1, x_no_elements=10, y_no_elements=10)
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('Vernetzungen/nodes.csv', 'Vernetzungen/elements.csv')

my_dynamical_system = dynamical_system.DynamicalSystem()
my_dynamical_system.load_mesh_from_csv('Vernetzungen/nodes.csv', 'Vernetzungen/elements.csv' )
my_dynamical_system.export_paraview('Versuche/dynamisches_system_5')

my_element = element.ElementPlanar(E_modul=210E9, poisson_ratio=0.3)
my_dynamical_system.set_element(my_element)

bottom_fixation = [None, range(22), None]
#bottom_fixation = [None, [1 + 2*x for x in range(10)], None]
#bottom_fixation2 = [None, [0, ], None]
conv = assembly.ConvertIndices(2)
master_node = conv.node2total(110, 1)
top_fixation = [master_node, [master_node + 2*x for x in range(11)], None]
top_fixation_2 = [None, [master_node - 1 + 2*x for x in range(11)], None]
dirichlet_boundary_list = [bottom_fixation, top_fixation, top_fixation_2]

# Test der assembly-funktion:

# my_dirichlet_boundary_list = [[None, np.arange(40), None], [200, [200 + 2*i for i in range(40)], None]]
my_neumann_boundary_list = [[[master_node,], 'ramp', (2E12, 0.0), None]]
my_dynamical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)
my_dynamical_system.apply_neumann_boundaries(my_neumann_boundary_list)


############################
# Static solution routine
#############################

from scipy.sparse import linalg


ndof_bc = my_dynamical_system.f_ext_global(None, None, 0).shape[0]

sys = my_dynamical_system

no_of_steps = 40
newton_damping = 1
n_max_iter = 1000
n_iter_goal = 20
eps = 1E-12
simplified_newton_factor = 10

stepwidth = 1/no_of_steps
u = np.zeros(ndof_bc)
f_ext = sys.f_ext_global(None, None, 1)
abs_f_ext = np.sqrt(f_ext.dot(f_ext))

# Newton-Loop:

for forcing_factor in np.arange(stepwidth, 1+stepwidth, stepwidth):
    # prediction
    res = sys.f_int_global(u) - f_ext*forcing_factor
    abs_res = np.sqrt(res.dot(res))
    n_iter = 0
    while (abs_res > eps*abs_f_ext) and (n_max_iter > n_iter):
        if n_iter%simplified_newton_factor is 0:
            K = sys.K_global(u)
        corr = linalg.spsolve(K, res)
        u -= corr*newton_damping
        res = sys.f_int_global(u) - f_ext*forcing_factor
        abs_res = np.sqrt(res.dot(res))
        n_iter += 1
        print('Stufe', forcing_factor, 'Iteration Nr.', n_iter, 'Residuum:', abs_res)

    sys.write_timestep(forcing_factor, u)

sys.export_paraview('Versuche/statisches_system5')



# Test
# my_dynamical_system.neumann_bc_class.function_list[0](3)

## Zuerst das statische Problem:
#u_bc = sp.sparse.linalg.spsolve(my_dynamical_system.K_global(), my_dynamical_system.f_ext_global(None, None, 1))
#u_global = my_dynamical_system.b_constraints.dot(u_bc)
#my_dynamical_system.write_timestep(2, u_bc)
#my_dynamical_system.export_paraview('Versuche/dynamisches_system')

# Dann das dynamische Problem:

#my_integrator = integrator.NewmarkIntegrator(alpha=0)
#my_integrator.delta_t = 1E-2
#my_integrator.set_dynamical_system(my_dynamical_system)
## my_integrator.f_ext = None
#my_integrator.verbose = True
#my_integrator.integrate_nonlinear_system(np.zeros(ndof_bc), np.zeros(ndof_bc),  np.arange(0,0.5,0.01))
#my_dynamical_system.export_paraview('Versuche/dynamisches_system_4')
#
#my_dynamical_system.export_paraview('ParaView/linear_static_testcase')
#my_dynamical_system.export_paraview('Para_View/nonlinear_dynamic_testcase')

