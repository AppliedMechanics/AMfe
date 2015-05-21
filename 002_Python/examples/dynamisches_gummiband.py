# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:42:30 2015

@author: johannesr
"""


import sys
import time

import numpy as np
import scipy as sp


# this is the way how to import the amfe-toolbox
sys.path.insert(0,'..')
import amfe



# Mesh generation
my_mesh_generator = amfe.MeshGenerator(x_len=1., y_len=1, x_no_elements=10, y_no_elements=10)
my_mesh_generator.build_mesh()
my_mesh_generator.save_mesh('../meshes/selbstgebaut/nodes.csv', '../meshes/selbstgebaut/elements.csv')

# Building the mechanical system
my_mechanical_system = amfe.MechanicalSystem()
my_mechanical_system.load_mesh_from_csv('../meshes/selbstgebaut/nodes.csv', '../meshes/selbstgebaut/elements.csv' )

# Building the Element and handing it over to the mechanical system
my_element = amfe.ElementPlanar(E_modul=210E9, poisson_ratio=0.3)
my_mechanical_system.set_element(my_element)


# Boundary handling
bottom_fixation = [None, range(22), None]
#bottom_fixation = [None, [1 + 2*x for x in range(10)], None]
#bottom_fixation2 = [None, [0, ], None]
master_node = amfe.node2total(110, 1, ndof_node=2)
top_fixation = [master_node, [master_node + 2*x for x in range(11)], None]
top_fixation_2 = [None, [master_node - 1 + 2*x for x in range(11)], None]
dirichlet_boundary_list = [bottom_fixation, top_fixation, top_fixation_2]

# my_dirichlet_boundary_list = [[None, np.arange(40), None], [200, [200 + 2*i for i in range(40)], None]]
# Vielleicht ist die Kraft a bisserl zu stark, so dass das Ding überklappt bzw. die Schrittweite zu groß ist
my_neumann_boundary_list = [[[master_node,], 'harmonic', (2E12, 10), None]]
my_mechanical_system.apply_dirichlet_boundaries(dirichlet_boundary_list)
my_mechanical_system.apply_neumann_boundaries(my_neumann_boundary_list)

ndof = my_mechanical_system.f_ext_global(None, None, 0).shape[0]

my_integrator = amfe.NewmarkIntegrator(verbose=True)
my_integrator.set_mechanical_system(my_mechanical_system)
my_integrator.integrate_nonlinear_system(np.zeros(ndof), np.zeros(ndof), np.arange(0,0.1,0.01))


export_path = '../results/dyn_gummistab' + time.strftime("_%Y%m%d_%H%M%S") + '/gummi_dehnstab'
my_mechanical_system.export_paraview(export_path)

