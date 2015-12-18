# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 13:18:08 2015

@author: johannesr
"""

import numpy
import scipy
import amfe



input_file = '../meshes/gmsh/pressure_corner.msh'
output_file = '../results/pressure_corner/pressure_corner'


my_material = amfe.KirchhoffMaterial()
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(input_file, 11, my_material)
my_system.apply_dirichlet_boundaries(9, 'x')
my_system.apply_dirichlet_boundaries(10, 'y')
my_system.apply_neumann_boundaries(12, 1E10, 'normal', lambda t: t)


#amfe.solve_linear_displacement(my_system)
amfe.solve_nonlinear_displacement(my_system, no_of_load_steps=50)

my_system.export_paraview(output_file)

