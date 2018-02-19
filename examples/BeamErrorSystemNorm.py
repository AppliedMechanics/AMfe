# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Cantilever beam loaded at tip.
"""


# load packages
import amfe
import scipy as sp
import numpy as np


# define input files
input_file1 = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad8.msh')
input_file2 = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad4.msh')


# define system 1
material1 = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system1 = amfe.MechanicalSystem()
system1.load_mesh_from_gmsh(input_file1, 1, material1)
system1.apply_dirichlet_boundaries(5, 'xy')
ndof1 = system1.dirichlet_class.no_of_constrained_dofs
system1.apply_neumann_boundaries(key=3, val=1.0, direct=(0, -1), time_func=lambda t: 1)
system1.apply_rayleigh_damping(1e0, 1e-5)
state_space_system1 = amfe.mechanical_system.convert_mechanical_system_to_state_space(system1,
                                                                                      regular_matrix=system1.K(),
                                                                                      overwrite=False)

# define system 2
material2 = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system2 = amfe.MechanicalSystem()
system2.load_mesh_from_gmsh(input_file2, 1, material2)
system2.apply_dirichlet_boundaries(5, 'xy')
ndof2 = system2.dirichlet_class.no_of_constrained_dofs
system2.apply_neumann_boundaries(key=3, val=1.0, direct=(0, -1), time_func=lambda t: 1)
system2.apply_rayleigh_damping(1e0, 1e-5)
state_space_system2 = amfe.mechanical_system.convert_mechanical_system_to_state_space(system2,
                                                                                      regular_matrix=system2.K(),
                                                                                      overwrite=False)

norm_error_system = amfe.linalg.lti_system_norm(A=sp.sparse.bmat([[state_space_system1.A(), None],
                                                                  [None, state_space_system2.A()]]),
                                                B=np.concatenate((state_space_system1.F_ext(x=np.zeros(2*ndof1), t=0.0),
                                                                  state_space_system2.F_ext(x=np.zeros(2*ndof2), t=0.0))),
                                                C=np.concatenate((+state_space_system1.F_ext(x=np.zeros(2*ndof1), t=0.0),
                                                                  -state_space_system2.F_ext(x=np.zeros(2*ndof2), t=0.0))),
                                                E=sp.sparse.bmat([[state_space_system1.E(), None],
                                                                  [None, state_space_system2.E()]]),
                                                ord=2, use_controllability_gramian=True)
print(norm_error_system)

