# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Comparison of POD bases for cantilever beam loaded at tip.
"""


# load packages
import amfe
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


# define in- and output files
input_file = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad8.msh')
output_file = amfe.amfe_dir('results/beam/Beam10x1Quad8_nonlinear_dynamics_generalizedalpha')


# define system
material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(5, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
# system.apply_rayleigh_damping(1e0, 1e-6)


# define simulation parameters
options = {
    't0': 0.0,
    't_end': 0.4,
    'dt': 1.0e-3,
    'output_frequency': 1,
    'rho_inf': 0.95,
    'initial_conditions': {
        'q0': np.zeros(ndof),
        'dq0': np.zeros(ndof)},
    'relative_tolerance': 1.0E-6,
    'absolute_tolerance': 1.0E-9,
    'verbose': True,
    'max_number_of_iterations': 30,
    'convergence_abort': True,
    'write_iterations': False,
    'track_iterations': False,
    'save_solution': True}


# calculate POD basis V1
system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1.0)
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
solver.solve()
__, V1 = amfe.pod(mechanical_system=system, n=53)
system.export_paraview(output_file + '_forV1')
system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: -1.0)  # reset Neumann BCs


# calculate POD basis V2
system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(1, 0), time_func=lambda t: 1.0)
solver.solve()
__, V2 = amfe.pod(mechanical_system=system, n=50)
system.export_paraview(output_file + '_forV2')


# compute principle/subspace angles
angles_amfe, F1, F2 = amfe.principal_angles(V1=V1, V2=V2, unit='deg', method='auto', principal_vectors=True)
angles_scipy = np.sort(sp.linalg.subspace_angles(A=V1, B=V2))/np.pi*180


# output results
print(F1.shape)
print(F2.shape)

print(len(angles_amfe))
print(len(angles_scipy))

fig, ax = plt.subplots()
ax.plot(angles_amfe, 'ro-', label='AMfe\'s principle_angles(...)')
ax.plot(angles_scipy, 'b+-', label='SciPy.linalg\'s subspace_angles(...)')
ax.grid(True)
ax.legend()
plt.xlabel('number - 1')
plt.ylabel('angle (deg)')
plt.show()

