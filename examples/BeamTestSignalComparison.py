# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information.
#
"""
Example: Cantilever beam loaded at tip.
"""


# load packages
import amfe
import numpy as np
import matplotlib.pyplot as plt


# define input file
input_file = amfe.amfe_dir('meshes/gmsh/beam/Beam10x1Quad4.msh')


# FOM
material = amfe.KirchhoffMaterial(E=2.1e11, nu=0.3, rho=7.867e3, plane_stress=False)
system = amfe.MechanicalSystem()
system.load_mesh_from_gmsh(input_file, 1, material)
system.apply_dirichlet_boundaries(5, 'xy')
ndof = system.dirichlet_class.no_of_constrained_dofs
system.apply_neumann_boundaries(key=3, val=2.5e8, direct=(0, -1), time_func=lambda t: 1)

options = {
    't0': 0.0,
    't_end': 0.4,
    'dt': 1e-3,
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
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
solver.solve()
x1 = np.array(np.delete(system.u_output, [0, 1, 6, 7], 1)).T
t1 = np.array(system.T_output)


# ROM
r = 2
__, V = amfe.pod(mechanical_system=system, n=r)
system = amfe.reduce_mechanical_system(mechanical_system=system, V=V, overwrite=True)


options['initial_conditions']['q0'] = np.zeros(r)
options['initial_conditions']['dq0'] = np.zeros(r)
solver = amfe.GeneralizedAlphaNonlinearDynamicsSolver(mechanical_system=system, **options)
solver.solve()
x2 = np.array(np.delete(system.u_output, [0, 1, 6, 7], 1)).T
t2 = np.array(system.T_output)


# compare outputs
plt.figure(figsize=(16, 9), dpi=120)
plt.plot(t1, x1[1, :], label='u_1 x1(t1)')
plt.plot(t2, x2[1, :], label='u_1 x2(t2)')
plt.plot(t1, x1[2, :], label='u_2 x1(t1)')
plt.plot(t2, x2[2, :], label='u_2 x2(t2)')
plt.legend()
plt.show()


print(amfe.compare_signals(x1=x1, t1=t1, x2=x2, t2=t2, method='norm', axis=1, ord_v=np.inf, ord_s=np.inf))
angles1, angles2, __, __ = amfe.compare_signals(x1=x1, t1=t1, x2=x2, t2=t2, method='angle', axis=1, num=7, unit='deg')
mac1, mac2, s1, s2 = amfe.compare_signals(x1=x1, t1=t1, x2=x2, t2=t2, method='mac', axis=1)
ccor, acor = amfe.compare_signals(x1=x1, t1=t1, x2=x2, t2=t2, method='correlation', axis=1)

plt.figure(figsize=(16, 9), dpi=120)
plt.plot(angles1, label='U')
plt.plot(angles2, label='V')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16, 9), dpi=120)
plt.pcolor(mac1, label='U')
plt.colorbar()
plt.legend()
plt.show()

plt.figure(figsize=(16, 9), dpi=120)
plt.pcolor(mac2, label='V')
plt.colorbar()
plt.legend()
plt.show()

plt.figure(figsize=(16, 9), dpi=120)
plt.plot(s1, label='s1')
plt.plot(s2, label='s2')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(16, 9), dpi=120)
plt.plot(ccor[0, :], label='u_0')
plt.plot(ccor[1, :], label='u_1')
plt.plot(ccor[4, :], label='u_4')
plt.plot(ccor[5, :], label='u_5')
plt.plot(ccor[9, :], label='u_9')
plt.plot(ccor[10, :], label='u_10')
plt.legend()
plt.grid()
plt.show()

