# Beam example

# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Example showing a cantilever beam which is loaded on the tip with a force
showing nonlinear displacements.

The beam is reduced with ECSW and NSKTS
"""

import time
import numpy as np

import amfe


times = dict([])
input_file = amfe.amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe.amfe_dir('results/beam_nonlinear/beam_ecsw')


my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(input_file, 7, my_material)
my_system.apply_dirichlet_boundaries(8, 'xy') # fixature of the left side
my_system.apply_neumann_boundaries(key=9, val=1E8, direct=(0,-1),
                                   time_func=lambda t: np.sin(31*t))


#
solverti = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_system, dt=0.001, t_end=1)
my_system.clear_timesteps()
t0 = time.time()
solverti.solve()
t1 = time.time()
my_system.export_paraview(output_file + '_nonlinear_ti_full')
times.update({'time integration of full problem:': t1-t0})


solverlinti = amfe.GeneralizedAlphaLinearDynamicsSolver(my_system, dt=0.001, t_end=1)
my_system.clear_timesteps()
t0 = time.time()
solverlinti.solve()
t1 = time.time()
my_system.export_paraview(output_file + '_linear_ti_full')
times.update({'time integration of linear problem:': t1-t0})
# Basis generation:

t0 = time.time()
omega, V = amfe.reduced_basis.vibration_modes(my_system, 6)
Theta = amfe.reduced_basis.static_derivatives(V, my_system.K)
V_extended = amfe.augment_with_derivatives(V, Theta)
t1 = time.time()
times.update({'nonlinear basis generation:' : t1-t0})

# Write linear basis
my_system.clear_timesteps()
for i in np.arange(V.shape[1]):
    my_system.write_timestep(i, V[:,i])
my_system.export_paraview(output_file + '_linear_basis')

# Write Static derivatives
my_system.clear_timesteps()
counter = 0
for i in np.arange(Theta.shape[1]):
    for j in np.arange(Theta.shape[1]):
        if i > j:
            my_system.write_timestep(counter, Theta[:,i,j])
            counter = counter + 1
my_system.export_paraview(output_file + '_static_derivatives')


# Training Set Generation
t0 = time.time()
nskts = amfe.hyper_red.compute_nskts(my_system)
t1 = time.time()
times.update({'Training-Set Generation (NSKTS):': t1-t0})
my_system.clear_timesteps()
for i in range(nskts.shape[1]):
    my_system.write_timestep(i, nskts[:, i])
my_system.export_paraview(output_file + '_nskts')


# Reduce system
t0 = time.time()
my_red_system = amfe.reduce_mechanical_system(my_system, V_extended)
t1 = time.time()
times.update({'Reduction step:': t1-t0})
ndofs = V_extended.shape[1]
initial_conditions = {'q0': np.zeros(ndofs), 'dq0': np.zeros(ndofs)}
solverti_red = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_red_system, dt=0.001, t_end=1, initial_conditions=initial_conditions)
my_red_system.clear_timesteps()

t0 = time.time()
solverti_red.solve()
t1 = time.time()
times.update({'Time integration of reduced system:': t1-t0})
my_red_system.export_paraview(output_file + '_nonlinear_ti_red')


# Hyperreduction ECSW
t0 = time.time()
my_hyperred_ecsw = amfe.reduce_mechanical_system_ecsw(my_system,V_extended)
q_training = np.linalg.solve((V_extended.T @ V_extended), V_extended.T @ nskts)
my_hyperred_ecsw.reduce_mesh(q_training)
t1 = time.time()
times.update({'Hyperreduction step:': t1-t0})

my_hyperred_ecsw.export_paraview(output_file + '_weights_ecsw')


solverti_hyperred_ecsw = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_hyperred_ecsw, dt=0.001, t_end=1, initial_conditions=initial_conditions)
my_hyperred_ecsw.clear_timesteps()
t0 = time.time()
solverti_hyperred_ecsw.solve()
t1 = time.time()
times.update({'Time integration of ECSW-hyperreduced system:': t1-t0})
my_hyperred_ecsw.export_paraview(output_file + '_nonlinear_ti_hyperred_ecsw')
del my_hyperred_ecsw

# Hyperreduction DEIM
t0 = time.time()
my_hyperred_deim = amfe.reduce_mechanical_system_deim(my_system,V_extended)
q_training = np.linalg.solve((V_extended.T @ V_extended), V_extended.T @ nskts)
my_hyperred_deim.reduce_mesh(V_extended @ q_training, no_of_force_modes=50)
t1 = time.time()
times.update({'Hyperreduction step DEIM:': t1-t0})



solverti_hyperred_deim = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_hyperred_deim, dt=0.001, t_end=1, initial_conditions=initial_conditions, max_number_of_iterations=50)
my_hyperred_deim.clear_timesteps()
t0 = time.time()
solverti_hyperred_deim.solve()
t1 = time.time()
times.update({'Time integration of DEIM-hyperreduced system:': t1-t0})
my_hyperred_deim.export_paraview(output_file + '_nonlinear_ti_hyperred_deim')
del my_hyperred_deim


with open(output_file + '_simulation_times', 'w+') as fp:
    for simtime in times:
        fp.write(simtime + ' {}\n'.format(times[simtime]))

print('END')