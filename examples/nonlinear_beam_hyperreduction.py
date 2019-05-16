# Beam example

# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Example showing a cantilever beam which is loaded on the tip with a force
showing nonlinear displacements.

The beam is reduced with ECSW and NSKTS
"""

import os
import time
import numpy as np
from h5py import File

from amfe.ui import *
from amfe.io import amfe_dir
from amfe.io.mesh import AmfeMeshObjMeshReader
from amfe.io.postprocessing import *
from amfe.material import KirchhoffMaterial
from amfe.solver import *
from amfe.mor import *
from amfe.mor.hyper_red import *
from amfe.structural_dynamics import vibration_modes

studies = []
# studies.append('full_ti')
studies.append('create_basis_1')
#studies.append('red_ti')
#studies.append('ecsw')
studies.append('poly3')

Omega = 31.0

times = dict([])
input_file = amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe_dir('results/beam_nonlinear_refactoring/beam_ecsw')

# Define material
material = KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
# Load Mesh
mesh = import_mesh_from_file(input_file)
# Create Component
component = create_structural_component(mesh)
# Assign material
component.assign_material(material, [7], 'S')
# Assign Dirichlet Boundaries
set_dirichlet_by_group(component, [8], ('ux', 'uy'))
# Assign Neumann Boundaries
force = component.neumann.create_fixed_direction_neumann(np.array([0, -1], dtype=float),
                                                         lambda t: 1E8*np.sin(Omega*t))
component.assign_neumann('Force', force, [9])

#
system, formulation = create_constrained_mechanical_system_from_component(component, constant_mass=True,
                                                                          constant_damping=True,
                                                                          constraint_formulation='boolean')

# Solver Factory:
solfac = SolverFactory()
solfac.set_system(system)
solfac.set_dt_initial(0.001)
solfac.set_large_deflection(True)
solfac.set_newton_maxiter(30)
solfac.set_newton_atol(1e-6)
solfac.set_newton_rtol(1e-8)
solfac.set_linear_solver('scipy-sparse')
solfac.set_nonlinear_solver('newton')
solfac.set_analysis_type('transient')
solfac.set_integrator('genalpha')

solver = solfac.create_solver()

sol_dyn_nl_full = AmfeSolution()


def write_callback(t, x, dx, ddx):
    u, du, ddu = formulation.recover(x, dx, ddx, t)
    sol_dyn_nl_full.write_timestep(t, u, du, ddu)


no_of_dofs = system.dimension
x0 = np.zeros(no_of_dofs)
dx0 = x0.copy()
t_start = 0.0
t_end = 1.0

if 'full_ti' in studies:
    t0 = time.time()
    solver.solve(write_callback, t_start, x0, dx0, t_end)
    t1 = time.time()
    times.update({'Nonlinear full solution:': t1-t0})
    print('Full dynamic solution took {} seconds'.format(t1-t0))

    write_results_to_paraview(sol_dyn_nl_full, component, output_file + '_dyn_nl_full')

# ------------------- SOLVE LINEAR DYNAMICS ------------------------------

if 'create_basis_1' in studies:
    t0 = time.time()
    K0 = system.K(x0, dx0, 0.0)
    M0 = system.M(x0, dx0, 0.0)
    omega, V = vibration_modes(K0, M0, 6, mass_orth=True)

    def sdK(x):
        return system.K(x, x0, 0.0)

    Theta = static_derivatives(V, sdK, M0)
    V_extended = augment_with_derivatives(V, Theta)
    t1 = time.time()
    times.update({'nonlinear basis generation:': t1-t0})
    print('nonlinear basis generation took {} seconds'.format(t1-t0))

    sol_basis_1 = AmfeSolution()
    for i in np.arange(V_extended.shape[1]):
        u = formulation.u(V_extended[:, i], 0.0)
        sol_basis_1.write_timestep(i, u)

    write_results_to_paraview(sol_basis_1, component, output_file + '_reduction_basis_1')

    sol_basis_sd = AmfeSolution()
    counter = 0
    for i in np.arange(Theta.shape[1]):
        for j in np.arange(Theta.shape[1]):
            if i > j:
                Theta_u = formulation.u(Theta[:, i, j], 0.0)
                sol_basis_sd.write_timestep(counter, Theta_u)
                counter = counter + 1

    write_results_to_paraview(sol_basis_sd, component, output_file + '_static_derivatives')

if 'nskts' in studies:
    # Training Set Generation
    t0 = time.time()
    K0 = system.K(x0, dx0, 0.0)
    M0 = system.M(x0, dx0, 0.0)
    t_max = np.pi/2/Omega
    F_ext_max = system.f_ext(x0, dx0, t_max)

    def fint_func(x):
        return system.f_int(x, dx0, 0.0)

    def K_func(x):
        return system.K(x, dx0, 0.0)

    nskts = compute_nskts(K0, M0, F_ext_max, fint_func, K_func)
    t1 = time.time()
    times.update({'Training-Set Generation (NSKTS):': t1-t0})
    print('Training-Set Generation (NSKTS) took {} seconds'.format(t1 - t0))

    sol_nskts = AmfeSolution()

    for i in range(nskts.shape[1]):
        # Recover unconstrained u
        u = formulation.u(nskts[:, i], 0.0)
        sol_nskts.write_timestep(i, u)

    write_results_to_paraview(sol_nskts, component, output_file + '_nskts')
else:
    nskts = np.load(output_file + '_nskts.npy')


if 'red_ti' in studies:
    # Reduce system
    t0 = time.time()
    red_system = reduce_mechanical_system(system, V_extended, constant_mass=True, constant_damping=True)
    t1 = time.time()
    times.update({'Reduction step:': t1-t0})
    print('Reduction step took {} seconds'.format(t1 - t0))
    solfac.set_system(red_system)

    red_solver = solfac.create_solver()

    sol_dyn_nl_red = AmfeSolution()


    def write_callback(t, x, dx, ddx):
        u, du, ddu = formulation.recover(V_extended.dot(x), V_extended.dot(dx), V_extended.dot(ddx), t)
        sol_dyn_nl_red.write_timestep(t, u, du, ddu)


    no_of_red_dofs = red_system.dimension
    x0 = np.zeros(no_of_red_dofs)
    dx0 = x0.copy()
    t_start = 0.0
    t_end = 1.0

    t0 = time.time()
    red_solver.solve(write_callback, t_start, x0, dx0, t_end)
    t1 = time.time()
    times.update({'Nonlinear full solution:': t1 - t0})
    print('Full dynamic solution took {} seconds'.format(t1 - t0))

    write_results_to_paraview(sol_dyn_nl_red, component, output_file + '_dyn_nl_red')


if 'ecsw_weight_generation' in studies:
    # Hyperreduction ECSW
    t0 = time.time()
    q_training = np.linalg.solve((V_extended.T @ V_extended), V_extended.T @ nskts)

    x_training = V_extended @ q_training
    weights, indices, stats = ecsw_get_weights_from_constrained_training(x_training, component, formulation, V_extended)

    np.save(output_file + '_ecsw_weights.npy', weights)
    np.save(output_file + '_ecsw_indices.npy', indices)
    t1 = time.time()
    times.update({'Hyperreduction step:': t1-t0})
    print('Hyperreduction step:'.format(t1 - t0))

if 'ecsw' in studies:
    weights = np.load(output_file + '_ecsw_weights.npy')
    indices = np.load(output_file + '_ecsw_indices.npy')

    # Create reduced system
    tagname = 'ecsw_weights'
    ecsw_system, ecsw_formulation, ecsw_component = create_ecsw_hyperreduced_mechanical_system_from_weights(component, V_extended, weights, indices, 'boolean',
                                                            constant_mass=True, constant_damping=True,
                                                            tagname=tagname)

    # Solve system
    solfac.set_system(ecsw_system)
    ecsw_solver = solfac.create_solver()

    sol_dyn_nl_ecsw = AmfeSolution()

    def write_callback(t, x, dx, ddx):
        u, du, ddu = ecsw_formulation.recover(V_extended.dot(x), V_extended.dot(dx), V_extended.dot(ddx), t)
        sol_dyn_nl_ecsw.write_timestep(t, u, du, ddu)


    # Set initial conditions
    no_of_dofs = ecsw_system.dimension
    x0 = np.zeros(no_of_dofs)
    dx0 = x0.copy()
    # Set start end endtime for time integration
    t_start = 0.0
    t_end = 1.0

    # Solve hyperreduced system
    t0 = time.time()
    ecsw_solver.solve(write_callback, t_start, x0, dx0, t_end)
    t1 = time.time()
    times.update({'ECSW solution:': t1 - t0})
    print('ECSW solution took {} seconds'.format(t1 - t0))

    # -- POSTPROCESSING --
    # Instantiate Hdf5PostProcessorWriter
    mreader = AmfeMeshObjMeshReader(component.mesh)
    ecsw_output = output_file + '_ecsw_weights.hdf5'
    if os.path.isfile(ecsw_output):
        os.remove(ecsw_output)
    hwriter = Hdf5PostProcessorWriter(mreader, ecsw_output)

    # Write Solution
    preader = AmfeSolutionReader(sol_dyn_nl_ecsw, component)
    preader.parse(hwriter)

    # Write ECSW weights
    data = ecsw_component.mesh.el_df[tagname].values
    indices = ecsw_component.mesh.el_df.index.values
    hwriter.write_field(tagname, PostProcessDataType.SCALAR, sol_dyn_nl_ecsw.t,
                        data, indices, MeshEntityType.ELEMENT)

    # Finish writing -> Call return result
    hwriter.return_result()

    # Write xdmf file from hdf5 for viewing in paraview
    paraviewfilename = output_file + '_ecsw_weights'
    hdf5resultsfilename = paraviewfilename + '.hdf5'
    xdmfresultsfilename = paraviewfilename + '.xdmf'

    fielddict = {'weights': {'mesh_entity_type': MeshEntityType.ELEMENT,
                             'data_type': PostProcessDataType.SCALAR,
                             'hdf5path': '/results/ecsw_weights'
                             },
                 'displacement': {'mesh_entity_type': MeshEntityType.NODE,
                                  'data_type': PostProcessDataType.VECTOR,
                                  'hdf5path': '/results/displacement'
                                  }
                 }

    with open(xdmfresultsfilename, 'wb') as xdmffp:
        with File(hdf5resultsfilename, mode='r') as hdf5fp:
            write_xdmf_from_hdf5(xdmffp, hdf5fp, '/mesh/nodes', '/mesh/topology', sol_dyn_nl_ecsw.t, fielddict)


if 'poly3' in studies:
    K1, K2, K3 = poly3_get_tensors(system, V_extended)
    poly3_system = create_poly3_hyperreduced_system(system, V_extended, K1, K2, K3)

    solfac.set_system(poly3_system)
    poly3_solver = solfac.create_solver()

    sol_dyn_nl_poly3 = AmfeSolution()


    def write_callback(t, x, dx, ddx):
        u, du, ddu = formulation.recover(V_extended.dot(x), V_extended.dot(dx), V_extended.dot(ddx), t)
        sol_dyn_nl_poly3.write_timestep(t, u, du, ddu)


    no_of_red_dofs = poly3_system.dimension
    x0 = np.zeros(no_of_red_dofs)
    dx0 = x0.copy()
    t_start = 0.0
    t_end = 1.0

    t0 = time.time()
    poly3_solver.solve(write_callback, t_start, x0, dx0, t_end)
    t1 = time.time()
    times.update({'Nonlinear full solution:': t1 - t0})
    print('Full dynamic solution took {} seconds'.format(t1 - t0))

    write_results_to_paraview(sol_dyn_nl_poly3, component, output_file + '_dyn_nl_poly3')
