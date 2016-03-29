"""
Make a comparison the given example
"""

import time
import copy
import dill
import numpy as np
from multiprocessing import Pool
import amfe
import os
from experiments.quadratic_manifold.benchmark_bar_arc import benchmark_system, \
    amfe_dir, alpha, neumann_domain

paraview_output_file = os.path.join(amfe_dir, 'results/test_examples/' +
                                    time.strftime("%Y%m%d_%H%M%S"))

def harmonic_y(t):
    return np.sin(2*np.pi*t*20) + np.sin(2*np.pi*t*30)

benchmark_system.apply_neumann_boundaries(key=neumann_domain, val=4E5,
                                          direct=(0,1),
                                          time_func=harmonic_y)

file_string = '_bar_arc_R1_h01_f4E5_'

om_shift = 30 * 2*np.pi

no_of_modes = 10
dt_calc = 2E-4
dt_output = 1E-3
t_end = 0.4


SQ_EPS = amfe.model_reduction.SQ_EPS


def run_dill_encoded(what):
    fun, args = dill.loads(what)
    return fun(*args)

def apply_async(pool, fun, args):
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),))


def compute_linearized_sol(mech_system, filename):
    '''
    Compute the linearized solution of the given benchmark problem
    '''
    ndof = mech_system.dirichlet_class.no_of_constrained_dofs
    q0, dq0 = np.zeros(ndof), np.zeros(ndof)
    time_range = np.arange(0, t_end, dt_output)
    amfe.integrate_linear_system(mech_system, q0, dq0, time_range, dt=dt_calc,
                             alpha=alpha)
    mech_system.export_paraview(filename)


def compute_qm_smd_sol(mech_system, filename):
    '''
    Compute the solution of the mechanical system usinig QM approach
    '''
    dofs_reduced = no_of_modes
    omega, V = amfe.vibration_modes(mech_system, n=no_of_modes)
    dofs_full = V.shape[0]
    M = mech_system.M()
    K = mech_system.K()

    # Create a static MD QM system
    theta = amfe.static_correction_theta(V, mech_system.K)
    # theta = amfe.modal_derivative_theta(V, omega, benchmark_system.K, M, h=SQ_EPS,\
#                                    symmetric=True)
    my_qm_sys = amfe.qm_reduce_mechanical_system(mech_system, V, theta)

    my_newmark = amfe.NewmarkIntegrator(my_qm_sys, alpha=alpha)
    my_newmark.verbose = True
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = 100
    my_newmark.atol = 1E-7
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(dofs_reduced), np.zeros(dofs_reduced), t_series)
    my_qm_sys.export_paraview(filename)

def compute_qm_smd_shift_sol(mech_system, filename):
    '''
    Compute the solution of the mechanical system usinig QM approach
    '''
    dofs_reduced = no_of_modes
    omega, V = amfe.vibration_modes(mech_system, n=no_of_modes)
    dofs_full = V.shape[0]
    M = mech_system.M()
    K = mech_system.K()
    # Create a static MD QM system
    theta = amfe.static_correction_theta(V, mech_system.K, M, om_shift)
    # theta = amfe.modal_derivative_theta(V, omega, benchmark_system.K, M, h=SQ_EPS,\
#                                    symmetric=True)
    my_qm_sys = amfe.qm_reduce_mechanical_system(mech_system, V, theta)

    my_newmark = amfe.NewmarkIntegrator(my_qm_sys, alpha=alpha)
    my_newmark.verbose = True
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = 100
    my_newmark.atol = 1E-7
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(dofs_reduced), np.zeros(dofs_reduced), t_series)
    my_qm_sys.export_paraview(filename)


def compute_qm_md_sol(mech_system, filename):
    '''
    Compute the solution of the mechanical system usinig QM approach
    '''
    dofs_reduced = no_of_modes
    omega, V = amfe.vibration_modes(mech_system, n=no_of_modes)
    dofs_full = V.shape[0]
    M = mech_system.M()
    K = mech_system.K()

    # Create a static MD QM system
    # theta = amfe.static_correction_theta(V, mech_system.K)
    theta = amfe.modal_derivative_theta(V, omega, benchmark_system.K, M, h=SQ_EPS,\
                                    symmetric=True)
    my_qm_sys = amfe.qm_reduce_mechanical_system(mech_system, V, theta)

    my_newmark = amfe.NewmarkIntegrator(my_qm_sys, alpha=alpha)
    my_newmark.verbose = True
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = 100
    my_newmark.atol = 1E-7
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(dofs_reduced), np.zeros(dofs_reduced), t_series)
    my_qm_sys.export_paraview(filename)



def compute_full_sol(mech_system, filename):
    '''
    Compute the full solution of the mechanical system and save it in filename
    '''
    ndof = mech_system.dirichlet_class.no_of_constrained_dofs
    my_newmark = amfe.NewmarkIntegrator(mech_system, alpha=alpha)
    my_newmark.delta_t = dt_calc
    my_newmark.atol = 1E-3
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(ndof), np.zeros(ndof), t_series)

    mech_system.export_paraview(filename)


#%% Parallel execution

pool = Pool()



filename_lin = paraview_output_file + file_string + 'linearized'
mech_sys_lin = copy.deepcopy(benchmark_system)
args_lin = [mech_sys_lin, filename_lin]
result_lin = apply_async(pool, compute_linearized_sol, args_lin)

filename_full = paraview_output_file + file_string + 'full'
mech_sys_full = copy.deepcopy(benchmark_system)
args_full = [mech_sys_full, filename_full]
result_full = apply_async(pool, compute_full_sol, args_full)

filename_qm_md = paraview_output_file + file_string + 'qm_md'
mech_sys_qm_md = copy.deepcopy(benchmark_system)
args_qm_md = [mech_sys_qm_md, filename_qm_md]
result_qm_md = apply_async(pool, compute_qm_md_sol, args_qm_md)

filename_qm_smd = paraview_output_file + file_string + 'qm_smd'
mech_sys_qm_smd = copy.deepcopy(benchmark_system)
args_qm_smd = [mech_sys_qm_smd, filename_qm_smd]
result_qm_smd = apply_async(pool, compute_qm_smd_sol, args_qm_smd)

filename_qm_smd_shift = paraview_output_file + file_string + 'qm_smd_shift'
mech_sys_qm_smd_shift = copy.deepcopy(benchmark_system)
args_qm_smd_shift = [mech_sys_qm_smd_shift, filename_qm_smd_shift]
result_qm_smd_shift = apply_async(pool, compute_qm_smd_shift_sol, args_qm_smd_shift)



answer_lin = result_lin.get()
answer_full = result_full.get()
answer_qm_md = result_qm_md.get()
answer_qm_smd = result_qm_smd.get()
answer_qm_smd_shift = result_qm_smd_shift.get()
