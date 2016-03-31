"""
Make a comparison of the given example using linearized simulation, QM 
simulation and a full reference simulation. 
"""

import time
import copy
import dill
import numpy as np
import scipy as sp
from multiprocessing import Pool
import amfe
import os
from experiments.quadratic_manifold.benchmark_bar import benchmark_system, \
    amfe_dir, alpha

paraview_output_file = os.path.join(amfe_dir, 'results/test_examples/' +
                                    time.strftime("%Y%m%d_%H%M%S"))

#def harmonic_y(t):
#    return np.sin(2*np.pi*t*20) + np.sin(2*np.pi*t*30)
#
#benchmark_system.apply_neumann_boundaries(key=neumann_domain, val=4E5,
#                                          direct=(0,1),
#                                          time_func=harmonic_y)

# file_string = '_bar_arc_R8_h01_f3E5_'
# file_string = '_bar_2_f5E5_'
file_string = '_bar_f2E7_'

om_shift = 30 * 2*np.pi

conv_abort = True
verbose = False
n_iter_max = 200
no_of_modes = 10
dt_calc = 1E-4
dt_output = 1E-3
t_end = 0.4
atol = 1E-5


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
    my_newmark.verbose = verbose
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = n_iter_max
    my_newmark.atol = atol
    my_newmark.conv_abort = conv_abort
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
    my_newmark.verbose = verbose
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = n_iter_max
    my_newmark.atol = atol
    my_newmark.conv_abort = conv_abort
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(dofs_reduced), np.zeros(dofs_reduced), t_series)
    my_qm_sys.export_paraview(filename)



def compute_qm_kry_sol(mech_system, filename):
    '''
    Compute the solution of the mechanical system usinig QM approach
    '''
    M = mech_system.M()
    K = mech_system.K()
    dofs_full = M.shape[0]
    dofs_reduced = no_of_modes
    f = mech_system.f_ext(np.zeros(dofs_full), np.zeros(dofs_full), sp.rand())
    # f /= np.sqrt(f @ f)
    V = amfe.krylov_subspace(M, K, f, no_of_moments=no_of_modes)

    # Create a static MD krylov QM system
    theta = amfe.static_correction_theta(V, mech_system.K)
    my_qm_sys = amfe.qm_reduce_mechanical_system(mech_system, V, theta)

    my_newmark = amfe.NewmarkIntegrator(my_qm_sys, alpha=alpha)
    my_newmark.verbose = verbose
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = n_iter_max
    my_newmark.atol = atol
    my_newmark.conv_abort = conv_abort
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
    my_newmark.verbose = verbose
    my_newmark.delta_t = dt_calc
    my_newmark.n_iter_max = n_iter_max
    my_newmark.atol = atol
    my_newmark.conv_abort = conv_abort
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
    my_newmark.n_iter_max = n_iter_max
    my_newmark.atol = atol
    my_newmark.conv_abort = conv_abort
    t_series = np.arange(0, t_end, dt_output)
    my_newmark.integrate(np.zeros(ndof), np.zeros(ndof), t_series)

    mech_system.export_paraview(filename)


#%% Parallel execution

pool = Pool()

job_funcs = {
    'linearized': compute_linearized_sol, 
    'full': compute_full_sol,
    'qm_md': compute_qm_md_sol,
    'qm_smd': compute_qm_smd_sol,
    'qm_smd_shift': compute_qm_smd_shift_sol,
    'qm_kry': compute_qm_kry_sol,
}




job_results = {}
for job in job_funcs:
    filename = paraview_output_file + file_string + job
    mech_sys = copy.deepcopy(benchmark_system)
    args =  [mech_sys, filename]
    job_results[job] = apply_async(pool, job_funcs[job], args)

for job in job_results:
    job_results[job].get()
