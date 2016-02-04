# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:52:21 2016

@author: rutzmoser
"""

import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import amfe

# % cd experiments/quadratic_manifold/
from experiments.quadratic_manifold.benchmark_bar import benchmark_system, amfe_dir

paraview_output_file = os.path.join(amfe_dir, 'results/qm_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

#%%
dofs_reduced = no_of_modes = 20
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)
dofs_full = V.shape[0]

#print('Take care! The density is put 10 times higher!')

print('Take care! Theta is divided by 2 and multiplied with -1 !')
theta = - amfe.static_correction_theta(V, benchmark_system.K)/2
# theta = sp.zeros((dofs_full, dofs_reduced, dofs_reduced))

my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%%
# plot the modes of the system

for t, phi in enumerate(V.T):
    benchmark_system.write_timestep(t, phi)
benchmark_system.export_paraview(paraview_output_file)

#%%
i_mode = 4
# plot the modal derivatives:
for t in np.arange(0,20,0.1):
    u = np.zeros(no_of_modes)
    u[i_mode] = t
    my_qm_sys.write_timestep(t, u)

#%%
# Export to paraview
my_qm_sys.export_paraview(paraview_output_file)

#%% 
#
# Perform some time integration
# 

my_newmark = amfe.NewmarkIntegrator(my_qm_sys)
my_newmark.verbose = True
my_newmark.delta_t = 1E-4
my_newmark.n_iter_max = 100
#my_newmark.write_iter = True

my_newmark.integrate(np.zeros(no_of_modes), 
                                      np.zeros(no_of_modes), np.arange(0, 0.4, 1E-4))

my_qm_sys.export_paraview(paraview_output_file)

#%%
# plot the stuff
q_red = np.array(my_qm_sys.u_red_output)
t = np.array(my_qm_sys.T_output)
plt.plot(t, q_red[:,:])
plt.grid()

#%%
# Check the condition of the projector P
conds = np.zeros_like(q_red[:,0])
for i, q in enumerate(q_red):
    P = V + 2*(theta @ q)
    conds[i] = np.linalg.cond(P)

plt.semilogy(t, conds); plt.grid()

#%%
# Check, how the modes in P look like
for t, phi in enumerate(P.T):
    benchmark_system.write_timestep(t, phi)

benchmark_system.export_paraview(paraview_output_file)
#%%
def jacobian(func, u):
    '''
    Compute the jacobian of func with respect to u using a finite differences scheme.

    '''
    ndof = u.shape[0]
    jac = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(u).copy()
    for i in range(ndof):
        u_tmp = u.copy()
        u_tmp[i] += h
        f_tmp = func(u_tmp)
        jac[:,i] = (f_tmp - f) / h
    return jac

#%%
#
# Test the stiffness matrix K
#

def func_f(u):
    K, f = my_qm_sys.K_and_f(u)
    return f

u = sp.rand(no_of_modes)
K_fd = jacobian(func_f, u)
K, f = my_qm_sys.K_and_f(u)
np.testing.assert_allclose(K, K_fd, rtol=1E-2, atol=1E-8)

plt.matshow(np.abs(K_fd), norm=mpl.colors.LogNorm())
plt.colorbar()

plt.matshow(np.abs(K_fd - K), norm=mpl.colors.LogNorm())
plt.colorbar()

#%%

#
# Test the dynamic S matrix
#

du = sp.rand(no_of_modes)
ddu = sp.rand(no_of_modes)
dt, t, beta, gamma = sp.rand(4)

dt *= 1E4
def func_res(u):
    S, res = my_qm_sys.S_and_res(u, du, ddu, dt, t, beta, gamma)
    return res

S, res = my_qm_sys.S_and_res(u, du, ddu, dt, t, beta, gamma)
S_fd = jacobian(func_res, u)
plt.matshow(np.abs(S_fd), norm=mpl.colors.LogNorm())
plt.colorbar()

plt.matshow(np.abs((S - S_fd)/S), norm=mpl.colors.LogNorm())
plt.colorbar()

np.testing.assert_allclose(S, S_fd, rtol=1E-2, atol=1E-8)

#%%

theta = sp.rand(dofs_full,dofs_reduced,dofs_reduced)
V = sp.rand(dofs_full, dofs_reduced)

my_qm_sys.V = V
my_qm_sys.Theta = theta
my_qm_sys.no_of_red_dofs = dofs_reduced

z = sp.rand(20)
dz = sp.rand(20)
ddz = sp.rand(20)
dt = 0.001
t = 1.0
beta = 1/2
gamma = 1.

my_qm_sys.S_and_res(z, dz, ddz, dt, t, beta, gamma)





