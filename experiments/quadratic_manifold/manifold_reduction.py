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

#%%
# create a regular QM system
dofs_reduced = no_of_modes = 5
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)
dofs_full = V.shape[0]


theta = amfe.static_correction_theta(V, benchmark_system.K)
# theta = sp.zeros((dofs_full, dofs_reduced, dofs_reduced))

my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%%
# Try on a purging algorithm
A = np.zeros((no_of_modes, no_of_modes))
for i in range(no_of_modes):
    for j in range(i+1):
        v = V[:,i]
        v /= np.sqrt(v @ v)
        th = theta[:,i,j]
        th /= np.sqrt(th @ th)
        A[i,j] = A[j,i] = abs(th @ v)
        
        # print('Inner product of i {0:d} and j {1:d} is {2:4.4f}'.format(i, j, th @ v))

plt.matshow(A)

#%%
# Other type of purging by setting stuff in theta to zero
theta[:,:,3] = 0
theta[:,3,:] = 0

#%%
# Build a QM system which is purged of the in-plane modes:
dofs_reduced = no_of_modes = 20
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)

select = np.ones((no_of_modes), dtype=bool)
# columns to purge
select[np.ix_((3,7,10,13,15,18))] = False
V = V[:,select]

dofs_full, dofs_reduced = V.shape
theta = amfe.static_correction_theta(V, benchmark_system.K)
my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%%
# plot the modes of the system

for t, phi in enumerate(V.T):
    benchmark_system.write_timestep(t, phi)
benchmark_system.export_paraview(paraview_output_file)

#%%
# plot the modal derivatives:
i_mode = 4
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

my_newmark.integrate(np.zeros(dofs_reduced), 
                                      np.zeros(dofs_reduced), np.arange(0, 0.4, 1E-4))

my_qm_sys.export_paraview(paraview_output_file)

#%%
# plot the time line of the reduced variable 
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





