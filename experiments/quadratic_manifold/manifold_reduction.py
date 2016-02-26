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
from experiments.quadratic_manifold.benchmark_u import benchmark_system, \
    amfe_dir, alpha

paraview_output_file = os.path.join(amfe_dir, 'results/qm_reduction' +
                                    time.strftime("_%Y%m%d_%H%M%S"))

SQ_EPS = amfe.model_reduction.SQ_EPS

def check_orthogonality(u,v):
    '''
    Check the orthogonality of two vectors, no matter what their length is. 
    '''
    u_n = u / np.sqrt(u @ u)
    v_n = v / np.sqrt(v @ v)
    return u_n @ v_n


def theta_m_orth_v(Theta, V, M):
    '''
    Make Theta mass orthogonal with respect to the parent modes via a 
    Gram-Schmid-Process. 
    
    Parameters
    ----------
    Theta : ndarray
        Third order Tensor describing the quadratic part of the basis
    V : ndarray
        Linear Basis 
    M : ndarray or scipy.sparse matrix
        Mass Matrix
    
    Returns
    -------
    Theta_orth : ndarray
        Third order tensor Theta mass orthogonalized, such that 
        Theta_orth[:,i,j] is mass orthogonal to V[:,i] and V[:,j]:

            >>> Theta_orth[:,i,j] @ M @ V[:,i] == np.zeros(ndim)
        
    '''
    __, no_of_modes = V.shape
    # Make sure, that V is M-normalized
    np.testing.assert_allclose(V.T @ M @ V, np.eye(no_of_modes), atol=1E-14)
    Theta_ret = Theta.copy()
    for i in range(no_of_modes):
        for j in range(no_of_modes):
            Theta_ret[:,i,j] -= V[:,i] * (Theta[:,i,j] @ M @ V[:,i])
            Theta_ret[:,i,j] -= V[:,j] * (Theta[:,i,j] @ M @ V[:,j])
    return Theta

#%% Create a static MD QM system

dofs_reduced = no_of_modes = 10
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)
dofs_full = V.shape[0]

# try to make one guy smaller!
# V[:,3] = V[:,3]/100

theta = amfe.static_correction_theta(V, benchmark_system.K)
# theta = sp.zeros((dofs_full, dofs_reduced, dofs_reduced))

my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%% create a MD QM system
 
dofs_reduced = no_of_modes = 10
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)

dofs_full = V.shape[0]
M = benchmark_system.M()
print('Take care! Theta is not symmetric now!')
theta = amfe.modal_derivative_theta(V, omega, benchmark_system.K, M, h=SQ_EPS,\
                                    symmetric=False)
# theta = 1/2*(theta * theta.transpose(0,2,1))
my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%% Show the inner products of theta with respect to the modes

M = benchmark_system.M()
A = np.zeros((no_of_modes, no_of_modes))
norm_mat = np.eye(V.shape[0])
norm_mat = M
for i in range(no_of_modes):
    for j in range(no_of_modes):
        v = V[:,i].copy()
        v /= np.sqrt(v @ norm_mat @ v)
        th = theta[:,i,j].copy()
        th /= np.sqrt(th @ norm_mat @ th)
        A[i,j] = abs(th @ norm_mat @ v)
        
        # print('Inner product of i {0:d} and j {1:d} is {2:4.4f}'.format(i, j, th @ v))

plt.matshow(A, norm=mpl.colors.LogNorm());plt.colorbar()
plt.title('Inner product of V with theta')

#%% Purging algorithm where theta is kept mass orthogonal to V



M = benchmark_system.M()
theta = make_theta_mass_orthogonal(theta, V, M)
my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%% Second approach: Show the norm of the vector in theta

L = np.einsum('ijk,ijk->jk', theta, theta)
L = np.sqrt(L)
plt.matshow(L, norm=mpl.colors.LogNorm());plt.colorbar()
plt.title('Length of the vectors in theta')

#%% Other type of purging by setting stuff in theta to zero

theta[:,:,3] = 0
theta[:,3,:] = 0

theta[:,:,7] = 0
theta[:,7,:] = 0
my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%% Build a QM system which is purged of the in-plane modes:

dofs_reduced = no_of_modes = 20
omega, V = amfe.vibration_modes(benchmark_system, n=no_of_modes)

select = np.ones((no_of_modes), dtype=bool)
# columns to purge
select[np.ix_((3,7,10,13,15,18))] = False
V = V[:,select]

dofs_full, dofs_reduced = V.shape
theta = amfe.static_correction_theta(V, benchmark_system.K)
my_qm_sys = amfe.qm_reduce_mechanical_system(benchmark_system, V, theta)

#%% export the modes of the system to ParaView

for t, phi in enumerate(V.T):
    benchmark_system.write_timestep(t, phi)
out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)

#%% exprot the modal derivatives of the system
for i in range(no_of_modes):
    #for j in range(i + 1):
    for j in range(no_of_modes):
        benchmark_system.write_timestep(i*100 + j, theta[:,i,j])

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)

#%%plot the modes growing with the modal derivatives 

i_mode = 1
for t in np.arange(0,20,0.1):
    u = np.zeros(no_of_modes)
    u[i_mode] = t
    my_qm_sys.write_timestep(t, u)


#%% Export the benchmark system to paravew

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)

#%% Perform some time integration
###############################################################################
# Perform some time integration
###############################################################################

my_newmark = amfe.NewmarkIntegrator(my_qm_sys, alpha=alpha)
my_newmark.verbose = True
my_newmark.delta_t = 1E-4
my_newmark.n_iter_max = 100
my_newmark.atol = 1E-7
#my_newmark.write_iter = True

my_newmark.integrate(np.zeros(dofs_reduced), np.zeros(dofs_reduced), 
                     np.arange(0, 0.4, 1E-4))

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)

#%% plot the time line of the reduced variable 

q_red = np.array(my_qm_sys.u_red_output)
t = np.array(my_qm_sys.T_output)
plt.figure()
plt.plot(t, q_red[:,:])
plt.grid()

#%% Check the condition of the projector P
# first column is condition number, second scaling, third orthogonality

conds = np.zeros_like(q_red[:,:3])
for i, q in enumerate(q_red):
    P = V + 2*(theta @ q)
    conds[i,0] = np.linalg.cond(P)
    diag = np.diag(P.T @ P)
    diag = np.sqrt(diag)
    conds[i,1] = np.max(diag)/np.min(diag)
    P_normal = np.einsum('ij,j->ij', P,1/diag)
    conds[i,2] = np.linalg.cond(P_normal)
    

plt.figure()
plt.semilogy(t, conds[:,0], label='cond P')
plt.semilogy(t, conds[:,1], label='cond P scaling')
plt.semilogy(t, conds[:,2], label='cond P vecs')
plt.legend()
plt.grid()


#%% Check condition number due to bad lengthes in scaling

conds_scaling = np.zeros_like(q_red[:,0])
for i, q in enumerate(q_red):
    P = V + 2*(theta @ q)
    conds_scaling[i] = np.linalg.cond(P)

plt.figure()
plt.semilogy(t, conds); plt.grid()


#%% Check, how the modes in P look like by export to ParaView

for t, phi in enumerate(P.T):
    benchmark_system.write_timestep(t, phi)

out_file = amfe.append_to_filename(paraview_output_file)
benchmark_system.export_paraview(out_file)

#%% Show the difference of MDs and SMDs of the system

ndim, nred = V.shape
symmetric = False
orthogonal = True
print('Pay attention. The MDs are NOT symmetric.')
Theta_MD = amfe.modal_derivative_theta(V, omega, benchmark_system.K, M, \
                                       h=SQ_EPS, symmetric=symmetric)
Theta_SMD = amfe.static_correction_theta(V, benchmark_system.K)

# orthogonalization:
if orthogonal:
    Theta_MD = theta_m_orth_v(Theta_MD, V, M)
    Theta_SMD = theta_m_orth_v(Theta_SMD, V, M)

# norming
norm_Theta_MD = np.sqrt(np.einsum('ijk,ijk->jk', Theta_MD, Theta_MD))
norm_Theta_SMD = np.sqrt(np.einsum('ijk,ijk->jk', Theta_SMD, Theta_SMD))

A = np.einsum('ijk, ijk->jk', Theta_MD, Theta_SMD)/(norm_Theta_MD*norm_Theta_SMD)

plt.matshow(A); plt.colorbar()

amfe.matshow_3d(1 - np.abs(A), thickness=0.4, alpha=0.3)
# matshow_bar(np.arccos(A)/(np.pi))

#%%
plt.matshow(norm_Theta_MD);plt.colorbar()
#%%
#%% Try to perform some tests on the QM system
#%%
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





