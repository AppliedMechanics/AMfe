# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:22:58 2015

@author: johannesr
"""

import numpy as np
import scipy as sp
import time

from matplotlib import pyplot as plt
# make amfe running
import sys
sys.path.insert(0,'..')
import amfe



# für den Bogen
gmsh_input_file = '../meshes/gmsh/bogen_grob.msh'
start_index = amfe.node2total(54, 0)
end_index = amfe.node2total(55, 1)

# fixation
bottom_bounds_1 = [None, [0,1,2,3], None]
bottom_bounds_2 = [None, np.arange(start_index, end_index + 1), None]

# für die Stange
gmsh_input_file = '../meshes/gmsh/stange_order1.msh'

paraview_output_file = '../results/stange' + time.strftime("_%Y%m%d_%H%M%S") + '/stange'

# Default values;
kwargs = {'E_modul' : 210E9, 'poisson_ratio' : 0.3, 'element_thickness' : 1, 'density' : 1E4}
element_class_dict = {'Tri3' : amfe.Tri3(**kwargs), 'Tri6' : amfe.Tri6(**kwargs)}


my_system = amfe.MechanicalSystem()
my_system.element_class_dict = element_class_dict
my_system.load_mesh_from_gmsh(gmsh_input_file)
# my_system.export_paraview(paraview_output_file)

nodes_to_fix = my_system.mesh_class.boundary_line_list[3]
bottom_bounds_1 = [None, [amfe.node2total(i, 0) for i in nodes_to_fix], None]
bottom_bounds_2 = [None, [amfe.node2total(i, 1) for i in nodes_to_fix], None]
my_dirichlet_bounds = [bottom_bounds_1, bottom_bounds_2]
my_system.apply_dirichlet_boundaries(my_dirichlet_bounds)


other_side = [2, 3, 24, 25]

neumann_bounds = [  [[amfe.node2total(i,1) for i in other_side], 'harmonic', (6E5, 20), None],
                    [[amfe.node2total(i,0) for i in other_side], 'harmonic', (2E5, 100), None]]
my_system.apply_neumann_boundaries(neumann_bounds)


ndof = my_system.ndof_global_constrained
#
################################################################################
### time integration
################################################################################
#
#my_newmark = amfe.NewmarkIntegrator()
#my_newmark.set_mechanical_system(my_system)
#my_newmark.delta_t = 8E-5
#my_newmark.integrate_nonlinear_system(np.zeros(ndof), np.zeros(ndof), np.arange(0,0.2,0.00008))



# import reduced_system


# modal derivatives:



K = my_system.K_global()
M = my_system.M_global()
print()

lambda_, V = sp.linalg.eigh(K.toarray(), M.toarray())
omega = np.sqrt(lambda_)


K = K.toarray()
M = M.toarray()

h = np.sqrt(np.finfo(float).eps)*100

# mass normalize V:
for i in range(V.shape[0]):
    V[:,i] /= np.sqrt(V[:,i].dot(M.dot(V[:,i])))

#%%
# d_phi_i / d_phi_j
i, j = 0, 0

def my_K_func(u):
    return my_system.K_global(u).toarray()

def calc_modal_deriv(omega, V, K_func, i, j):
    '''Calculates the modal derivative of the given system'''
    h = np.sqrt(np.finfo(float).eps)*100
    ndof = V.shape[0]
    K = K_func(np.zeros(ndof))
    x_i = V[:,i]
    eta_j = V[:,j]
    om = omega[i]
    dK_deta_j = (K_func(eta_j*h) - K)/h
    F_i = (x_i.dot(dK_deta_j.dot(x_i))*M - dK_deta_j).dot(x_i)
    K_dyn_i = K - om**2*M
    row_index = np.argmax(abs(x_i))
    K_dyn_i[:,row_index], K_dyn_i[row_index,:], K_dyn_i[row_index,row_index] = 0, 0, 1
    F_i[row_index] = 0
    v_i = sp.linalg.solve(K_dyn_i, F_i)
    c_i = -v_i.dot(M.dot(x_i))
    dphi_i_deta_j = v_i + c_i*x_i

    # Simplified mds:
    simplified_md = sp.linalg.solve(K, - dK_deta_j.dot(x_i))
    return dphi_i_deta_j, simplified_md


no_of_mod_devs = 7
for i in range(no_of_mod_devs):
    for j in range(no_of_mod_devs):
        dphi_deta, simplified_md = calc_modal_deriv(omega, V, my_K_func, i, j)
        # my_system.write_timestep((i+1)*10+j+1, dphi_deta)
        my_system.write_timestep((i+1)*10+j+1, simplified_md)

# my_system.write_timestep(10, dphi_i_deta_j)
#my_system.write_timestep(11, v_i/amfe.norm_of_vector(v_i))


#%%

# plotting results
plt.semilogy(omega)
#
for i in range(no_of_mod_devs):
    my_system.write_timestep((i+1)*100, V[:,i])




my_system.export_paraview(paraview_output_file)


