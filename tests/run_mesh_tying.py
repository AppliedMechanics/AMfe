"""
run the mesh tying algorithm
"""

import numpy as np
import scipy as sp
from scipy import spatial
from matplotlib import pyplot as plt
import amfe

#%%

def proj_quad4(X, p, niter_max=20, eps=1E-10, verbose=False):
    '''
    Commpute properties for point p projected on quad4 master element.

    Parameters
    ----------
    X : np.ndarray, shape = (12,)
        points of the quad4 element in reference configuratoin
    p : np.ndarray, shape = (3,)
        point which should be tied onto the master element
    niter_max : int, optional
        number of maximum iterations of the Newton-Raphson iteration.
        Default value: 20
    eps : float, optional
        tolerance for the Newton-Raphson iteration. Default value: 1E-10
    verbose : bool, optional
        flag for verbose behavior. Default value: False

    Returns
    -------
    valid_element : bool
        boolean flag setting, if point lies withinn the master element
    N : ndarray, shape (4,)
        weights of the nodal coordinates
    local_basis : ndarray, shape (3,3)
        local tangential basis. local_basis[:,0] forms the normal vector,
        local_basis[:,1:] the vectors along xi and eta
    xi_vec : ndarray, shape(2,)
        position of p in the local element coordinate system
    '''
    # Newton solver to find the element coordinates xi, eta of point p
    X_mat = X.reshape((-1,3))
    xi, eta = xi_vec = np.array([0.,0.]) # starting point 0
    N = np.array([(-eta + 1)*(-xi + 1)/4,
                  (-eta + 1)*(xi + 1)/4,
                  (eta + 1)*(xi + 1)/4,
                  (eta + 1)*(-xi + 1)/4])
    dN_dxi = np.array([[ eta/4 - 1/4,  xi/4 - 1/4],
                       [-eta/4 + 1/4, -xi/4 - 1/4],
                       [ eta/4 + 1/4,  xi/4 + 1/4],
                       [-eta/4 - 1/4, -xi/4 + 1/4]])
    res = X_mat.T @ N - p
    jac = X_mat.T @ dN_dxi

    n_iter = 0
    while res.T @ jac @ jac.T @ res > eps and niter_max > n_iter:
        delta_xi_vec = sp.linalg.solve(jac.T @ jac, - jac.T @ res)
        xi_vec += delta_xi_vec
        xi, eta = xi_vec
        N = np.array([(-eta + 1)*(-xi + 1)/4,
                      (-eta + 1)*(xi + 1)/4,
                      (eta + 1)*(xi + 1)/4,
                      (eta + 1)*(-xi + 1)/4])
        dN_dxi = np.array([[ eta/4 - 1/4,  xi/4 - 1/4],
                           [-eta/4 + 1/4, -xi/4 - 1/4],
                           [ eta/4 + 1/4,  xi/4 + 1/4],
                           [-eta/4 - 1/4, -xi/4 + 1/4]])
        jac = X_mat.T @ dN_dxi
        res = X_mat.T @ N - p
        n_iter += 1
    if np.min(N) >= 0:
        valid_element = True
    else:
        valid_element = False
    if verbose:
        print('Projection of point to Quad4',
              'in {0:1d} iterations'.format(n_iter))
    normal = np.cross(jac[:,0], jac[:,1])
    normal /= np.sqrt(normal @ normal)
    e1 = jac[:,0]
    e1 /= np.sqrt(e1 @ e1)
    e2 = np.cross(normal, e1)
    rot_basis = np.zeros((3,3))
    rot_basis[:,0] = normal
    rot_basis[:,1] = e1
    rot_basis[:,2] = e2
    return valid_element, N, rot_basis, xi_vec

#%%

input_file = amfe.amfe_dir('meshes/gmsh/plate_mesh_tying.msh')
output_file = amfe.amfe_dir('results/mesh_tying/plate_mesh_tying')

my_mesh = amfe.Mesh()
my_mesh.import_msh(input_file)

my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)

my_mesh.load_group_to_mesh(1, my_material, 'phys_group') # box
my_mesh.load_group_to_mesh(2, my_material, 'phys_group') # platform

my_mesh.save_mesh_xdmf(output_file + '_all')

#%%
ndim = 3

df = my_mesh.el_df

master_elements = df[df['phys_group']  == 5]
slave_elements = df[df['phys_group']  == 6]

slave_nodes = np.unique(slave_elements.iloc[:,my_mesh.node_idx:].values)
slave_nodes = np.array(slave_nodes[np.isfinite(slave_nodes)], dtype=int)

master_ele = master_elements.iloc[:,my_mesh.node_idx:my_mesh.node_idx + 4].values

ele_center_points = np.zeros((master_ele.shape[0], ndim))

for i, element in enumerate(master_ele):
    node_xyz = my_mesh.nodes[element]
    ele_center_points[i,:] = node_xyz.mean(axis=0)

distances = sp.spatial.distance.cdist(my_mesh.nodes[slave_nodes], ele_center_points)
element_ranking = np.argsort(distances, axis=1)

#%%
for i, ranking_table in enumerate(element_ranking):
    slave_node_idx = slave_nodes[i]
    for ele_index in ranking_table: # Go through the suggestions of the heuristics
        master_element = master_ele[ele_index]
        X = my_mesh.nodes[master_element]
        slave_node = my_mesh.nodes[slave_node_idx]
        valid, N, local_basis, xi_vec = proj_quad4(X, slave_node)
        print(valid)
        if valid:
            break

#%%
