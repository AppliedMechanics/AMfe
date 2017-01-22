"""
Mesh tying module allowing for master-slave nodal interaction
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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


def point_distances(a, b, ndim=3):
    '''
    Find the nearest neighbor of point cloud a to point cloud b

    Parameters
    ----------
    a : ndarray
        point coordinate array
    b : ndarray
        point coordinate array
    ndim : int, optional
        dimension of the problem. Default value: 3

    Returns
    -------
    A : ndarray
        Distance array of a to b. A[:,i] gives the distances of all points in a
        to point b[i]
    '''
    A = a.reshape((-1,1,ndim))
    B = b.reshape((1,-1,ndim))
    dist_square = np.einsum('ijk->ij', (A - B)**2)
    return np.sqrt(dist_square)


#%%
X_quad4 = np.array([0,0,0,1,0,0,1,1,0,0,1,0], dtype=float)
rand = np.random.rand(12)*0.4

p = np.array([1/2, 1, 0])

x = X_quad4 + rand
valid, N, A, xi_vec = proj_quad4(x, p, verbose=True)

print(xi_vec, N)

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[0::3], x[1::3], x[2::3])
ax.scatter(p[0], p[1], p[2])

#%% Big asessment
#%%time
for i in range(10000):
    X_quad4 = np.array([0,0,0,1,0,0,1,1,0,0,1,0], dtype=float)
    rand = np.random.rand(12)*0.4
    p = np.array([1/2, 1/2, 0])
    x = X_quad4 + rand
    valid, N, A, xi_vec = proj_quad4(x, p, verbose=False)

#%%



#%%
a = np.zeros((10,3))
a[:,0] = np.arange(10)
b = a


dist = point_distances(a,b)
# this is the way how to move along the best elements
idxs = dist.argsort()

