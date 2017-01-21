"""
Mesh tying module allowing for master-slave nodal interaction
"""
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def proj_quad4(X, p, eps=1E-10, niter_max=20, verbose=False):
    '''
    Compute the shape function weights and the element coordinates for the
    given point p relativ to the element with nodal coordinates X
    '''
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
    if verbose:
        print('Projection of point to Quad4',
              'in {0:1d} iterations'.format(n_iter))
    return N, xi_vec

#%%
X_quad4 = np.array([0,0,0,1,0,0,1,1,0,0,1,0], dtype=float)
rand = np.random.rand(12)*0.4

p = np.array([1/2, 1/2, 0])

x = X_quad4 + rand
N, xi_vec = proj_quad4(x, p, verbose=True)

print(xi_vec, N)

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x[0::3], x[1::3], x[2::3])
ax.scatter(p[0], p[1], p[2])