"""
Mesh tying module allowing for master-slave nodal interaction
"""
import numpy as np
import scipy as sp
from scipy import spatial


def quad4_shape_functions(xi_vec):
    xi, eta = xi_vec
    N = np.array([(-eta + 1)*(-xi + 1)/4,
                  (-eta + 1)*(xi + 1)/4,
                  (eta + 1)*(xi + 1)/4,
                  (eta + 1)*(-xi + 1)/4])
    dN_dxi = np.array([[ eta/4 - 1/4,  xi/4 - 1/4],
                       [-eta/4 + 1/4, -xi/4 - 1/4],
                       [ eta/4 + 1/4,  xi/4 + 1/4],
                       [-eta/4 - 1/4, -xi/4 + 1/4]])
    valid = xi >= -1 and xi <= 1 and eta >= -1 and eta <= 1
    return N, dN_dxi, valid


def quad8_shape_functions(xi_vec):
    xi, eta = xi_vec
    N = np.array([(-eta + 1)*(-xi + 1)*(-eta - xi - 1)/4,
                  (-eta + 1)*(xi + 1)*(-eta + xi - 1)/4,
                  (eta + 1)*(xi + 1)*(eta + xi - 1)/4,
                  (eta + 1)*(-xi + 1)*(eta - xi - 1)/4,
                  (-eta + 1)*(-xi**2 + 1)/2,
                  (-eta**2 + 1)*(xi + 1)/2,
                  (eta + 1)*(-xi**2 + 1)/2,
                  (-eta**2 + 1)*(-xi + 1)/2])

    dN_dxi = np.array([
        [-(eta - 1)*(eta + 2*xi)/4, -(2*eta + xi)*(xi - 1)/4],
        [ (eta - 1)*(eta - 2*xi)/4,  (2*eta - xi)*(xi + 1)/4],
        [ (eta + 1)*(eta + 2*xi)/4,  (2*eta + xi)*(xi + 1)/4],
        [-(eta + 1)*(eta - 2*xi)/4, -(2*eta - xi)*(xi - 1)/4],
        [             xi*(eta - 1),            xi**2/2 - 1/2],
        [          -eta**2/2 + 1/2,            -eta*(xi + 1)],
        [            -xi*(eta + 1),           -xi**2/2 + 1/2],
        [           eta**2/2 - 1/2,             eta*(xi - 1)]])

    valid = xi >= -1 and xi <= 1 and eta >= -1 and eta <= 1
    return N, dN_dxi, valid

def tri3_shape_functions(xi_vec):
    L1, L2 = xi_vec
    L3 = 1 - L1 - L2
    N = np.array([L1, L2, L3])
    dN_dxi = np.array([[ 1,  0],
                       [ 0,  1],
                       [-1, -1]])
    valid = L1 >= 0 and L1 <= 1 and L2 >= 0 and L2 <= 1 and L3 >= 0 and L3 <= 1
    return N, dN_dxi, valid

def tri6_shape_functions(xi_vec):
    L1, L2 = xi_vec
    L3 = 1 - L1 - L2
    N = np.array([L1*(2*L1 - 1),
                  L2*(2*L2 - 1),
                  L3*(2*L3 - 1),
                  4*L1*L2,
                  4*L2*L3,
                  4*L1*L3])

    dN_dxi = np.array([[    4.0*L1 - 1.0,                0],
                       [               0,     4.0*L2 - 1.0],
                       [   -4.0*L3 + 1.0,    -4.0*L3 + 1.0],
                       [          4.0*L2,           4.0*L1],
                       [         -4.0*L2, -4.0*L2 + 4.0*L3],
                       [-4.0*L1 + 4.0*L3,          -4.0*L1]])

    valid = L1 >= 0 and L1 <= 1 and L2 >= 0 and L2 <= 1 and L3 >= 0 and L3 <= 1

    return N, dN_dxi, valid

shape_function_dict = {'Quad4' : quad4_shape_functions,
                       'Quad8' : quad8_shape_functions,
                       'Tri3' : tri3_shape_functions,
                       'Tri6' : tri6_shape_functions,
                      }


def proj_point_to_element(X, p, ele_type, niter_max=20, eps=1E-10,
                          verbose=False):
    '''
    Compute local element coordinates and weights for for point p projected on
    an element.

    This function is heavily used to project the slave nodes on master elements.

    Parameters
    ----------
    X : np.ndarray, shape = (12,)
        points of the quad4 element in reference configuratoin
    p : np.ndarray, shape = (3,)
        point which should be tied onto the master element
    ele_type : str {'Quad4', 'Quad8', 'Tri3', 'Tri6'}
        element type on which is projected
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
    xi : ndarray, shape(2,)
        position of p in the local element coordinate system

    '''
    # Newton solver to find the element coordinates xi, eta of point p
    X_mat = X.reshape((-1,3))
    xi = np.array([0.,0.]) # starting point 0
    shape_function = shape_function_dict[ele_type]
    N, dN_dxi, valid_element = shape_function(xi)
    res = X_mat.T @ N - p
    jac = X_mat.T @ dN_dxi

    n_iter = 0
    while res.T @ jac @ jac.T @ res > eps and niter_max > n_iter:
        delta_xi = sp.linalg.solve(jac.T @ jac, - jac.T @ res)
        xi += delta_xi
        N, dN_dxi, valid_element = shape_function(xi)
        jac = X_mat.T @ dN_dxi
        res = X_mat.T @ N - p
        n_iter += 1

    if verbose:
        print('Projection of point to', ele_type,
              'in {0:1d} iterations.'.format(n_iter),
              'The element is a valid element: ', valid_element)

    normal = np.cross(jac[:,0], jac[:,1])
    normal /= np.sqrt(normal @ normal)
    e1 = jac[:,0]
    e1 /= np.sqrt(e1 @ e1)
    e2 = np.cross(normal, e1)
    rot_basis = np.zeros((3,3))
    rot_basis[:,0] = normal
    rot_basis[:,1] = e1
    rot_basis[:,2] = e2
    return valid_element, N, rot_basis, xi


def master_slave_constraint(master_nodes, master_obj, slave_nodes, nodes,
                            tying_type = 'fixed', robustness=4, verbose=True):
    '''
    Add a master-slave relationship to the given mesh.

    robustness : int
        factor indicating, how many elements are considered for contact search.
    '''
    no_of_nodes, ndim = nodes.shape
    ele_center_points = np.zeros((len(master_nodes), ndim))
    master_ele_nodes = []

    # compute the element center points
    for i, element_raw in enumerate(master_nodes):
        element = np.array(element_raw[np.isfinite(element_raw)], dtype=int)
        master_ele_nodes.append(element)
        node_xyz = nodes[element]
        ele_center_points[i,:] = node_xyz.mean(axis=0)

    distances = sp.spatial.distance.cdist(nodes[slave_nodes], ele_center_points)
    element_ranking = np.argsort(distances, axis=1)

    slave_dofs = []
    # B = np.eye(no_of_dofs)

    row = []
    col = []
    val = []

    # loop over all slave points
    for i, ranking_table in enumerate(element_ranking):
        slave_node_idx = slave_nodes[i]
        # Go through the suggestions of the heuristics and compute weights
        for ele_index in ranking_table[:robustness]:
            master_nodes_idx = master_ele_nodes[ele_index]
            X = nodes[master_nodes_idx]
            slave_node = nodes[slave_node_idx]
            ele_type = master_obj[ele_index]
            valid, N, local_basis, xi = proj_point_to_element(X, slave_node,
                                                              ele_type=ele_type)
            if valid:
                break
            if verbose:
                print('A non valid master element was chosen.')
        # Now build the B-matrix or something like that...
        if tying_type == 'fixed' and valid:

            for dim in range(ndim):
                master_nodes_dofs = master_nodes_idx * ndim + dim
                slave_node_dof = slave_node_idx * ndim + dim

                # B[slave_node_dof, slave_node_dof] -= 1 # remove diagonal entry
                row.append(slave_node_dof)
                col.append(slave_node_dof)
                val.append(-1)

                # B[slave_node_dof, master_nodes_dofs] += N
                row.extend(np.ones_like(master_nodes_dofs) * slave_node_dof)
                col.extend(master_nodes_dofs)
                val.extend(N)

                slave_dofs.append(slave_node_dof)

        elif tying_type == 'slide' and valid:
            normal = local_basis[:,0]
            slave_node_dofs = np.arange(ndim) + slave_node_idx * ndim
            # B[slave_node_dofs, slave_node_dofs] -= 1
            row.extend(slave_node_dofs)
            col.extend(slave_node_dofs)
            val.extend(- np.ones_like(slave_node_dofs))

            # B[np.ix_(slave_node_dofs, slave_node_dofs[1:])] += local_basis[:,1:]
            row.extend(np.ravel(slave_node_dofs.reshape(ndim, 1)
                                @ np.ones((1,ndim-1), dtype=int) ))
            col.extend(np.ravel(np.ones((ndim,1), dtype=int)
                                @ slave_node_dofs[1:].reshape(1,-1)))
            val.extend(np.ravel(local_basis[:,1:]))

            # delete the first element of the slave_node_dofs
            slave_dofs.append(slave_node_dofs[0])

            # Handling for the normal force constraing
            for dim in range(ndim):
                master_nodes_dofs = master_nodes_idx * ndim + dim
                slave_node_dof = slave_node_idx * ndim + dim
                # B[slave_node_dof, master_nodes_dofs] += N * normal[dim]
                row.extend(np.ones_like(master_nodes_dofs) * slave_node_dof)
                col.extend(master_nodes_dofs)
                val.extend(N * normal[dim])
        elif not valid and verbose:
            print('The slave node is not associated to a master element.')
        elif verbose:
            print("I don't know the mesh tying type", tying_type)

    return slave_dofs, row, col, val
