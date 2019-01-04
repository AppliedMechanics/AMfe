"""
TODO: Write introduction to ECSW
"""

import logging
from copy import deepcopy, copy

import numpy as np
from scipy.linalg import solve as linsolve
from scipy.sparse import csc_matrix

from amfe.assembly import EcswAssembly


__all__ = ['sparse_nnls',
           'assemble_g_and_b',
           'reduce_with_ecsw']


def sparse_nnls(G, b, tau, conv_stats=True):
    r"""
    Run the sparse NNLS-solver in order to find a sparse vector xi satisfying

    .. math::
        || G \xi - b ||_2 \leq \tau ||b||_2 \quad\text{with}\quad \min||\xi||_0

    Parameters
    ----------
    G : ndarray, shape: (n*m, no_of_elements)
        force contribution matrix
    b : ndarray, shape: (n*m)
        force contribution vector
    tau : float
        tolerance
    conv_stats : bool
        Flag for setting, that more detailed output is produced with
        convergence information.

    Returns
    -------
    x : csc_matrix
        sparse vector containing the weights
    stats : ndarray
        Infos about the convergence of the system. The first column shows the
        size of the active set, the second column the residual. If conv_info is
        set to False, an empty array is returned.

    References
    ----------
    .. [1]  C. L. Lawson and R. J. Hanson. Solving least squares problems,
            volume 15. SIAM, 1995.

    .. [2]  T. Chapman, P. Avery, P. Collins, and C. Farhat. Accelerated mesh
            sampling for the hyper reduction of nonlinear computational models.
            International Journal for Numerical Methods in Engineering, 2016.

    """
    no_of_elements = G.shape[1]
    norm_b = np.linalg.norm(b)
    r = b

    xi = np.zeros(no_of_elements) # the resulting vector
    zeta = np.zeros(no_of_elements) # the trial vector which is iterated over

    # Boolean active set; allows quick and easys indexing through masking with
    # high performance at the same time
    active_set = np.zeros(no_of_elements, dtype=bool)

    stats = []
    while np.linalg.norm(r) > tau * norm_b:
        mu = G.T @ r
        idx = np.argmax(mu)
        if active_set[idx] == True:
            raise RuntimeError('snnls: The index has {} has already been added and is considered to be the best again.')
        active_set[idx] = True
        print('Added element {}'.format(idx))
        while True:
            # Trial vector zeta is solved for the sparse solution
            zeta[~active_set] = 0.0
            G_red = G[:, active_set]
            zeta[active_set] = linsolve(G_red.T @ G_red, G_red.T @ b)

            # check, if gathered solution is full positive
            if np.min(zeta[active_set]) >= 0.0:
                xi[:] = zeta[:]
                break
            # remove the negative elements from the active set
            # Get all elements which violate the constraint, i.e. are in the
            # active set and are smaller than zero
            mask = np.logical_and(zeta <= 0.0, active_set)

            ele_const = np.argmin(xi[mask] / (xi[mask] - zeta[mask]))
            const_idx = np.where(mask)[0][ele_const]
            print('Remove element {} '.format(const_idx) +
                   'violating the constraint.')
            # Amplify xi with the difference of zeta and xi such, that the
            # largest mismatching negative point becomes zero.
            alpha = np.min(xi[mask] / (xi[mask] - zeta[mask]))
            xi += alpha * (zeta - xi)
            # Set active set manually as otherwise floating point roundoff
            # errors are not considered.
            # active_set = xi != 0
            active_set[const_idx] = False

        r = b - G[:, active_set] @ xi[active_set]
        logger = logging.getLogger('amfe.hyper_red.ecsw.snnls')
        logger.debug("snnls: residual {} No of active elements: {}".format(np.linalg.norm(r), len(np.where(xi)[0])))
        if conv_stats:
            stats.append((len(np.where(xi)[0]), np.linalg.norm(r)))

    # sp.optimize.nnls(A, b)
    indices = np.where(xi)[0]  # remove the nasty tuple from np.where()
    xi_red = xi[active_set]
    indptr = np.array([0, len(xi_red)])
    x = csc_matrix((xi_red, indices, indptr), shape=(G.shape[1], 1))
    if conv_stats and not stats:
        stats.append((0, np.linalg.norm(r)))
    stats = np.array(stats)
    return x, stats


def assemble_g_and_b(component, S, timesteps=None):
    """
    Assembles the element contribution matrix G for the given snapshots S.

    This function is needed for cubature bases Hyper reduction methods
    like the ECSW.

    Parameters
    ----------
    component : amfe.MeshComponent
        amfe.Component, if a reduction basis should be used, it should already
        be the component that is reduced by this reduction basis
    S : ndarray, shape (no_of_dofs, no_of_snapshots)
        Snapshots gathered as column vectors.
    timesteps : ndarray, shape(no_of_snapshots)
        optional, the timesteps of where the snapshots have been generated can be passed,
        this is important for systems with certain constraints

    Returns
    -------
    G : ndarray, shape (n*m, no_of_elements)
        Contribution matrix of internal forces. The columns form the
        internal force contributions on the basis V for the m snapshots
        gathered in S.
    b : ndarray, shape (n*m, )
        summed force contribution

    Note
    ----
    This assembly works on constrained variables
    """
    # Check the raw dimension
    # Currently not applicable
    # assert(component.no_of_dofs == S.shape[0])

    logger = logging.getLogger('amfe.hyper_red.ecsw.assemble_g_and_b')

    no_of_dofs, no_of_snapshots = S.shape

    no_of_elements = component.no_of_elements
    logger.info('Start building large selection matrix G. In total {0:d} elements are treated:'.format(
                  no_of_elements))

    G = np.zeros((no_of_dofs*no_of_snapshots, no_of_elements))

    # Temporarily replace Assembly of component:
    old_assembly = component.assembly
    g_assembly = EcswAssembly([], [])
    component.assembly = g_assembly

    # Weight only one element by one
    g_assembly.weights = 1

    # check if timesteps are None:
    if timesteps is None:
        timesteps = np.zeros(no_of_snapshots)

    # loop over all elements
    for element_no in range(no_of_elements):
        # Change nonzero weighted elements to current element
        g_assembly.indices = [element_no]

        logger.debug('Assemble element {:10d} / {:10d}'.format(element_no+1, no_of_elements))
        # loop over all snapshots
        for snapshot_number, (snapshot_vector, t) in enumerate((S.T, timesteps)):
            G[snapshot_number*no_of_dofs:(snapshot_number+1)*no_of_dofs, element_no] = component.f_int(snapshot_vector,
                                                                                                       t)

    b = np.sum(G, axis=1)

    # reset assembly
    component.assembly = old_assembly
    return G, b


def reduce_with_ecsw(component, S, timesteps, tau=0.001, copymode='overwrite',
                                   conv_stats=False, tagname=None):
    """
    Reduce the given MeshComponent

    Parameters
    ----------
    component : instance of MeshComponent
        MeshComponent
    S : ndarray, shape (no_of_dofs, no_of_snapshots)
        Snapshots
    copymode : str {'ovewrite', 'shallow', 'deep'}
        Select if the component shall be verwritten, shallow copied or deepcopied
    conv_stats : bool
        Flag if conv_stats shall be collected
    tagname : str
        optional, if a string is given the weights are written into the mesh as tag values
        the tagname is the given string

    Returns
    -------
    reduced_system : instance of MeshComponent
        Reduced system with same properties of the passed Mesh Component
        but with ECSW reduced mesh
    stats : list
        Information about hyperreduction convergence
    """

    # If overwrite use existent component, else create new one by copying
    if copymode == 'overwrite':
        hyperreduced_component = component
    elif copymode == 'shallow':
        hyperreduced_component = copy(component)
    elif copymode == 'deep':
        hyperreduced_component = deepcopy(component)
    else:
        raise ValueError("copymode must be 'overwrite', 'shallow' or 'deep', got {}".format(copymode))

    # Create G and b from snapshots:
    G, b = assemble_g_and_b(hyperreduced_component, S, timesteps)

    # Calculate indices and weights
    x, stats = sparse_nnls(G, b, tau, conv_stats)
    indices = x.indices
    weights = x.data

    # Create new assembly
    ecswassembly = EcswAssembly(weights, indices)

    # Assign new assembly got reduced_component
    hyperreduced_component.assembly = ecswassembly

    # create a new tag for ecsw weights
    if tagname is not None:
        # TODO: THIS IS TEMPORARY: AND BAD STYLE
        elementids = component._ele_obj_df.index.levels[1][indices].values
        tag_value_dict = {weight: [elementid] for weight, elementid in zip(weights, elementids)}
        component.mesh.insert_tag(tagname, tag_value_dict)

    if conv_stats:
        return hyperreduced_component, stats
    else:
        return hyperreduced_component
