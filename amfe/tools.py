"""
A collection of tools which to not fit to one topic of the other modules.

Some tools here might be experimental.
"""

__all__ = ['node2total', 'total2node', 'inherit_docs', 'read_hbmat',
           'append_interactively', 'matshow_3d', 'amfe_dir', 'h5_read_u',
           'test']

import os
import numpy as np
import scipy as sp
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def node2total(node_index, coordinate_index, ndof_node=2):
    '''
    Converts the node index and the corresponding coordinate index to the index
    of the total dof.

    Parameters
    ----------
    node_index : int
        Index of the node as shown in tools like paraview
    coordinate_index: int
        Index of the coordinate; 0 if it's x, 1 if it's y etc.
    ndof_node: int, optional
        Number of degrees of freedom per node

    Returns
    -------
    total_index : int
        Index of the total dof

    '''
    if coordinate_index >= ndof_node:
        raise ValueError('coordinate index is greater than dof per node.')
    return node_index*ndof_node + coordinate_index

def total2node(total_index, ndof_node=2):
    '''
    Converts the total index in the global dofs to the coordinate index and the
    index fo the coordinate.

    Parameters
    ----------
    total_index : int
        Index of the total dof
    ndof_node: int, optional
        Number of degrees of freedom per node

    Returns
    -------
    node_index : int
        Index of the node as shown in tools like paraview
    coordinate_index : int
        Index of the coordinate; 0 if it's x, 1 if it's y etc.

    '''
    return total_index // ndof_node, total_index % ndof_node


def inherit_docs(cls):
    '''
    Decorator function for inheriting the docs of a class to the subclass.
    '''
    for name, func in vars(cls).items():
        if not func.__doc__:
            print(func, 'needs doc')
            for parent in cls.__bases__:
                parfunc = getattr(parent, name)
                if parfunc and getattr(parfunc, '__doc__', None):
                    func.__doc__ = parfunc.__doc__
                    break
    return cls

def read_hbmat(filename):
    '''
    Reads the hbmat file and returns it in the csc format.

    Parameters
    ----------
    filename : string
        string of the filename

    Returns
    -------
    matrix : sp.sparse.csc_matrix
        matrix which is saved in harwell-boeing format

    Notes
    ----_
    Information on the Harwell Boeing format:
    http://people.sc.fsu.edu/~jburkardt/data/hb/hb.html

    When the hbmat file is exported as an ASCII-file, the truncation of the
    numerical values can cause issues, for example

    - eigenvalues change
    - zero eigenvalues vanish
    - stiffness matrix becomes indefinite
    - etc.

    Thus do not trust matrices which are imported with this method. The method
    is correct, but the truncation error in the hbmat file of the floating
    point digits might cause some issues.

    '''
    with open(filename, 'r') as infile:
        matrix_data = infile.read().splitlines()

    # Analsysis of further line indices
    n_total, n_indptr, n_indices, n_data, n_rhs = map(int, matrix_data[1].split())
    matrix_keys, n_rows, n_cols, _, _ = matrix_data[2].split()

    n_rows, n_cols = int(n_rows), int(n_cols)

    symmetric = False
    if matrix_keys[1] == 'S':
        symmetric = True

    idx_0 = 4
    if n_rhs > 0:
        idx_0 += 1

    indptr = sp.zeros(n_indptr, dtype=int)
    indices = sp.zeros(n_indices, dtype=int)
    data = sp.zeros(n_data)

    indptr[:] = list(map(int, matrix_data[idx_0 : idx_0 + n_indptr]))
    indices[:] = list(map(int, matrix_data[idx_0 + n_indptr :
                                           idx_0 + n_indptr + n_indices]))
    # consider the fortran convention with D instead of E in double precison floats
    data[:] = [float(x.replace('D', 'E')) for x in
               matrix_data[idx_0 + n_indptr + n_indices : ]]

    # take care of the indexing notation of fortran
    indptr -= 1
    indices -= 1

    matrix = sp.sparse.csc_matrix((data, indices, indptr), shape=(n_rows, n_cols))
    if symmetric:
        diagonal = matrix.diagonal()
        matrix = matrix + matrix.T
        matrix.setdiag(diagonal)
    return matrix


def append_interactively(filename):
    '''
    Open an input dialog for interactively appending a string to a filename.

    This filename function should make it easy to save output files from
    numerical experiments containing a time stamp with an additonal tag
    requested at time of saving.

    Parameters
    ----------
    filename : string
        filename path, e.g. for saving

    Returns
    -------
    filename : string
        filename path with additional stuff, maybe added for convenience or
        better understanding
    '''
    print('The filename is:', filename)
    raw = input('You can now add a string to the output file name:\n')

    if raw is not '':
        string = '_' + raw.replace(' ', '_')
    else:
        string = ''

    return filename + string

def matshow_3d(A, thickness=0.8, cmap=mpl.cm.plasma, alpha=1.0):
    '''
    Show a matrix as bar-plot using matplotlib.bar3d plotting tools similar to
    `pyplot.matshow`.

    Parameters
    ----------
    A : ndarray
        Array to be plotted
    thickness : float, optional
        thickness of the bar. Default: 0.8
    cmap : matplotlib.cm function, optional
        Colormap-function of matplotlib. Default. mpl.cm.jet
    alpha : float
        alpha channel value (transparency): alpha=1.0 is not transparent at all,
        alpha=0.0 is full transparent and thus invisible.

    Returns
    -------
    barplot : instance of mpl_toolkits.mplot3d.art3d.Poly3DCollection

    See Also
    --------
    matplotlib.pyplot.matshow

    '''
    xdim, ydim = A.shape
    fig = plt.figure()
    ax = Axes3D(fig)
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
    xx = xx.flatten() + 1 - thickness/2
    yy = yy.flatten() + 1 - thickness/2
    zz = np.zeros_like(xx)
    dx = np.ones_like(xx)*thickness
    dy = np.ones_like(xx)*thickness
    dz = A.flatten()
    colors = cmap(dz)
    barplot = ax.bar3d(xx, yy, zz, dx, dy, dz, color=colors, alpha=alpha)
    # fig.colorbar(barplot)
    return barplot


def reorder_sparse_matrix(A):
    '''
    Reorder the sparse matrix A such that the bandwidth of the matrix is
    minimized using the Cuthill–McKee (RCM) algorithm.

    Parameters
    ----------
    A : CSR or CSC sprarse symmetric matrix
        Sparse and symmetric matrix

    Returns
    -------
    A_new : CSR or CSC sparse symmetric matrix
        reordered sparse and symmetric matrix
    perm : ndarray
        vector of row and column permutation

    References
    ----------
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric Matrices",
    ACM '69 Proceedings of the 1969 24th national conference, (1969).

    '''
    perm = sp.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    return A[perm,:][:,perm], perm

def amfe_dir(filename=''):
    '''
    Return the absolute path of the filename given relative to the amfe
    directory.

    Returns
    -------
    dir : string, optional
        string of the filename inside the AMFE-directory. Default value is '',
        so the AMFE-directory is returned.

    '''
    amfe_abs_path = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(amfe_abs_path, filename.lstrip('/'))


def h5_read_u(h5filename):
    '''
    Extract the displacement field of a given hdf5-file.

    Parameters
    ---------
    h5filename : str
        Full filename (with e.g. .hdf5 ending) of the hdf5 file.

    Returns
    -------
    u_constr : ndarray
        Displacement time series of the dofs with constraints implied.
        Shape is (ndof_constr, no_of_timesteps), i.e. u_constr[:,0] is the
        first timestep.
    u_unconstr : ndarray
        Displacement time series of the dofs without constraints. I.e. the
        dofs are as in the mesh file. Shape is (ndof_unconstr, no_of_timesteps).
    T : ndarray
        Time. Shape is (no_of_timesteps,).

    '''
    with h5py.File(h5filename, 'r') as f:
        u_full = f['time_vals/Displacement'][:]
        T = f['time'][:]
        h5_mat = f['mesh/bmat']
        csr_list = []
        for par in ('data', 'indices', 'indptr', 'shape'):
            csr_list.append(h5_mat[par][:])

    bmat = sp.sparse.csr_matrix(tuple(csr_list[:3]), shape=tuple(csr_list[3]))

    # If the problem is 2D but exported to 3D, u_full has to be reduced.
    ndof_unconstr, ndof_constr = bmat.shape
    if ndof_unconstr == u_full.shape[0]: # this is the 3D-case
        pass
    elif ndof_unconstr*3//2 == u_full.shape[0]: # problem is 2D but u_full is 3D
        mask = np.ones_like(u_full[:,0], dtype=bool)
        mask[2::3] = False
        u_full = u_full[mask, :]
    return bmat.T @ u_full, u_full, T


def test(*args, **kwargs):
    '''
    Run all tests for AMfe.
    '''
    import nose
    nose.main(*args, **kwargs)
