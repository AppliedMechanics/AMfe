# Copyright (c) 2017, Lehrstuhl fuer Angewandte Mechanik, Technische
# Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
A collection of tools which to not fit to one topic of the other modules.

Some tools here might be experimental.
"""


__all__ = [
    'read_hbmat',
    'append_interactively',
    'matshow_3d',
    'test',
    'reorder_sparse_matrix',
    'eggtimer',
    'compute_relative_error',
    'principal_angles',
    'query_yes_no',
    'compare_signals',
    'time_measured',
]

import numpy as np
import scipy as sp
from scipy import linalg
from scipy import interpolate
import time
import subprocess
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from amfe.linalg import signal_norm, vector_norm
from amfe.structural_dynamics import modal_assurance


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
    Information on the Harwell Boeing format: http://people.sc.fsu.edu/~jburkardt/data/hb/hb.html

    When the hbmat file is exported as an ASCII-file, the truncation of the numerical values can cause issues, for
    example

    - eigenvalues change
    - zero eigenvalues vanish
    - stiffness matrix becomes indefinite
    - etc.

    Thus do not trust matrices which are imported with this method. The method is correct, but the truncation error in
    the hbmat file of the floating point digits might cause some issues.
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

    This filename function should make it easy to save output files from numerical experiments containing a time stamp
    with an additonal tag requested at time of saving.

    Parameters
    ----------
    filename : string
        filename path, e.g. for saving

    Returns
    -------
    filename : string
        filename path with additional stuff, maybe added for convenience or better understanding
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
    Show a matrix as bar-plot using matplotlib.bar3d plotting tools similar to `pyplot.matshow`.

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
    Reorder the sparse matrix A such that the bandwidth of the matrix is minimized using the Cuthill–McKee (RCM)
    algorithm.

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
    E. Cuthill and J. McKee, "Reducing the Bandwidth of Sparse Symmetric Matrices", ACM '69 Proceedings of the 1969
    24th national conference, (1969).
    '''

    perm = sp.sparse.csgraph.reverse_cuthill_mckee(A, symmetric_mode=True)
    return A[perm,:][:,perm], perm


def compute_relative_error(u_ref, u_red, T_ref, T_red, M=None):
    """
    Compute the farhat error of two given displacement sets.

    Parameters
    ----------
    red_file : str
        Filename of the reduced or to be investigated file.
    ref_file : str
        Filename of the reference file.
    M : array-like, optional
        mass matrix of the given system. If None, no mass matrix is used.
        Default: None.

    Returns
    -------
    err : float
        Error.

    Note
    ----

    .. math::
        ER = \frac{\sqrt{\sum\limits_{t\in T} \Delta u(t)^T M \Delta u(t)}}{
                   \sqrt{\sum\limits_{t\in T} u_{ref}(t)^T M u_{ref}(t)}}
    """

    if len(T_red) < len(T_ref): # The time integration has aborted
        return np.nan
    delta_u = u_red - u_ref
    # take the inner product of the columns; Use the mass matrix, if necesary
    if M is None:
        err_sq = np.einsum('ij,ij->j', delta_u, delta_u)
        ref_sq = np.einsum('ij,ij->j', u_ref, u_ref)
    else:
        err_sq = np.einsum('ij,ij->j', delta_u, M @ delta_u)
        ref_sq = np.einsum('ij,ij->j', u_ref, M @ u_ref)

    err = np.sqrt(np.sum(err_sq)) / np.sqrt(np.sum(ref_sq))
    return err


def principal_angles(V1, V2, unit='deg', method=None, principal_vectors=False):
    """
    Return the principal/subspace angles of span(V1) and span(V2) subspaces of R^n.

    Parameters
    ----------
    V1 : 2darray
        Matrix spanning subspace 1. Dimensions n x r1.
    V2 : 2darray
        Matrix spanning subspace 2. Dimension n x r2.
    unit : {'deg', 'rad', None}, optional
        Unit in which angles are returned. Default is 'deg'.
    method : {None, 'cos', 'sin'}, optional
        Method used for computation of angles:
             - 'cos' for large angles
             - 'sin' for small angles
             - None for all angles (combines both methods).
        Default is None.
    principal_vectors : boolean, optional
        Flag for returning principal vectors. Default is False.

    Returns
    -------
    theta : 1darray
        Vector with principle/subspace angles.
    F1 : 2darray, optional
        Matrix with principal vectors of subspace span(V1). Columns give principal vectors, i.e. F1[:,0] is first
        principal vector of span(V1) associated with principle angle theta[0] and so on. Only returned, if
        principal_vectors=True.
    F2 : 2darray, optional
        Matrix with principal vectors of subspace span(V2). Only returned, if principal_vectors=True.

    References
    ----------
       [1]  A.Bjorck, G.H. Golub (1973): Numerical methods for computing angles between linear subspaces. Mathematics
            of Computation 27(123) 579--594. DOI: 10.2307/2005662.
       [2]  G.H. Golub and C.F. Van Loan (1996): Matrix computations. Volume 3. JHU Press.
       [3]  A.V. Knyazev and M.E. Argentati (2002): Principle angles between subspaces in an A-based scalar product:
            algorithms and perturbation estimates. SIAM Journal on Scientific Computing 23(6) 2009--2041.
            DOI: 10.1137/S1064827500377332.
       [4]  G.H. Golub and C.F. Van Loan (2013): Matrix computations. Volume 4. JHU Press.
       [5]  J.B. Rutzmoser, F.M. Gruber and D.J. Rixen (2015): A comparison on model order reduction techniques for
            geometrically nonlinear systems based on a modal derivative approach using subspace angles. 11th
            International Conference on Engineering Vibration, Ljubljana, Slovenia.
    """

    Q1, __ = linalg.qr(a=V1, mode='economic')
    Q2, __ = linalg.qr(a=V2, mode='economic')

    if method is None:
        U, sigma, VT = linalg.svd(a=Q1.T @ Q2, full_matrices=False)  # cos
        sigma[sigma > 1.0] = 1.0  # cos
        theta = np.arccos(sigma)  # cos, rad
        if principal_vectors:
            F1 = Q1 @ U  # cos
            F2 = Q2 @ VT.T  # cos

        if Q1.shape[1] >= Q2.shape[1]:
            U_sin, sigma_sin, VT_sin = linalg.svd(a=Q2 - Q1@(Q1.T@Q2), full_matrices=False)
        else:
            U_sin, sigma_sin, VT_sin = linalg.svd(a=Q1 - Q2@(Q2.T@Q1), full_matrices=False)
        # TODO: Change flipup(...) and fliplr(...) back to new function flip(..., axis=0/1) after updating test server
        # TODO: U_sin = np.flip(m=U_sin, axis=1)
        U_sin = np.fliplr(m=U_sin)
        # TODO: sigma_sin = np.flip(m=sigma_sin, axis=0)
        sigma_sin = np.flipud(m=sigma_sin)
        # TODO: VT_sin = np.flip(m=VT_sin, axis=0)
        VT_sin = np.flipud(m=VT_sin)
        sigma_sin[sigma_sin > (1.0 - 1.0e-16)] = (1.0 - 1.0e-16)
        theta_sin = np.arcsin(sigma_sin)  # rad
        if principal_vectors:
            if Q1.shape[1] >= Q2.shape[1]:
                F2_sin = Q2@VT_sin.T
                F1_sin = Q1@(Q1.T@F2_sin)/np.sqrt(1 - sigma_sin**2)
            else:
                F1_sin = Q1@VT_sin.T
                F2_sin = Q2@(Q2.T@F1_sin)/np.sqrt(1 - sigma_sin**2)

        index = theta < (np.pi/4)
        sigma[index] = sigma_sin[index]
        theta[index] = theta_sin[index]
        if principal_vectors:
            F1[:, index] = F1_sin[:, index]
            F2[:, index] = F2_sin[:, index]
    elif method == 'cos':
        U, sigma, VT = linalg.svd(a=Q1.T@Q2, full_matrices=False)
        sigma[sigma > 1.0] = 1.0
        theta = np.arccos(sigma)  # rad
        if principal_vectors:
            F1 = Q1@U
            F2 = Q2@VT.T
    elif method == 'sin':
        if Q1.shape[1] >= Q2.shape[1]:
            U, sigma, VT = linalg.svd(a=Q2 - Q1@(Q1.T@Q2), full_matrices=False)
        else:
            U, sigma, VT = linalg.svd(a=Q1 - Q2@(Q2.T@Q1), full_matrices=False)
        # TODO: Change flipup(...) and fliplr(...) back to new function flip(..., axis=0/1) after updating test server
        # TODO: U = np.flip(m=U, axis=1)
        U = np.fliplr(m=U)
        # TODO: sigma = np.flip(m=sigma, axis=0)
        sigma = np.flipud(m=sigma)
        # TODO: VT = np.flip(m=VT, axis=0)
        VT = np.flipud(m=VT)
        sigma[sigma > 1.0] = 1.0
        theta = np.arcsin(sigma)  # rad
        if principal_vectors:
            if Q1.shape[1] >= Q2.shape[1]:
                F2 = Q2@VT.T
                F1 = Q1@(Q1.T@F2)/np.sqrt(1 - sigma**2)
            else:
                F1 = Q1@VT.T
                F2 = Q2@(Q2.T@F1)/np.sqrt(1 - sigma**2)
    else:
        raise ValueError('Invalid method. Chose either None, \'cos\' or \'sin\'.')

    if unit == 'deg':
        theta = np.rad2deg(theta)  # deg
    elif unit == 'rad':
        pass
    elif unit is None:
        theta = sigma
        if method is None:
            print('Warning: Mixed cosine and sine values.')
    else:
        raise ValueError('Invalid unit. Chose either \'deg\', \'rad\' or None.')

    if principal_vectors:
        return theta, F1, F2
    else:
        return theta


def time_measured(fkt):
    """
    Decorator to measure execution time of a function

    It prints out the measured time

    Parameters
    ----------
    fkt : function
        function that shall be measured

    Returns
    -------
    None

    """
    def fkt_wrapper(*args, **kwargs):
        t1 = time.time()
        return_vals = fkt(*args, **kwargs)
        t2 = time.time()
        print("Job needed: {} seconds".format(t2-t1))
        return return_vals
    return fkt_wrapper


def eggtimer(fkt):
    """
    Egg timer for functions which reminds via speech, when the function has terminated.

    The intention of this function is, that the user gets reminded, when longer simulations are over.

    Parameters
    ----------
    fkt : function
        any function

    Returns
    -------
    fkt : function
        function decorated with eggtimer. It reminds you via speech, when the function has terminated.

    Examples
    --------
    Import eggtimer function:

    >>> from amfe.tools import eggtimer

    working directly on function:

    >>> def square(a):
    ...     return a**2
    ...
    >>> timed_square = eggtimer(square)
    >>> timed_square(6)
    36

    working as decorator:

    >>> @eggtimer
    ... def square(a):
    ...     return a**2
    ...
    >>> square(6)
    36
    """

    def fkt_wrapper(*args, **kwargs):
        t1 = time.time()
        return_vals = fkt(*args, **kwargs)
        t2 = time.time()
        speech = '"Your job has finished. ' \
                      + 'It took {0:0.0f} seconds."'.format(t2-t1)
        headline = 'Python job finished'
        text = 'The job you egg-clocked in amfe took {0:0.0f} seconds'.format(t2-t1)

        if sys.platform == 'linux': # Linux
            subprocess.call(['notify-send', headline, text])
            subprocess.call(['speech-dispatcher']) #start speech dispatcher
            subprocess.call(['spd-say', speech])
        elif sys.platform == 'darwin': # OS X
            subprocess.call(['say', '-v', 'Samantha', speech])
            notification_text = 'display notification ' + \
                                '"{0}" with title "{1}"'.format(headline, text)
            subprocess.call(['osascript', '-e', notification_text])
        return return_vals
    return fkt_wrapper


def test(*args, **kwargs):
    """
    Run all tests for AMfe.
    """

    import nose
    nose.main(*args, **kwargs)


def query_yes_no(question, default="yes"):
    '''
    Ask a yes/no question and return their answer.

    Parameters
    ----------
    question: String
        The question to be asked

    default: String "yes" or "no"
        The default answer

    Returns:
    --------
    answer: Boolean
        Answer: True if yes, False if no.
    '''

    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no'.\n")


def compare_signals(x1, t1, x2, t2=None, method='norm', axis=-1, **kwargs):
    """
    Compare signal x2(t2) [slave] with signal x1(t1) [master] using the specified method.

    Parameters
    ----------
    x1 : ndarray
        1D or 2D input array containing samples of signal 1 [master] along specified axis.
    t1 : 1darray
        Input array containing time samples corresponding to x1 [master].
    x2 : ndarray
        1D or 2D input array containing samples of signal 2 [slave] along specified axis.
    t2 : 1darray, optional
        Input array containing time samples corresponding to x2 [slave]. Default None. If not specified (None) t2 = t1
        is used. If specified x2 is linearly interpolated at time samples t1.
    method : {'norm', 'angle', 'mac', 'correlation'}
        Method on which comparison is based:
            - 'norm': Vector norm (v) of signal norm (s) of deviation x2 - x1 normalized w.r.t. x1,
                ||(||x2 - x1||_s/||x1||_s)||_v. Order of signal norm (ord_s) and order of vector norm (ord_v) have to
                be specified in **kwargs. ord_v can additionally be None, then vector of signal norms
                ||x2 - x1||_s/||x1||_s is returned. Defaults are ord_s = 2 and ord_v = None.
            - 'angle': Principle angles between SVDs of signals. Number of considered modes (num <= #dofs) and unit for
                angles (unit) have to be specified in **kwargs. Defaults are num = 13 and unit = 'deg'.
            - 'mac': Modal assurance criterion between SVDs of signals. Number of considered directions (num <= #dofs)
                has to be specified in **kwargs. Default is num = #dofs.
            - 'correlation' : Cross-correlation between signals (based on numpy's correlate function with option
                'full') normalized w.r.t. value of auto-correlation of x1 at full overlap.

    Returns
    -------
    result : float, 1darray(s) or 2darrays
        Result(s) of the comparison:
            - 'norm': Float/1darray with norm(s).
            - 'angle': 1darray with principle angles and 2x 1darrays with singular values of x1 and x2.
            - 'mac': 2darray with mac values and 2x 1darrays with singular values of x1 and x2.
            - 'correlation' : 2darray with cross-correlation and 2darray with auto-correlation of x1.
    """

    # make time samples fit
    if t2 is not None:
        x2 = (interpolate.interp1d(x=t2, y=x2, kind='linear', axis=axis, copy=False, bounds_error=True,
                                   fill_value=None, assume_sorted=True))(t1)

    if method == 'norm':
        # read kwargs
        if 'ord_s' in kwargs:
            ord_s = kwargs['ord_s']
        else:
            print('Attention: No signal norm order was given, setting ord_s = 2.')
            ord_s = 2

        if 'ord_v' in kwargs:
            ord_v = kwargs['ord_v']
        else:
            print('Attention: No vector norm order was given, setting ord_v = None.')
            ord_v = None

        # calculate norm(s)
        norm = signal_norm(x=(x2 - x1), t=t1, dt=None, ord=ord_s, axis=axis)
        norm /= signal_norm(x=x1, t=t1, dt=None, ord=ord_s, axis=axis)
        if ord_v is not None:
            norm = vector_norm(x=norm, ord=ord_v, axis=0, keepdims=False)
        return norm

    elif method == 'angle':
        # read kwargs
        if 'num' in kwargs:
            num = kwargs['num']
        else:
            print('Attention: No number of considered modes was given, setting num = 13.')
            num = 13

        if 'unit' in kwargs:
            unit = kwargs['unit']
        else:
            print('Attention: No unit was given, setting unit = \'deg\'.')
            unit = 'deg'

        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        U1, sigma1, __ = linalg.svd(a=x1, full_matrices=False)
        U2, sigma2, __ = linalg.svd(a=x2, full_matrices=False)
        return principal_angles(V1=U1[:, 0:num], V2=U2[:, 0:num], unit=unit, method=None, principal_vectors=False), \
               sigma1, sigma2

    elif method == 'mac':
        # read kwargs
        if 'num' in kwargs:
            num = kwargs['num']
        else:
            print('Attention: No number of considered modes was given, setting num = #dofs.')
            num = None

        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        U1, sigma1, __ = linalg.svd(a=x1, full_matrices=False)
        U2, sigma2, __ = linalg.svd(a=x2, full_matrices=False)
        return modal_assurance(U=U1[:, 0:num], V=U2[:, 0:num]), sigma1, sigma2

    elif method == 'correlation':
        if axis == 0:
            x1 = x1.T
            x2 = x2.T
        ccor = np.zeros((x1.shape[0], 2*x1.shape[1] - 1))
        acor = np.zeros((x1.shape[0], 2*x1.shape[1] - 1))
        for i in np.arange(0, x1.shape[0]):
            normalization = np.sum(x1[i, :]**2)
            ccor[i, :] = np.correlate(a=x1[i, :], v=x2[i, :], mode='full')/normalization
            acor[i, :] = np.correlate(a=x1[i, :], v=x1[i, :], mode='full')/normalization
        return ccor, acor
    else:
        raise ValueError('Invalid method. Chose either \'norm\', \'angle\', \'mac\' or \'correlation\' with ' \
                         + 'appropriate **kwargs.')

