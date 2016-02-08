# -*- coding: utf-8 -*-

"""
Created on Mon Jun  8 17:06:59 2015

@author: johannesr
"""

import copy
import numpy as np
import scipy as sp
from scipy import linalg

from amfe.mechanical_system import ReducedSystem, QMSystem

def reduce_mechanical_system(mechanical_system, V, overwrite=False):
    '''
    Reduce the given mechanical system with the linear basis V.
    
    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem. 
    V : ndarray
        Reduction Basis for the reduced system
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory 
        intensive for large systems) or not.
    
    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and 
        reduction basis V
        
    Example
    -------

    '''
    
    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
    reduced_sys.__class__ = ReducedSystem
    reduced_sys.V = V.copy()
    reduced_sys.u_red_output = []
    return reduced_sys

def qm_reduce_mechanical_system(mechanical_system, V, Theta, overwrite=False):
    '''
    Reduce the given mechanical system to a QM system with the basis V and the  
    quadratic part Theta. 
        
    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem. 
    V : ndarray
        Reduction Basis for the reduced system
    Theta : ndarray
        Quadratic tensor for the Quadratic manifold. Has to be symmetric with 
        respect to the last two indices and is of shape (n_full, n_red, n_red). 
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory 
        intensive for large systems) or not.
    
    Returns
    -------
    reduced_system : instance of ReducedSystem
        Quadratic Manifold reduced system with same properties of the 
        mechanical system and reduction basis V and Theta
        
    Example
    -------

    
    '''
    # consistency check
    assert(V.shape[-1] == Theta.shape[-1])
    assert(Theta.shape[1] == Theta.shape[2])
    assert(Theta.shape[0] == V.shape[0])
    
    no_of_red_dofs = V.shape[-1]
    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)
        
    reduced_sys.__class__ = QMSystem
    reduced_sys.V = V.copy()
    reduced_sys.Theta = Theta.copy()

    # define internal variables
    reduced_sys.u_red_output = []
    reduced_sys.no_of_red_dofs = no_of_red_dofs
    return reduced_sys
    

SQ_EPS = np.sqrt(np.finfo(float).eps)

def modal_derivative(x_i, x_j, K_func, M, omega_i, h=500*SQ_EPS, verbose=True):
    '''
    Compute the real modal derivative of the given system using Nelson's formulation.

    The modal derivative computed is :math:`\\frac{dx_i}{dx_j}`, i.e. the change of the
    mode x_i when the system is perturbed along x_j.

    Parameters
    ----------
    x_i : ndarray
        modeshape-vector
    x_j : ndarray
        modeshape-vector
    K_func : function
        function for the tangential stiffness matrix; It is evoked by K_func(u)
        with the displacement vector u
    M : ndarray
        mass matrix
    omega_i : float
        eigenfrequency corresponding to the modeshape x_i
    h : float, optional
        step size for the computation of the finite difference scheme. Default 
        value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.

    Returns
    -------
    dx_i / dx_j : ndarray
        The modal derivative dx_i / dx_j with mass consideration

    Note
    ----
    The the vectors x_i and x_j are internally mass normalized;

    See Also
    --------
    static_correction_derivative

    Examples
    --------
    todo

    References
    ---------
    S. R. Idelsohn and A. Cardona. A reduction method for nonlinear structural
    dynamic analysis. Computer Methods in Applied Mechanics and Engineering,
    49(3):253–279, 1985.

    S. R. Idelsohn and A. Cardona. A load-dependent basis for reduced nonlinear
    structural dynamics. Computers & Structures, 20(1):203–210, 1985.


    '''
    # mass normalization
    x_i /= np.sqrt(x_i.dot(M.dot(x_i)))
    x_j /= np.sqrt(x_j.dot(M.dot(x_j)))

    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_x_j = (K_func(x_j*h) - K)/h
    d_omega_2_d_x_i = x_i @ dK_x_j @ x_i
    F_i = (d_omega_2_d_x_i*M - dK_x_j) @ x_i
    K_dyn_i = K - omega_i**2 * M
    # fix the point with the maximum displacement of the vibration mode
    row_index = np.argmax(abs(x_i))
    K_dyn_i[:,row_index], K_dyn_i[row_index,:], K_dyn_i[row_index,row_index] = 0, 0, 1
    F_i[row_index] = 0
    v_i = linalg.solve(K_dyn_i, F_i)
    c_i = - v_i @ M @ x_i
    dx_i_dx_j = v_i + c_i*x_i
    if verbose:
        print('\nComputation of modal derivatives. ')
        print('Influence of the change of the eigenfrequency:', d_omega_2_d_x_i)
        print('The condition number of the problem is', np.linalg.cond(K_dyn_i))
        res = (K - omega_i**2 * M).dot(dx_i_dx_j) - (d_omega_2_d_x_i*M - dK_x_j).dot(x_i)
        print('The residual is', np.sqrt(res.dot(res)),
              ', the relative residual is', np.sqrt(res.dot(res))/np.sqrt(F_i.dot(F_i)))
    return dx_i_dx_j

def modal_derivative_theta(V, omega, K_func, M, h=500*SQ_EPS, verbose=True, 
                           symmetric=True):
    r'''
    Compute the basis theta based on real modal derivatives. 
    
    Parameters
    ----------
    V : ndarray
        array containing the linear basis
    omega : ndarray
        eigenfrequencies of the system in rad/s.
    K_func : function
        function returning the tangential stiffness matrix for a given 
        displacement. Has to work like K = K_func(u). 
    M : ndarray or sparse matrix
        Mass matrix of the system. 
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine 
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True        
        
    Returns
    -------
    Theta : ndarray
        three dimensional array of modal derivatives. Theta[:,i,j] contains 
        the modal derivative 1/2 * dx_i / dx_j. The basis Theta is made symmetric, so 
        that Theta[:,i,j] == Theta[:,j,i]. 
    
    '''
    no_of_dofs = V.shape[0]
    no_of_modes = V.shape[1]
    Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes))

    # Check, if V is mass normalized:
    if not np.allclose(np.eye(no_of_modes), V.T @ M @ V, rtol=1E-5, atol=1E-8):
        Exception('The given modes are not mass normalized!')
        
    K = K_func(np.zeros(no_of_dofs))
    
    for i in range(no_of_modes): # looping over the columns
        x_i = V[:,i]        
        K_dyn_i = K - omega[i]**2 * M
        # fix the point with the maximum displacement of the vibration mode
        fix_idx = np.argmax(abs(x_i))
        K_dyn_i[:,fix_idx], K_dyn_i[fix_idx,:], K_dyn_i[fix_idx, fix_idx] = 0, 0, 1
        # factorization of the dynamic stiffness matrix
        if verbose: 
            print('Factorizing the dynamic stiffness matrix for eigenfrequency',
                  '{0:d} with {1:4.2f} rad/s.'.format(i, omega[i]) )
        LU_object = sp.sparse.linalg.splu(K_dyn_i)

        for j in range(no_of_modes): # looping over the rows
            x_j = V[:,j]
            # finite difference scheme
            dK_x_j = (K_func(x_j*h) - K)/h
            d_omega_2_d_x_i = x_i @ dK_x_j @ x_i
            F_i = (d_omega_2_d_x_i*M - dK_x_j) @ x_i
            F_i[fix_idx] = 0
            v_i = LU_object.solve(F_i)
            c_i = - v_i @ M @ x_i
            Theta[:,i,j] = v_i + c_i*x_i

    if symmetric:
        Theta = 1/4*(Theta + Theta.transpose((0,2,1)))
    return Theta


def static_correction_derivative(x_i, x_j, K_func, h=500*SQ_EPS, verbose=True):
    r'''
    Computes the static correction vectors 
    :math:`\frac{\partial x_i}{\partial x_j}` of the system with a nonlinear 
    force.

    Parameters
    ----------
    x_i : ndarray
        array containing displacement vectors i in the rows. x_i[:,i] is the 
        i-th vector
    x_j : ndarray
        displacement vector j 
    K_func : function
        function for the tangential stiffness matrix to be called in the form 
        K_tangential = K_func(x_j)
    h : float, optional
        step size for the computation of the finite difference scheme. Default 
        value 500 * machine_epsilon
    verbose : bool
        additional output provided; Default value True.

    Returns
    -------
    dx_i_dx_j : ndarray
        static correction derivative (if x_i and x_j is a modal vector it's 
        the modal derivative neglecting mass terms) of displacement x_i with 
        respect to displacement x_j

    Notes
    -----
    The static correction is done purely on the arrays x_i and x_j, so there is 
    no mass normalization. This is a difference in contrast to the technique 
    used in the related function modal_derivative.

    See Also
    --------
    modal_derivative

    '''
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_dx_j = (K_func(x_j*h) - K)/h
    b = - dK_dx_j.dot(x_i) # rigth hand side of equation
    dx_i_dx_j = linalg.solve(K, b)
    if verbose:
        res = K.dot(dx_i_dx_j) + dK_dx_j.dot(x_i)
        print('\nComputation of static correction derivative. ')
        print('The condition number of the solution procedure is', np.linalg.cond(K))
        print('The residual is', linalg.norm(res),
              ', the relative residual is', linalg.norm(res)/linalg.norm(b))
    return dx_i_dx_j


def static_correction_theta(V, K_func, h=500*SQ_EPS, verbose=True):
    '''
    Computes the static correction derivatives of the basis V
    
    Parameters
    ----------
    V : ndarray
        array containing the linear basis
    K_func : function
        function returning the tangential stiffness matrix for a given 
        displacement. Has to work like K = K_func(u). 
    h : float, optional
        step width for finite difference scheme. Default value is 500 * machine 
        epsilon
    verbose : bool, optional
        flag for verbosity. Default value: True        
        
    Returns
    -------
    Theta : ndarray
        three dimensional array of static corrections derivatives. Theta[:,i,j] 
        contains the static derivative 1/2 * dx_i / dx_j. As the static derivatives 
        are symmetric, Theta[:,i,j] == Theta[:,j,i]. 
    '''
    no_of_dofs = V.shape[0]
    no_of_modes = V.shape[1]
    Theta = np.zeros((no_of_dofs, no_of_modes, no_of_modes))
    K = K_func(np.zeros(no_of_dofs))
    for i in range(no_of_modes):
        if verbose: print('Computing finite difference K-matrix')
        dK_dx_i = (K_func(h*V[:,i]) - K)/h
        b = - dK_dx_i @ V
        if verbose: print('Sovling linear system #', i)
        Theta[:,:,i] = sp.sparse.linalg.spsolve(K, b)
        if verbose: print('Done solving linear system #', i)
    if verbose:
        residual = np.sum(Theta - Theta.transpose(0,2,1))
        print('The residual, i.e. the unsymmetric values, are', residual)
    # make Theta symmetric
    Theta = 1/4*(Theta + Theta.transpose(0,2,1))
    return Theta

def principal_angles_and_vectors(V1, V2, cosine=True):
    '''
    Return the cosine of the principal angles of the two bases V1 and V2.

    Parameters
    ----------
    V1 : ndarray
        array denoting n-dimensional subspace spanned by V1 (Mxn)
    V2 : ndarray
        array denoting subspace 2. Dimension is (MxO)
    cosine : bool, optional
        flag stating, if the cosine of the angles is to be used

    Returns
    -------
    sigma : ndarray
        cosine of subspace angles
    F1 : ndarray
        array of principal vectors of subspace spanned by V1. The columns give
        the principal vectors, i.e. F1[:,0] is the first principal vector
        associated with theta[0] and so on.
    F2 : ndarray
        array of principal vectors of subspace spanned by V2.

    Note
    ----
    Both matrices V1 and V2 have live in the same vector space, i.e. they have
    to have the same number of rows

    Examples
    --------
    TODO

    See Also
    --------
    principal_angles

    References
    ----------
    G. H. Golub and C. F. Van Loan. Matrix computations, volume 3. JHU Press, 2012.

    '''
    Q1, R1 = linalg.qr(V1, mode='economic')
    Q2, R2 = linalg.qr(V2, mode='economic')
    U, sigma, V = linalg.svd(Q1.T.dot(Q2))
    F1 = Q1.dot(U)
    F2 = Q2.dot(V)
    if not cosine:
        sigma = np.arccos(sigma)
    return sigma, F1, F2


def principal_angles(V1, V2, cosine=True):
    '''
    Return the cosine of the principal angles of V1 and V2 in the vectornorm M.

    Parameters
    ----------
    V1 : ndarray
        array denoting n-dimensional subspace spanned by V1 (Mxn)
    V2 : ndarray
        array denoting subspace 2. Dimension is (MxO)
    cosine : bool, optional
        flag stating, if the cosine of the angles is to be used

    Returns
    -------
    sigma : ndarray
        cosine of subspace angles

    Examples
    --------
    TODO

    Note
    ----
    Both matrices V1 and V2 have live in the same vector space, i.e. they have
    to have the same number of rows

    See Also
    --------
    principal_angles_and_vectors

    References
    ----------
    G. H. Golub and C. F. Van Loan. Matrix computations, volume 3. JHU Press, 2012.

    '''
    Q1, R1 = linalg.qr(V1, mode='economic')
    Q2, R2 = linalg.qr(V2, mode='economic')
    sigma = linalg.svdvals(Q1.T.dot(Q2))
    if not cosine:
        sigma = np.arccos(sigma)

    return sigma


def krylov_subspace(M, K, b, omega=0, no_of_moments=3):
    '''
    Computes the Krylov Subspace associated with the input matrix b at the 
    frequency omega.

    Parameters
    ----------
    M : ndarray
        Mass matrix of the system.
    K : ndarray
        Stiffness matrix of the system.
    b : ndarray
        input vector of external forcing.
    omega : float, optional
        frequency for the frequency shift of the stiffness. Default value 0.
    no_of_moments : int, optional
        number of moments matched. Default value 3.

    Returns
    -------
    V : ndarray
        Krylov basis where vectors V[:,i] give the basis vectors.

    Examples
    --------
    TODO

    References
    ----------

    '''
    ndim = M.shape[0]
    no_of_inputs = b.size//ndim
    V = np.zeros((ndim, no_of_moments*no_of_inputs))
    lu = linalg.lu_factor(K - omega**2 * M)
    b_new = linalg.lu_solve(lu, b)
    b_new /= linalg.norm(b_new)
    V[:,0:no_of_inputs] = b_new.reshape((-1, no_of_inputs))
    for i in np.arange(1, no_of_moments):
        f = M.dot(b_new)
        b_new = linalg.lu_solve(lu, f)
        b_new /= linalg.norm(b_new)
        V[:,i*no_of_inputs:(i+1)*no_of_inputs] = b_new.reshape((-1, no_of_inputs))
        V[:,:(i+1)*no_of_inputs], R = linalg.qr(V[:,:(i+1)*no_of_inputs], mode='economic')
        b_new = V[:,i*no_of_inputs:(i+1)*no_of_inputs]
    sigmas = linalg.svdvals(V)
    print('Krylov Basis constructed. The singular values of the basis are', sigmas)
    return V


def craig_bampton(M, K, b, no_of_modes=5, one_basis=True):
    '''
    Computes the Craig-Bampton basis for the System M and K with the input Matrix b.

    Parameters
    ----------
    M : ndarray
        Mass matrix of the system.
    K : ndarray
        Stiffness matrix of the system.
    b : ndarray
        Input vector of the system
    no_of_modes : int, optional
        Number of internal vibration modes for the reduction of the system.
        Default is 5.
    one_basis : bool, optional
        Flag for setting, if one Craig-Bampton basis should be returned or if
        the static and the dynamic basis is chosen separately
        
    Returns
    -------
    V : array
        Basis constisting of static displacement modes and internal vibration modes

    if one_basis=True is chosen:

    V_static : ndarray
        Static displacement modes corresponding to the input vectors b with
        V_static[:,i] being the corresponding static displacement vector to b[:,i].
    V_dynamic : ndarray
        Internal vibration modes with the boundaries fixed.
    omega : ndarray
        eigenfrequencies of the internal vibration modes.

    Examples
    --------
    TODO

    Note
    ----
    There is a filter-out command to remove the interface eigenvalues of the system.

    References
    ----------
    TODO
    '''
    # boundaries
    ndof = M.shape[0]
    b_internal = b.reshape((ndof, -1))
    indices = sp.nonzero(b)
    boundary_indices = list(set(indices[0])) # indices
    no_of_inputs = b_internal.shape[-1]
    V_static_tmp = np.zeros((ndof, len(boundary_indices)))
    K_tmp = K.copy()
    K_tmp[:, boundary_indices] *= 0
    K_tmp[boundary_indices, :] *= 0
    K_tmp[boundary_indices, boundary_indices] = 1
    for i, index  in enumerate(boundary_indices):
        f = - K[:,index]
        f[boundary_indices] = 0
        f[index] = 1
        V_static_tmp[:,i] = linalg.solve(K_tmp, f)
    # Static Modes:
    V_static = np.zeros((ndof, no_of_inputs))
    for i in range(no_of_inputs):
        V_static[:,i] = V_static_tmp.dot(b_internal[boundary_indices, [i,]])

    # inner modes
    M_tmp = M.copy()
    # Attention: introducing eigenvalues of magnitude 1 into the system
    M_tmp[:, boundary_indices] *= 0
    M_tmp[boundary_indices, :] *= 0
    M_tmp[boundary_indices, boundary_indices] = 1E0
    K_tmp[boundary_indices, boundary_indices] = 1E0
    omega, V_dynamic = linalg.eigh(K_tmp, M_tmp)
    indexlist = np.nonzero(np.round(omega - 1, 3))[0]
    omega = np.sqrt(omega[indexlist])
    V_dynamic = V_dynamic[:, indexlist]
    if one_basis:
        return sp.hstack((V_static, V_dynamic[:, :no_of_modes]))
    else:
        return V_static, V_dynamic[:, :no_of_modes], omega[:no_of_modes]


def vibration_modes(mechanical_system, n=10, save=False):
    '''
    Compute the n first vibration modes of the given mechanical system using 
    a power iteration method. 
    
    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system to be analyzed.
    n : int
        number of modes to be computed.
    save : bool
        Flag for saving the modes in mechanical_system for ParaView export. 
        Default: True. 
    
    Returns
    -------
    omega : ndarray
        vector containing the eigenfrequencies of the mechanical system in 
        rad / s. 
    Phi : ndarray
        Array containing the vibration modes. Phi[:,0] is the first vibration 
        mode corresponding to eigenfrequency omega[0]

    Example
    -------
    
    Notes
    -----
    The core command using the ARPACK library is a little bit tricky. One has 
    to use the shift inverted mode for the solution of the mechanical 
    eigenvalue problem with the largest eigenvalues. Generally no convergence 
    is gained when the smallest eigenvalue is to be found. 
    '''
    K = mechanical_system.K()
    M = mechanical_system.M()
    
    lambda_, V = sp.sparse.linalg.eigsh(K, M=M, k=n, sigma=0, which='LM', 
                                        maxiter=100)
    omega = np.sqrt(lambda_)

    if save:
        for i, om in enumerate(omega):
            mechanical_system.write_timestep(om, V[:, i])
    
    return omega, V

def pod(mechanical_system, n):
    '''
    Compute the POD basis of a mechanical system. 
    
    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        MechanicalSystem which has run a time simulation and thus displacement 
        fields stored internally. 
    n : int
        Number of POD basis vectors which should be returned. 
        
    Returns
    -------
    sigma : ndarray
        Array of the singular values. 
    V : ndarray
        Array containing the POD vectors. V[:,0] contains the POD-vector 
        associated with sigma[0] etc. 
    
    Example
    -------
    '''
    # TODO: think about how to store the displacements and eventually the 
    # stresses internally. 
    pass
