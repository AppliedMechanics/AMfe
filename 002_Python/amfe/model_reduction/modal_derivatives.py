# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:06:59 2015

@author: johannesr
"""


def modal_derivative(x_i, x_j, K_func, M, omega_i):
    '''
    Compute the real modal derivative of the given system using Nelson's formulation.

    The modal derivative computed is dx_i / dx_j, i.e. the change of the
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
    x_i /= np.sqrt(x_i.dot(M.dot(x_i)))
    x_j /= np.sqrt(x_j.dot(M.dot(x_j)))
    h = np.sqrt(np.finfo(float).eps)*100 # step size length
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_x_j = (K_func(x_j*h) - K)/h
    d_omega_2_d_x_i = x_i.dot(dK_x_j.dot(x_i))
    print('Influence of the change of the eigenfrequency:', d_omega_2_d_x_i)
    F_i = (d_omega_2_d_x_i*M - dK_x_j).dot(x_i)
    K_dyn_i = K - omega_i**2 * M
    row_index = np.argmax(abs(x_i))
    K_dyn_i[:,row_index], K_dyn_i[row_index,:], K_dyn_i[row_index,row_index] = 0, 0, 1
    F_i[row_index] = 0
    v_i = sp.linalg.solve(K_dyn_i, F_i)
    c_i = -v_i.dot(M.dot(x_i))
    dx_i_dx_j = v_i + c_i*x_i
    return dx_i_dx_j


    # Simplified mds:
    simplified_md = sp.linalg.solve(K, - dK_x_j.dot(x_i))
    return dphi_i_deta_j, simplified_md



def static_correction_derivative(x_i, x_j, K_func):
    '''
    Computes the static correction vectors.

    Parameters
    ----------
    x_i : ndarray
        displacement vector i
    x_j : ndarray
        displacement vector j
    K_func : function
        function for the tangential stiffness matrix to be called in the form K_tangential = K_func(x_i)

    Returns
    -------
    dx_i_dx_j : ndarray
        static correction derivative (if x_i and x_j is a modal vector it's the modal derivative neglecting mass terms) of displacement x_i with respect to displacement x_j

    Notes
    -----
    The static correction is done purely on the arrays x_i and x_j, so there is no mass normalization. This is a difference in contrast to the technique used in the related function modal_derivative.

    See Also
    --------
    modal_derivative

    '''
    h = np.sqrt(np.finfo(float).eps)*100 # step size length
    ndof = x_i.shape[0]
    K = K_func(np.zeros(ndof))
    dK_x_j = (K_func(x_j*h) - K)/h
    dx_i_dx_j = sp.linalg.solve(K, - dK_x_j.dot(x_i))
    return dx_i_dx_j












