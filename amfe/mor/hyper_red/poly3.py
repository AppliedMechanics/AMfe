#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np

from ..reduction_basis import jacobian_finite_difference


__all__ = ['compute_quadratic_force_tensor',
           'compute_cubic_force_tensor',
           'Poly3']


def compute_quadratic_force_tensor(K_func, V, h=1.0, method='central'):
    """
    Compute the quadratic tensor H of the nonlinear force via finite difference
    scheme.

    Parameters
    ----------
    V : ndarray, shape (ndim_full, ndim_red)
        Basis projection array.
    mechanical_system : instance of amfe.MechanicalSystem
        Instance of the unreduced MechanicalSystem.
    h : float, optional
        finite difference step width. Default value: 1.

    Returns
    -------
    H_red : ndarray
        Reduced quadratic tensor H of the nonlinear force. Has dimension
        (ndim_red, ndim_red, ndim_red).

    """

    n_full, n_red = V.shape
    H_red = np.zeros((n_red, n_red, n_red))

    print('>> Computing unique K(2) entries:')

    x0 = np.zeros(n_full)
    dx0 = x0.copy()

    def K_deriv(x):
        return K_func(x, dx0, 0.0)

    for i, vec in enumerate(V.T):
        print('\tentry = {:3d}/{:3d}'.format(i+1, V.shape[1]))
        dK_dvec = jacobian_finite_difference(K_deriv, vec, x0, h, method)
        H_red[:, :, i] = V.T @ dK_dvec @ V

    return H_red


def compute_cubic_force_tensor(K_func, V, h=1., method='central'):
    r"""
    Compute the cubic part of the nonlinear internal force for a given
    reduction basis V:

    .. math::
        \mathbf{f}_{\text{int, reduced}}(\mathbf{q}) = \mathbf{V}^T
        \mathbf{f}_{int}(\mathbf{V}\mathbf{q}) \in \mathbb{R}^{n}\\
        \frac{\partial^3 \mathbf{f}_{\text{int, reduced}}(\mathbf{q})}
             {\partial \mathbf{q}^3}
        = \mathbf{K}_3 \in \mathbb{R}^{n \times n \times n \times n}

    Parameters
    ----------
    V : ndarray, shape (ndim_full, ndim_red)
        Reduction basis.
    mechanical_system : instance of amfe.MechanicalSystem
        Instance of the mechanical system class. The function used here is the
        tangential stiffness matrix K and not the nonlinear force f_int for
        efficiency and accuracy reasons.
    h : float, optional
        Step width for the finite difference scheme. Note, that the general
        rule of numerical mathematics to use :math:`h = \sqrt{\epsilon}` is
        way too small. As mentioned below in the Notes, the symmetry of the
        resultin tensor can be used to find a problem-specific h.
        Default value: 1.

    Returns
    -------
    K_3_red : ndarray, shape (ndim_red, ndim_red, ndim_red, ndim_red)
        Fourth order tensor describing the negative fourth derivative of the
        internal elastic potential, or, the coefficients of the cubic parts
        of the nonlinear force.

    Note
    ----
    * This is directly the cubic part from the third derivative of the force
      computed via finite differences.
    * In order to receive the nonlinear force, the tensor has to be multiplied
      with a factor of 1/6, as this tensor is the third derivative of the
      nonlinear internal force.
    * This function is extremely sensitive to the stepwidth h. A good indicator
      if the stepwidth h is right, is the symmetry of the tensor forming the
      fourth derivative of the internal elastic potential, which should be
      perfectly symmetric with respect to *all* axes.

    """
    print('>> Computing unique K(3) entries:')

    n_full, n_red = V.shape
    K_3_red = np.zeros((n_red, n_red, n_red, n_red))

    no_computed_entries = np.sum(np.arange(0, n_red)+1)
    count=0

    dx0 = np.zeros(n_full)

    def K_deriv(x):
        return K_func(x, dx0, 0.0)

    for i in range(n_red):
        for j in range(i+1):
            count+=1
            print('\tentry = {:3d}/{:3d}\t(i={:2d}/{:2d}\tj={:2d}/{:2d})'.format(count,no_computed_entries,i+1, V.shape[1], j+1, i+1))
            V_i = V[:,i]
            V_j = V[:,j]
            if method == 'central':
                dK_di_dj = ( K_deriv(+h*V_i + h*V_j) - K_deriv(-h*V_i + h*V_j) -K_deriv(+h*V_i - h*V_j) + K_deriv(-h*V_i - h*V_j) )/(4*(h**2)) #4x auswertung von K in einer iteration
            else:
                raise NotImplementedError('Other methods as central are not implemented to compute the hessian')
            K_3_red[:, :, i, j] = K_3_red[:, :, j, i] = V.T @ dK_di_dj @ V

    return K_3_red


class Poly3:
    def __init__(self, K1, K2, K3):
        self.K1 = K1
        self.K2_full = K2
        self.K3_full = K3

    def K_and_f_int(self, x, dx, t):
        K2_temp = self.K2_full @ x
        K3_temp = self.K3_full @ x @ x
        K = self.K1 + K2_temp + 0.5 * K3_temp
        f_int = (self.K1 + 0.5 * K2_temp + (1. / 6) * K3_temp) @ x
        return K, f_int
