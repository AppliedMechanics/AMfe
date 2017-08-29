# -*- coding: utf-8 -*-

'''
Quadratic Manifold system...
'''

import os
import h5py
import copy
import numpy as np

from ..mechanical_system import MechanicalSystem, ReducedSystem

__all__ = ['QMSystem',
           'reduce_mechanical_system_qm',
           ]


class QMSystem(MechanicalSystem):
    '''
    Quadratic Manifold Finite Element system.

    '''

    def __init__(self, **kwargs):
        MechanicalSystem.__init__(self, **kwargs)
        self.V = None
        self.Theta = None
        self.no_of_red_dofs = None
        self.u_red_output = []

    def M(self, u=None, t=0):
        # checks, if u is there and M is already computed
        if u is None:
            u = np.zeros(self.no_of_red_dofs)
        if self.M_constr is None:
            MechanicalSystem.M(self)

        P = self.V + self.Theta @ u
        M_red = P.T @ self.M_constr @ P
        return M_red

    def K_and_f(self, u=None, t=0):
        '''
        Take care here! It is not clear yet how to compute the tangential
        stiffness matrix!

        It seems to be like the contribution of geometric and material
        stiffness.
        '''
        if u is None:
            u = np.zeros(self.no_of_red_dofs)
        theta_u = self.Theta @ u
        u_full = (self.V + 1/2*theta_u) @ u
        P = self.V + theta_u
        K_unreduced, f_unreduced = MechanicalSystem.K_and_f(self, u_full, t)
        K1 = P.T @ K_unreduced @ P
        K2 = self.Theta.T @ f_unreduced
        K = K1 + K2
        f = P.T @ f_unreduced
        return K, f

    def S_and_res(self, u, du, ddu, dt, t, beta, gamma):
        '''
        TODO: checking the contributions of the different parts of the
        iteration matrix etc.

        '''
        # checking out that constant unreduced M is built
        if self.M_constr is None:
            MechanicalSystem.M(self)
        M_unreduced = self.M_constr

        theta = self.Theta
        theta_u = theta @ u
        u_full = (self.V + 1/2*theta_u) @ u

        K_unreduced, f_unreduced = MechanicalSystem.K_and_f(self, u_full, t)
        f_ext_unred = MechanicalSystem.f_ext(self, u_full, None, t)
        # nonlinear projector P
        P = self.V + theta_u

        # computing the residual
        res_accel = M_unreduced @ (P @ ddu)
        res_gyro = M_unreduced @ (theta @ du) @ du
        res_full = res_accel + res_gyro + f_unreduced - f_ext_unred
        # the different contributions to stiffness
        K1 = theta.T @ res_full
        K2 = P.T @ M_unreduced @ (theta @ ddu)
        K3 = P.T @ K_unreduced @ P
        K = K1 + K2 + K3
        # gyroscopic matrix and reduced mass matrix
        G = P.T @ M_unreduced @ (2*theta @ du)
        M = P.T @ M_unreduced @ P

        res = P.T @ res_full
        f_ext = P.T @ f_ext_unred
        S = 1/(dt**2 * beta) * M + gamma/(dt*beta) * G + K
        return S, res, f_ext

    def f_ext(self, u, du, t):
        '''
        Return the reduced external force. The velocity du is by now ignored.
        '''
        theta_u = self.Theta @ u
        u_full = (self.V + 1/2*theta_u) @ u
        P = self.V + theta_u
        f_ext_unred = MechanicalSystem.f_ext(self, u_full, None, t)
        f_ext = P.T @ f_ext_unred
        return f_ext

    def write_timestep(self, t, u):
        u_full = self.V @ u + (self.Theta @ u) @ u * 1/2
        MechanicalSystem.write_timestep(self, t, u_full)
        # own reduced output
        self.u_red_output.append(u.copy())
        return

    def export_paraview(self, filename, field_list=None):
        '''
        Export the produced results to ParaView via XDMF format.
        '''
        ReducedSystem.export_paraview(self, filename, field_list)
        filename_no_ext, _ = os.path.splitext(filename)

        # add Theta to the hdf5 file
        with h5py.File(filename_no_ext + '.hdf5', 'r+') as f:
            f.create_dataset('reduction/Theta', data=self.Theta)

        return


def reduce_mechanical_system_qm(mechanical_system, V, Theta, overwrite=False):
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
    assert V.shape[-1] == Theta.shape[-1]
    assert Theta.shape[1] == Theta.shape[2]
    assert Theta.shape[0] == V.shape[0]

    no_of_red_dofs = V.shape[-1]
    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)

    reduced_sys.__class__ = QMSystem
    reduced_sys.V = V.copy()
    reduced_sys.Theta = Theta.copy()

    # reduce Rayleigh damping matrix
    if reduced_sys.D_constr is not None:
        reduced_sys.D_constr = V.T @ reduced_sys.D_constr @ V

    # define internal variables
    reduced_sys.u_red_output = []
    reduced_sys.no_of_red_dofs = no_of_red_dofs
    return reduced_sys
