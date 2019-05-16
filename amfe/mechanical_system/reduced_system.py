#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Reduced mechanical system.
"""

import numpy as np
from scipy.sparse import csc_matrix
from os import path
from h5py import File

from .mechanical_system import MechanicalSystem

__all__ = [
    'ReducedSystem'
]


class ReducedSystem(MechanicalSystem):
    '''
    Class for reduced systems. It is directly inherited from MechanicalSystem. Provides the interface for an
    integration scheme and so on where a basis vector is to be chosen...

    Notes
    -----
    The Basis V is a Matrix with x = V*q mapping the reduced set of coordinates q onto the physical coordinates x. The
    coordinates x are constrained, i.e. the x denotes the full system in the sense of the problem set and not of the
    pure finite element set.

    The system runs without providing a V_basis when constructing the method only for the unreduced routines.

    Attributes
    ----------
    V : ?
        Set of basis vectors the system has been reduced with u_constr = V*q.
    V_unconstr : ?
        Extended reduction basis that is extended by the displacement coordinates of the constrained degrees of freedom.
    u_red_output : ?
        Stores the timeseries of the generalized coordinates (similar to u_output).
    assembly_type : {'indirect', 'direct'}
        Stores the type of assembly method how the reduced system is computed.
    
    Examples
    --------
    my_system = amfe.MechanicalSystem()
    V = vibration_modes(my_system, n=20)
    my_reduced_system = amfe.reduce_mechanical_system(my_system, V)
    '''

    def __init__(self, V_basis=None, assembly='indirect', **kwargs):
        '''
        Parameters
        ----------
        V_basis : ndarray, optional
            Basis onto which the problem will be projected with an Galerkin-Projection.
        assembly : str {'direct', 'indirect'}
            flag setting, if direct or indirect assembly is done. For larger reduction bases, the indirect method is
            much faster.
        **kwargs : dict, optional
            Keyword arguments to be passed to the mother class MechanicalSystem.
        '''

        MechanicalSystem.__init__(self, **kwargs)
        self.V = V_basis
        self.u_red_output = []
        self.V_unconstr = self.dirichlet_class.unconstrain_vec(V_basis)
        self.assembly_type = assembly

    def K_and_f(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])
        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr, u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u, t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system is not valid.')
        return K, f_int

    def K(self, u=None, t=0):
        if u is None:
            u = np.zeros(self.V.shape[1])

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr, u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u, t)
            K = self.V_unconstr.T @ K_raw @ self.V_unconstr
        else:
            raise ValueError('The given assembly type for a reduced system is not valid.')
        return K

    def f_ext(self, u=None, du=None, t=0):
        return self.V.T @ MechanicalSystem.f_ext(self, self.V @ u, du, t)

    def f_int(self, u, t=0):

        if self.assembly_type == 'direct':
            # this is really slow! So this is why the assembly is done diretly
            K, f_int = self.assembly_class.assemble_k_and_f_red(self.V_unconstr, u, t)
        elif self.assembly_type == 'indirect':
            K_raw, f_raw = self.assembly_class.assemble_k_and_f(self.V_unconstr @ u, t)
            f_int = self.V_unconstr.T @ f_raw
        else:
            raise ValueError('The given assembly type for a reduced system is not valid.')
        return f_int

    # TODO: Remove workaround for update of damping matrix self.D_constr >>>
    def D(self, u=None, t=0, force_update=False):
        if self.D_constr is None or force_update:
            if self.rayleigh_damping:
                self.D_constr = self.rayleigh_damping_alpha * self.M() + self.rayleigh_damping_beta * self.K()
            else:
                self.D_constr = csc_matrix(self.M().shape)
        return self.D_constr

    # TODO: <<< Remove workaround for update of damping matrix self.D_constr

    def M(self, u=None, t=0, force_update=False):
        if self.M_constr is None or force_update:
            if u is None:
                u_full = None
            else:
                u_full = self.V @ u
            self.M_constr = self.V.T @ MechanicalSystem.M(self, u_full, t, force_update=True) @ self.V
        return self.M_constr

    def write_timestep(self, t, u):
        MechanicalSystem.write_timestep(self, t, self.V @ u)
        self.u_red_output.append(u.copy())

    def K_unreduced(self, u=None, t=0):
        '''
        Unreduced Stiffness Matrix.

        Parameters
        ----------
        u : ndarray, optional
            Displacement of constrained system. Default is zero vector.
        t : float, optionial
            Time. Default is 0.

        Returns
        -------
        K : sparse csr matrix
            Stiffness matrix.
        '''

        return MechanicalSystem.K(self, u, t)

    def f_int_unreduced(self, u, t=0):
        '''
        Internal nonlinear force of the unreduced system.

        Parameters
        ----------
        u : ndarray
            Displacement of unreduces system.
        t : float, optional
            Time, default value: 0.

        Returns
        -------
        f_nl : ndarray
            nonlinear force of unreduced system.
        '''

        return MechanicalSystem.f_int(self, u, t)

    def M_unreduced(self):
        '''
        Unreduced mass matrix.
        '''

        return MechanicalSystem.M(self)

    def export_paraview(self, filename, field_list=None):
        '''
        Export the produced results to ParaView via XDMF format.
        '''

        u_red_export = np.array(self.u_red_output).T
        u_red_dict = {'ParaView': 'False', 'Name': 'q_red'}

        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        new_field_list.append((u_red_export, u_red_dict))

        MechanicalSystem.export_paraview(self, filename, new_field_list)

        # add V and Theta to the hdf5 file
        filename_no_ext, _ = path.splitext(filename)
        with File(filename_no_ext + '.hdf5', 'r+') as f:
            f.create_dataset('reduction/V', data=self.V)

        return

    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.u_red_output = []
