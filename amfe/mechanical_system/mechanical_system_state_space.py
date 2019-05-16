#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Mechanical state-space system.
"""

import numpy as np
from scipy.sparse import bmat

from .mechanical_system import MechanicalSystem

__all__ = [
    'MechanicalSystemStateSpace'
]


class MechanicalSystemStateSpace(MechanicalSystem):
    def __init__(self, regular_matrix=None, **kwargs):
        MechanicalSystem.__init__(self, **kwargs)
        if regular_matrix is None:
            self.R_constr = self.K()
        else:
            self.R_constr = regular_matrix
        self.x_red_output = []
        self.E_constr = None

    def M(self, x=None, t=0, force_update=False):
        if self.M_constr is None or force_update:
            if x is not None:
                self.M_constr = MechanicalSystem.M(self, x[0:int(x.size / 2)], t, force_update)
            else:
                self.M_constr = MechanicalSystem.M(self, None, t, force_update)
        return self.M_constr

    def E(self, x=None, t=0, force_update=False):
        if self.E_constr is None or force_update:
            if self.M_constr is None or force_update:
                self.M(x, t, force_update)
            self.E_constr = bmat([[self.R_constr, None], [None, self.M_constr]])
        return self.E_constr

    def D(self, x=None, t=0, force_update=False):
        if self.D_constr is None or force_update:
            if x is not None:
                self.D_constr = MechanicalSystem.D(self, x[0:int(x.size / 2)], t, force_update)
            else:
                self.D_constr = MechanicalSystem.D(self, None, t, force_update)
        return self.D_constr

    def K(self, x=None, t=0):
        if x is not None:
            K = MechanicalSystem.K(self, x[0:int(x.size / 2)], t)
        else:
            K = MechanicalSystem.K(self, None, t)
        return K

    def A(self, x=None, t=0):
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-self.K(x, t), None]])
        else:
            A = bmat([[None, self.R_constr], [-self.K(x, t), -self.D_constr]])
        return A

    def f_int(self, x=None, t=0):
        if x is None:
            x = np.zeros(2 * self.dirichlet_class.no_of_constrained_dofs)
        f_int = MechanicalSystem.f_int(self, x[0:int(x.size / 2)], t)
        return f_int

    def F_int(self, x=None, t=0):
        if x is None:
            x = np.zeros(2 * self.dirichlet_class.no_of_constrained_dofs)
        if self.D_constr is None:
            F_int = np.concatenate((self.R_constr @ x[int(x.size / 2):], -self.f_int(x, t)), axis=0)
        else:
            F_int = np.concatenate((self.R_constr @ x[int(x.size / 2):], -self.D_constr @ x[int(x.size / 2):]
                                    - self.f_int(x, t)), axis=0)
        return F_int

    def f_ext(self, x, t):
        if x is None:
            f_ext = MechanicalSystem.f_ext(self, None, None, t)
        else:
            f_ext = MechanicalSystem.f_ext(self, x[0:int(x.size / 2)], x[int(x.size / 2):], t)
        return f_ext

    def F_ext(self, x, t):
        F_ext = np.concatenate((np.zeros(self.dirichlet_class.no_of_constrained_dofs), self.f_ext(x, t)), axis=0)
        return F_ext

    def K_and_f(self, x=None, t=0):
        if x is not None:
            K, f_int = MechanicalSystem.K_and_f(self, x[0:int(x.size / 2)], t)
        else:
            K, f_int = MechanicalSystem.K_and_f(self, None, t)
        return K, f_int

    def A_and_F(self, x=None, t=0):
        if x is None:
            x = np.zeros(2 * self.dirichlet_class.no_of_constrained_dofs)
        K, f_int = self.K_and_f(x, t)
        if self.D_constr is None:
            A = bmat([[None, self.R_constr], [-K, None]])
            F_int = np.concatenate((self.R_constr @ x[int(x.size / 2):], -f_int), axis=0)
        else:
            A = bmat([[None, self.R_constr], [-K, -self.D_constr]])
            F_int = np.concatenate((self.R_constr @ x[int(x.size / 2):], -self.D_constr @ x[int(x.size / 2):] - f_int),
                                   axis=0)
        return A, F_int

    def write_timestep(self, t, x):
        MechanicalSystem.write_timestep(self, t, x[0:int(x.size / 2)])
        self.x_output.append(x.copy())
        return

    def export_paraview(self, filename, field_list=None):
        x_export = np.array(self.x_output).T
        x_dict = {'ParaView': 'False', 'Name': 'x'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_export, x_dict))
        MechanicalSystem.export_paraview(self, filename, new_field_list)
        return

    def clear_timesteps(self):
        MechanicalSystem.clear_timesteps(self)
        self.x_output = []
        return
