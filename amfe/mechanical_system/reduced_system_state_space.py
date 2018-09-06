#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Reduced mechanical state space system.
"""

import numpy as np

from .mechanical_system_state_space import MechanicalSystemStateSpace

__all__ = [
    'ReducedSystemStateSpace'
]


class ReducedSystemStateSpace(MechanicalSystemStateSpace):
    def __init__(self, right_basis=None, left_basis=None, **kwargs):
        MechanicalSystemStateSpace.__init__(self, **kwargs)
        self.V = right_basis
        self.W = left_basis
        self.x_red_output = []

    def E(self, x=None, t=0, force_update=False):
        if self.E_constr is None or force_update:
            if x is not None:
                self.E_constr = self.W.T @ MechanicalSystemStateSpace.E(self, self.V @ x, t, force_update) @ self.V
            else:
                self.E_constr = self.W.T @ MechanicalSystemStateSpace.E(self, None, t, force_update) @ self.V
        return self.E_constr

    def E_unreduced(self, x_unreduced=None, t=0, force_update=False):
        return MechanicalSystemStateSpace.E(self, x_unreduced, t, force_update)

    def A(self, x=None, t=0):
        if x is not None:
            A = self.W.T @ MechanicalSystemStateSpace.A(self, self.V @ x, t) @ self.V
        else:
            A = self.W.T @ MechanicalSystemStateSpace.A(self, None, t) @ self.V
        return A

    def A_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.A(self, x_unreduced, t)

    def F_int(self, x=None, t=0):
        if x is not None:
            F_int = self.W.T @ MechanicalSystemStateSpace.F_int(self, self.V @ x, t)
        else:
            F_int = self.W.T @ MechanicalSystemStateSpace.F_int(self, None, t)
        return F_int

    def F_int_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.F_int(self, x_unreduced, t)

    def F_ext(self, x, t):
        if x is not None:
            F_ext = self.W.T @ MechanicalSystemStateSpace.F_ext(self, self.V @ x, t)
        else:
            F_ext = self.W.T @ MechanicalSystemStateSpace.F_ext(self, None, t)
        return F_ext

    def F_ext_unreduced(self, x_unreduced, t):
        return MechanicalSystemStateSpace.F_ext(self, x_unreduced, t)

    def A_and_F(self, x=None, t=0):
        if x is not None:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, self.V @ x, t)
        else:
            A_, F_int_ = MechanicalSystemStateSpace.A_and_F(self, None, t)
        A = self.W.T @ A_ @ self.V
        F_int = self.W.T @ F_int_
        return A, F_int

    def A_and_F_unreduced(self, x_unreduced=None, t=0):
        return MechanicalSystemStateSpace.A_and_F(self, x_unreduced, t)

    def write_timestep(self, t, x):
        MechanicalSystemStateSpace.write_timestep(self, t, self.V @ x)
        self.x_red_output.append(x.copy())
        return

    def export_paraview(self, filename, field_list=None):
        x_red_export = np.array(self.x_red_output).T
        x_red_dict = {'ParaView': 'False', 'Name': 'x_red'}
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()
        new_field_list.append((x_red_export, x_red_dict))
        MechanicalSystemStateSpace.export_paraview(self, filename, new_field_list)
        return

    def clear_timesteps(self):
        MechanicalSystemStateSpace.clear_timesteps(self)
        self.x_red_output = []
        return
