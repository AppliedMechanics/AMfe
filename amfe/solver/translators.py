#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Translation-module enabling the user to translate different types of models and components to the standard solver-API
or provide only certain parts of the model to the solver.

The translator has to provide the following methods:
M()
D()
f_int()
K()
f_ext()
"""

from amfe.solver.tools import MemoizeStiffness

__all__ = [
    'MechanicalSystemBase',
    'MechanicalSystem',
]


class MechanicalSystemBase:
    """
    Most basic translator, that just hands over the structural-component methods to the solver.
    The default case is a mechanical system here due to tradition, but might be something else as long as API and implementation are similar.
    """
    def __init__(self, structural_component):
        self.structural_component = structural_component

    def M(self, q, dq, t):
        return self.structural_component.M(q, dq, t)

    def D(self, q, dq, ddq, t):
        return self.structural_component.D(q, dq, ddq, t)

    def f_int(self, q, dq, ddq, t):
        return self.structural_component.f_int(q, dq, ddq, t)

    def K(self, q, dq, ddq, t):
        return self.structural_component.K(q, dq, ddq, t)

    def f_ext(self, q, dq, ddq, t):
        return self.structural_component.f_ext(q, dq, ddq, t)

    def unconstrain_vector(self, vector):
        return self.structural_component.unconstrain_vector(vector)

    def constrain_vector(self, vector):
        return self.structural_component.constrain_vector(vector)


class MechanicalSystem(MechanicalSystemBase):
    # constant m
    def __init__(self, structural_component, constant_mass=True):
        super().__init__(structural_component)
        self._memoize_m = constant_mass
        self._M = None
        self._f_int = MemoizeStiffness(self.structural_component.K_and_f_int)
        self._K = self._f_int.derivative
        
    def M(self, q, dq, t):
        if self._memoize_m:
            if self._M is None:
                self._M = self.structural_component.M(q, dq, t)
            return self._M
        else:
            return self.structural_component.M(q, dq, t)
    
    def f_int(self, q, dq, ddq, t):
        return self._f_int(q, dq, ddq, t)
    
    def K(self, q, dq, ddq, t):
        return self._K(q, dq, ddq, t)
