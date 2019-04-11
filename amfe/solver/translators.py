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

import numpy as np

from amfe.solver.tools import MemoizeStiffness, MemoizeJac
from amfe.constraint.constraint_formulation_boolean_elimination import BooleanEliminationConstraintFormulation
from amfe.constraint.constraint_formulation_lagrange_multiplier import SparseLagrangeMultiplierConstraintFormulation

__all__ = [
    'MechanicalSystemBase',
    'MechanicalSystem',
    'ConstrainedMechanicalSystem'
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

    def D(self, q, dq, t):
        return self.structural_component.D(q, dq, t)

    def F(self, q, dq, t):
        return self.structural_component.f_ext(q, dq, t) - self.structural_component.f_int(q, dq, t)

    def K(self, q, dq, t):
        return self.structural_component.K(q, dq, t)

    def unconstrain(self, x, dx, ddx, t):
        return x, dx, ddx

    @property
    def dimension(self):
        return self.structural_component._constraints.no_of_dofs_unconstrained


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
    
    def F(self, x, dx, t):
        return self.structural_component.f_ext(x, dx, t) - self._f_int(x, dx, t)


class ConstrainedMechanicalSystem(MechanicalSystemBase):

    def __init__(self, structural_component, formulation='boolean', constant_mass=True, **formulation_options):
        super().__init__(structural_component)
        self._constraint_formulation = None
        self._memoize_m = constant_mass
        self._M = None
        self._h_func = MemoizeJac(self.structural_component.h_and_dh_dq)
        self._dh_dq_func = self._h_func.derivative
        self._create_constraint_formulation(formulation, formulation_options)
        self._f_ext = None

    def M(self, x, dx, t):
        if self._memoize_m:
            if self._M is None:
                self._M = self._constraint_formulation.M(x, dx, t)
            return self._M
        else:
            return self.structural_component.M(x, dx, t)

    def F(self, x, dx, t):
        return self._constraint_formulation.F(x, dx, t)

    def f_int(self, x, dx, t):
        return -self._constraint_formulation.F(x, dx, t)

    def f_ext(self, x, dx, t):
        if self._f_ext is None:
            self._f_ext = np.zeros_like(x)
        return self._f_ext

    def K(self, x, dx, t):
        return self._constraint_formulation.K(x, dx, t)

    def D(self, x, dx, t):
        return self._constraint_formulation.D(x, dx, t)

    def _create_constraint_formulation(self, formulation, formulation_options):
        if formulation == 'boolean':
            no_of_dofs_unconstrained = self.structural_component._constraints.no_of_dofs_unconstrained
            self._constraint_formulation = BooleanEliminationConstraintFormulation(no_of_dofs_unconstrained,
                                                                                   self.structural_component.M,
                                                                                   self._h_func,
                                                                                   self.structural_component.B_holo,
                                                                                   self._dh_dq_func,
                                                                                   self.structural_component.dh_ddq,
                                                                                   self.structural_component.g_holo)

        elif formulation == 'lagrange':
            no_of_dofs_unconstrained = self.structural_component._constraints.no_of_dofs_unconstrained
            self._constraint_formulation = SparseLagrangeMultiplierConstraintFormulation(no_of_dofs_unconstrained,
                                                                                         self.structural_component.M,
                                                                                         self._h_func,
                                                                                         self.structural_component.B_holo,
                                                                                         self._dh_dq_func,
                                                                                         self.structural_component.dh_ddq,
                                                                                         self.structural_component.g_holo)
        else:
            raise ValueError('formulation not valid')

        if formulation_options is not None:
            self._constraint_formulation.set_options(**formulation_options)


    @property
    def dimension(self):
        return self._constraint_formulation.dimension

    def unconstrain(self, x, dx, ddx, t):
        return self._constraint_formulation.recover(x, dx, ddx, t)
