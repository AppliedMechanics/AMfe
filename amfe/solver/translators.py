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

import  numpy as np
from scipy.sparse import hstack as sphstack
from scipy.sparse import vstack as spvstack
from scipy.sparse import csr_matrix
from amfe.solver.tools import MemoizeStiffness

__all__ = [
    'MechanicalSystemBase',
    'MechanicalSystem',
    'ReducedMechanicalSystem',
    'StateSpaceSystem',
    'StateSpaceSystemT',
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
    def __init__(self, structural_component, constant_mass=True, constant_damping=False):
        super().__init__(structural_component)
        self._memoize_m = constant_mass
        self._memoize_d = constant_damping
        self._M = None
        self._D = None
        self._f_int = MemoizeStiffness(self.structural_component.K_and_f_int)
        self._K = self._f_int.derivative
        
    def M(self, q, dq, t):
        if self._memoize_m:
            if self._M is None:
                self._M = self.structural_component.M(q, dq, t)
            return self._M
        else:
            return self.structural_component.M(q, dq, t)

    def D(self, q, dq, ddq, t):
        if self._memoize_d:
            if self._D is None:
                self._D = self.structural_component.D(q, dq, ddq, t)
            return self._D
        else:
            return self.structural_component.D(q, dq, ddq, t)

    def f_int(self, q, dq, ddq, t):
        return self._f_int(q, dq, ddq, t)
    
    def K(self, q, dq, ddq, t):
        return self._K(q, dq, ddq, t)


class StateSpaceSystemT:
    def __init__(self, mechanical_system):
        self._system = mechanical_system

    def A(self, q, dq, ddq, t):
        K = self._system.K(q, dq, ddq, t)
        M = self._system.M(q, dq, t)
        empty = csr_matrix(M.shape)
        return spvstack((sphstack((-K, empty)), sphstack((empty, M))))

    def E(self, q, dq, ddq, t):
        M = self._system.M(q, dq, t)
        D = self._system.D(q, dq, ddq, t)
        empty = csr_matrix(M.shape)
        return spvstack((sphstack((D, M)), sphstack((M, empty))))

    def B_f_ext(self, q, dq, ddq, t):
        f_ext = self._system.f_ext(q, dq, ddq, t)
        B_f_ext = np.zeros(2*len(f_ext))
        B_f_ext[:len(f_ext)] = f_ext
        return B_f_ext


class StateSpaceSystem(MechanicalSystem):
    def __init__(self, structural_component, constant_mass=True, constant_damping=False):
        super().__init__(structural_component, constant_mass, constant_damping)

    def A(self, q, dq, ddq, t):
        K = super().K(q, dq, ddq, t)
        M = super().M(q, dq, t)
        empty = csr_matrix(M.shape)
        return spvstack((sphstack((-K, empty)), sphstack((empty, M))))

    def E(self, q, dq, ddq, t):
        M = super().M(q, dq, t)
        D = super().D(q, dq, ddq, t)
        empty = csr_matrix(M.shape)
        return spvstack((sphstack((D, M)), sphstack((M, empty))))

    def B_f_ext(self, q, dq, ddq, t):
        f_ext = super().f_ext(q, dq, ddq, t)
        B_f_ext = np.zeros(2*len(f_ext))
        B_f_ext[:len(f_ext)] = f_ext
        return B_f_ext


class ReducedMechanicalSystem(MechanicalSystemBase):
    def __init__(self, structural_component, V, constant_mass=True, constant_damping=False):
        super().__init__(structural_component)
        self._memoize_m = constant_mass
        self._memoize_d = constant_damping
        self._M = None
        self._D = None
        self._f_int = MemoizeStiffness(self._K_and_f_int_red)
        self._K = self._f_int.derivative
        self._V = V

    def _K_and_f_int_red(self, q, dq, ddq, t):
        u = self.unconstrain_vector(q)
        du = self.unconstrain_vector(dq)
        ddu = self.unconstrain_vector(ddq)
        K, f_int = self.structural_component.K_and_f_int(u, du, ddu, t)
        return self._V.T @ K @ self._V, self._V.T @ f_int

    def M(self, q, dq, t):
        if self._memoize_m:
            if self._M is None:
                self._M = self._V.T @ self.structural_component.M(self.unconstrain_vector(q),
                                                                  self.unconstrain_vector(dq),
                                                                  t) @ self._V
            return self._M
        else:
            return self._V.T @ self.structural_component.M(self.unconstrain_vector(q),
                                                           self.unconstrain_vector(dq),
                                                           t) @ self._V

    def f_int(self, q, dq, ddq, t):
        return self._f_int(q, dq, ddq, t)

    def K(self, q, dq, ddq, t):
        return self._K(q, dq, ddq, t)

    def D(self, q, dq, ddq, t):
        if self._memoize_d:
            if self._D is None:
                self._D = self._V.T @ self.structural_component.D(self.unconstrain_vector(q), self.unconstrain_vector(dq), self.unconstrain_vector(ddq), t) @ self._V
            return self._D
        else:
            return self._V.T @ self.structural_component.D(self.unconstrain_vector(q), self.unconstrain_vector(dq), self.unconstrain_vector(ddq), t) @ self._V

    def f_ext(self, q, dq, ddq, t):
        return self._V.T @ self.structural_component.f_ext(self.unconstrain_vector(q), self.unconstrain_vector(dq), self.unconstrain_vector(ddq), t)

    def unconstrain_vector(self, vector):
        return self.structural_component.unconstrain_vector(self._V @ vector)
