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
K()
f_ext()
f_int()
dimension
"""
import numpy as np

from amfe.solver.tools import MemoizeStiffness, MemoizeConstant
from amfe.constraint.constraint_formulation_boolean_elimination import BooleanEliminationConstraintFormulation
from amfe.constraint.constraint_formulation_lagrange_multiplier import SparseLagrangeMultiplierConstraintFormulation
from amfe.constraint.constraint_formulation_nullspace_elimination import NullspaceConstraintFormulation

__all__ = [
    'MechanicalSystem',
    'MulticomponentMechanicalSystem',
    'create_constrained_mechanical_system_from_component',
    '_create_constraint_formulation',
    'create_mechanical_system_from_structural_component',
]


class MechanicalSystem:
    """
    Describes a second order system of this type:

    .. math::
        M \ddot{x} + f_{int}(x, \dot x, t) = f_{ext}

    Linearized entities:

    .. math::
        K = \frac{\partial (f_{int} - f_{ext}}{\partial x}
        D = \frac{\partial (f_{int} - f_{ext}}{\partial \dot x}
    """
    def __init__(self, dimension, M_func, D_func, K_func, f_ext_func, f_int_func):

        self._dimension = dimension
        self._M_func = M_func
        self._K_func = K_func
        self._f_ext_func = f_ext_func
        self._f_int_func = f_int_func
        self._D_func = D_func

    def M(self, q, dq, t):
        return self._M_func(q, dq, t)

    def D(self, q, dq, t):
        return self._D_func(q, dq, t)

    def f_ext(self, q, dq, t):
        return self._f_ext_func(q, dq, t)

    def f_int(self, q, dq, t):
        return self._f_int_func(q, dq, t)

    def K(self, q, dq, t):
        return self._K_func(q, dq, t)

    @property
    def dimension(self):
        return self._dimension


class MulticomponentMechanicalSystem:
    """
    Translator for a system, that consists of several components.
    """
    def __init__(self, composite, constant_mass=False, constant_damping=False, constraint_formulation='boolean',
                                                                                            ** formulation_options):
        self.mechanical_systems = dict()
        self.constraint_formulations = dict()
        for comp_id, component in composite.components.items():
            if component._constraints.no_of_constraints == 0:
                new_mechanical_system = create_mechanical_system_from_structural_component(component, constant_mass,
                                                                                           constant_damping)
            else:
                new_mechanical_system, formulation = create_constrained_mechanical_system_from_component(component,
                                                                constant_mass, constant_damping, constraint_formulation,
                                                                                                **formulation_options)
                self.constraint_formulations[comp_id] = formulation

            self.mechanical_systems[comp_id] = new_mechanical_system

        self.connector = composite.connector

        self.connections = dict()
        for key in self.connector.constraints:
            self.connections[key] = self.B(key)

    def B(self, key):
        """
        Returns the constraint-matrix connecting two components at an interface. The key describes, which components are
        connected.

        Parameters
        ----------
        key : tuple
            interface-key, that consists of both connected components

        Returns
        -------
        B : ndarray
            connection-matrix with dof-dimension of the constrained system
        """
        if key[-1] in self.constraint_formulations:
            B_constrained = self.constraint_formulations[key[-1]].L.T @ self.connector.constraints[key].T
        else:
            B_constrained = self.connector.constraints[key].T
        return B_constrained.T

    def recover(self, x_dict, dx_dict=None, ddx_dict=None, t=0):
        """
        Method to recover the full solutions of the components

        Parameters
        ----------
        x_dict : dict
            dictionary of the components' final ode variables in the constrained space
        dx_dict : dict
            dictionary of the components' 1st time derivative of the final ode variables in the constrained space
        ddx_dict : dict
            dictionary of the components' 2nd time derivative of the final ode variables in the constrained space
        t : float
            time

        Returns
        -------
        u : dict
            dictionary of the components' full displacements
        du : dict
            dictionary of the components' full velocities
        ddu : dict
            dictionary of the components' full accelerations
        """
        u = dict()
        du = dict()
        ddu = dict()
        for comp_id in self.mechanical_systems.keys():

            x = x_dict[comp_id]

            if dx_dict is None and ddx_dict is None:
                dx = np.zeros_like(x)
                ddx = np.zeros_like(x)
            else:
                dx = dx_dict[comp_id]
                ddx = ddx_dict[comp_id]

            if comp_id in self.constraint_formulations:
                u[comp_id], du[comp_id], ddu[comp_id] = self.constraint_formulations[comp_id].recover(x, dx, ddx, t)
            else:
                u[comp_id], du[comp_id], ddu[comp_id] = x, dx, ddx

        return u, du, ddu


def create_mechanical_system_from_structural_component(structural_component, constant_mass=False,
                                                       constant_damping=False):
    """
    Create a MechanicalSystem Object from a structural component

    Parameters
    ----------
    structural_component : amfe.component.StructuralComponent
        Structural component that will describe the mechanical system
    constant_mass : bool
        flag if the mass is constant
    constant_damping : bool
        flag indicating if damping matrix is constant

    Returns
    -------
    system : MechanicalSystem
        Created MechanicalSystem object describing the Structural Component
    """
    if constant_mass:
        M = MemoizeConstant(structural_component.M)
    else:
        M = structural_component.M

    if constant_damping:
        D = MemoizeConstant(structural_component.D)
    else:
        D = structural_component.D

    f_int = MemoizeStiffness(structural_component.K_and_f_int)
    K = f_int.derivative
    f_ext = structural_component.f_ext

    dimension = structural_component.mapping.no_of_dofs

    system = MechanicalSystem(dimension, M, D, K, f_ext, f_int)
    return system


def create_constrained_mechanical_system_from_component(structural_component, constant_mass=False,
                                                        constant_damping=False, constraint_formulation='boolean',
                                                        **formulation_options):
    """
    Create a mechanical system from a component where the constraints are applied by a constraint formulation

    Parameters
    ----------
    structural_component : amfe.component.StructuralComponent
        Structural component describing the mechanical system
    constant_mass : bool
        Flag indicating if mass matrix is constant
    constant_damping : bool
        Flag indicating if damping matrix is constant
    constraint_formulation : str {'boolean', 'lagrange', 'nullspace_elimination'}
        String describing the constraint formulation that shall be used
    formulation_options : dict
        options passed to the set_options method of the constraint formulation

    Returns
    -------
    system : amfe.solver.translators.MechanicalSystem
    formulation : amfe.constraint.ConstraintFormulation
    """
    system_unconstrained = create_mechanical_system_from_structural_component(structural_component)
    constraint_formulation = _create_constraint_formulation(system_unconstrained, structural_component,
                                                            constraint_formulation, **formulation_options)

    if constant_mass:
        M = MemoizeConstant(constraint_formulation.M)
    else:
        M = constraint_formulation.M

    if constant_damping:
        D = MemoizeConstant(constraint_formulation.D)
    else:
        D = constraint_formulation.D

    f_int = constraint_formulation.f_int
    K = constraint_formulation.K
    f_ext = constraint_formulation.f_ext

    dimension = constraint_formulation.dimension

    system = MechanicalSystem(dimension, M, D, K, f_ext, f_int)

    return system, constraint_formulation


def _create_constraint_formulation(mechanical_system, component, formulation, **formulation_options):
    """
    Internal method that creates a constraint formulation for a mechanical system combined with the constraints
    from a component

    Parameters
    ----------
    mechanical_system : MechanicalSystem
        Mechanical system whose matrices M, K, D, f_int, f_ext will be used in the constraint formulation
    component : amfe.component.StructuralComponent
        Structural Component having constraint functions, such as g_holo, B, b, a.
    formulation : str {'boolean', 'lagrange', 'nullspace_elimination'}
        String describing the constraint formulation that shall be used
    formulation_options : dict
        options passed to the set_options method of the constraint formulation

    Returns
    -------
    constraint_formulation : amfe.constraint.ConstraintFormulation
        A ConstraintFormulation object that applies the constraints on the mechanical system.
    """
    no_of_dofs_unconstrained = mechanical_system.dimension
    if formulation == 'boolean':

        constraint_formulation = BooleanEliminationConstraintFormulation(no_of_dofs_unconstrained,
                                                                         mechanical_system.M,
                                                                         mechanical_system.f_int,
                                                                         component.B,
                                                                         mechanical_system.f_ext,
                                                                         mechanical_system.K,
                                                                         mechanical_system.D,
                                                                         g_func=
                                                                         component.g_holo)
    elif formulation == 'lagrange':
        constraint_formulation = SparseLagrangeMultiplierConstraintFormulation(no_of_dofs_unconstrained,
                                                                               mechanical_system.M,
                                                                               mechanical_system.f_int,
                                                                               component.B,
                                                                               mechanical_system.f_ext,
                                                                               mechanical_system.K,
                                                                               mechanical_system.D,
                                                                               g_func=
                                                                               component.g_holo)
    elif formulation == 'nullspace_elimination':
        constraint_formulation = NullspaceConstraintFormulation(no_of_dofs_unconstrained,
                                                                mechanical_system.M,
                                                                mechanical_system.f_int,
                                                                component.B,
                                                                mechanical_system.f_ext,
                                                                mechanical_system.K,
                                                                mechanical_system.D,
                                                                g_func=component.g_holo,
                                                                b_func=component.b,
                                                                a_func=component.a)
    else:
        raise ValueError('formulation not valid')

    if formulation_options is not None:
        constraint_formulation.set_options(**formulation_options)
    return constraint_formulation
