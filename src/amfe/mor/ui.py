#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

from copy import deepcopy, copy
import numpy as np
from amfe.solver.tools import MemoizeConstant, MemoizeStiffness
from amfe.solver.translators import MechanicalSystem, create_constrained_mechanical_system_from_component

from .hyper_red.ecsw import ecsw_get_weights_by_component, EcswAssembly
from .hyper_red.poly3 import *


__all__ = ['create_reduced_mechanical_system',
           'reduce_mechanical_system',
           'create_ecsw_hyperreduced_mechanical_system_from_training',
           'create_ecsw_hyperreduced_component_from_weights',
           'create_ecsw_hyperreduced_mechanical_system_from_weights',
           'ecsw_get_weights_from_constrained_training',
           'create_poly3_hyperreduced_system',
           'poly3_get_tensors'
           ]


def create_reduced_mechanical_system(M_func, D_func, K_func, f_int_func, f_ext_func, V, W=None, constant_mass=False,
                                     constant_damping=False):
    """
    Create a mechanical system from a component where the constraints are applied by a constraint formulation

    Parameters
    ----------
    M_func : callable
        Mass matrix function M(x, dx, t)
    D_func : callable
        Linearized viscous damping matrix function D(x, dx, t)
    K_func : callable
        Linearized Stiffness matrix function K(x, dx, t)
    f_int_func : callable
        Internal restoring force function f_int(x, dx, t)
    f_ext_func : callable
        External force function f_ext(x, dx, t)
    V : ndarray
        reduction basis for system state
    W : ndarray, optional
        projection basis or projection, if None (default) W = V
    constant_mass : bool, optional
        Flag indicating if mass matrix is constant
    constant_damping : bool, optional
        Flag indicating if damping matrix is constant

    Returns
    -------
    system : amfe.solver.translators.MechanicalSystem
    """

    if W is None:
        W = V

    def M_red(x, dx, t):
        return W.T @ M_func(V.dot(x), V.dot(dx), t) @ V

    def D_red(x, dx, t):
        return W.T @ D_func(V.dot(x), V.dot(dx), t) @ V

    def K_red(x, dx, t):
        return W.T @ K_func(V.dot(x), V.dot(dx), t) @ V

    def f_int_red(x, dx, t):
        return W.T @ f_int_func(V.dot(x), V.dot(dx), t)

    def f_ext_red(x, dx, t):
        return W.T @ f_ext_func(V.dot(x), V.dot(dx), t)

    if constant_mass:
        M = MemoizeConstant(M_red)
    else:
        M = M_red

    if constant_damping:
        D = MemoizeConstant(D_red)
    else:
        D = D_red

    f_ext = f_ext_red
    f_int = f_int_red
    K = K_red

    dimension = V.shape[1]

    system = MechanicalSystem(dimension, M, D, K, f_ext, f_int)

    return system


def reduce_mechanical_system(mechanical_system, V, W=None, constant_mass=False,
                             constant_damping=False):
    """
    Reduce a mechanical system with reduction basis V and projection basis W

    Parameters
    ----------
    mechanical_system : amfe.solver.translators.MechanicalSystem
        MechanicalSystem to reduce
    V : ndarray
        Reduction basis
    W : ndarray, optional
        Projection basis. If None (default), W = V
    constant_mass : bool
        Flag indicating if reduced system has a constant mass
    constant_damping : bool
        Flag indicating if reduced system has a constant damping

    Returns
    -------
    system : MechanicalSystem
    """

    system = create_reduced_mechanical_system(mechanical_system.M,
                                              mechanical_system.D,
                                              mechanical_system.K,
                                              mechanical_system.f_int,
                                              mechanical_system.f_ext,
                                              V, W, constant_mass, constant_damping)

    return system


def create_ecsw_hyperreduced_mechanical_system_from_weights(component, V, weights, indices, new_formulation,
                                                            constant_mass, constant_damping, copymode='deep',
                                                            tagname='_ecsw_weights',
                                                            **new_formulation_options):
    """
    Creates an ECSW hyperreduced mechanical system for a given component and Ecsw weights and indices
    set.

    Parameters
    ----------
    component : amfe.component.StructuralComponent
        Structural Component to reduce
    V : ndarray
        reduction basis
    weights : ndarray
        weights for the ECSW assembly
    indices : ndarray
        row based indices for the ECSW assembly
    new_formulation : str
        new formulation for the reduced component
    constant_mass : bool
        flag if reduced system has a constant mass
    constant_damping : bool
        flag if reduced system has a constant damping
    copymode : str, {'deep, 'shallow', 'overwrite'}
        copymode that indicates if given component shall be copied, overwritten or shallow copied
    tagname : str, default: '_ecsw_weights'
        tagname for _ecsw_weights that will be added to the component's mesh for analyses and postprocessing
        if None, no tag will be added
    new_formulation_options : dict
        formulation options for the new formulation

    Returns
    -------
    ecsw_sys : amfe.solver.translators.MechanicalSystem
        reduced mechanical system
    ecsw_form : amfe.constraint.ConstraintFormulation
        ConstraintFormulation for the reduced mechanical component
    ecsw_comp : amfe.component.StructuralComponent
        StructuralComponent of the reduced mechanical system

    """

    ecsw_component = create_ecsw_hyperreduced_component_from_weights(component, weights, indices, tagname, copymode)

    ecsw_system, ecsw_formulation = create_constrained_mechanical_system_from_component(ecsw_component,
                                                                                        constraint_formulation=new_formulation,
                                                                                        **new_formulation_options)
    ecsw_red_system = reduce_mechanical_system(ecsw_system, V, constant_mass=constant_mass,
                                               constant_damping=constant_damping)
    return ecsw_red_system, ecsw_formulation, ecsw_component


def create_ecsw_hyperreduced_component_from_weights(component, weights, indices, tagname='_ecsw_weights', copymode='deep'):
    """
    Creates an ECSW hyperreduced component for a given component and Ecsw weights and indices
    set.

    Parameters
    ----------
    component : amfe.component.StructuralComponent
        Structural Component to reduce
    weights : ndarray
        weights for the ECSW assembly
    indices : ndarray
        row based indices for the ECSW assembly
    tagname : str, default: '_ecsw_weights'
        tagname for _ecsw_weights that will be added to the component's mesh for analyses and postprocessing
        if None, no tag will be added
    copymode : str, {'deep, 'shallow', 'overwrite'}
        copymode that indicates if given component shall be copied, overwritten or shallow copied

    Returns
    -------
    ecsw_component : amfe.component.StructuralComponent
        StructuralComponent of the reduced mechanical system

    """
    # If overwrite use existent component, else create new one by copying
    if copymode == 'overwrite':
        ecsw_component = component
    elif copymode == 'shallow':
        ecsw_component = copy(component)
    elif copymode == 'deep':
        ecsw_component = deepcopy(component)
    else:
        raise ValueError("copymode must be 'overwrite', 'shallow' or 'deep', got {}".format(copymode))
    # Create new assembly
    ecswassembly = EcswAssembly(weights, indices)
    # Assign new assembly got reduced_component
    ecsw_component.assembly = ecswassembly
    # create a new tag for ecsw weights
    if tagname is not None:
        mesh = ecsw_component.mesh
        eleids = ecsw_component._ele_obj_df.iloc[indices]['fk_mesh']
        mesh.el_df.loc[eleids, tagname] = weights
        mesh.el_df[tagname] = mesh.el_df[tagname].fillna(0.0)
    return ecsw_component


def create_ecsw_hyperreduced_mechanical_system_from_training(component, formulation, V, x_training, new_formulation,
                                                             constant_mass=False, constant_damping=False,
                                                             timesteps_training=None,
                                                             tau=0.001, copymode='deep', tagname='_ecsw_weights',
                                                             **new_formulation_options):
    """
    Creates an ECSW hyperreduced mechanical system for a given component, its formulation, reduction basis and training
    set.

    Parameters
    ----------
    component : amfe.component.StructuralComponent
        Structural Component to reduce
    formulation : amfe.constraint.ConstraintFormulation
        The constraint formulation of the structural component
    V : ndarray
        reduction basis
    x_training : ndarray
        training set with dimension of the constrained unreduced component
    new_formulation : str
        new formulation for the reduced component
    constant_mass : bool
        flag if reduced system has a constant mass
    constant_damping : bool
        flag if reduced system has a constant damping
    timesteps_training : ndarray
        if timesteps are important for the training set, it can be passed here
    tau : float
        tolerance for the ECSW hyperreduction
    copymode : str, {'deep, 'shallow', 'overwrite'}
        copymode that indicates if given component shall be copied, overwritten or shallow copied
    tagname : str, default: '_ecsw_weights'
        tagname for _ecsw_weights that will be added to the component's mesh for analyses and postprocessing
        if None, no tag will be added
    new_formulation_options : dict
        formulation options for the new formulation

    Returns
    -------
    ecsw_sys : amfe.solver.translators.MechanicalSystem
        reduced mechanical system
    ecsw_form : amfe.constraint.ConstraintFormulation
        ConstraintFormulation for the reduced mechanical component
    ecsw_comp : amfe.component.StructuralComponent
        StructuralComponent of the reduced mechanical system

    """
    weights, indices, stats = ecsw_get_weights_from_constrained_training(x_training, component, formulation, V, tau,
                                                                         timesteps_training)

    ecsw_sys, ecsw_form, ecsw_comp = create_ecsw_hyperreduced_mechanical_system_from_weights(component, V, weights,
                                                                                             indices,
                                                                                             new_formulation,
                                                                                             constant_mass,
                                                                                             constant_damping,
                                                                                             copymode,
                                                                                             tagname,
                                                                                             **new_formulation_options)
    return ecsw_sys, ecsw_form, ecsw_comp


def ecsw_get_weights_from_constrained_training(x_training, component, formulation, V, tau=0.001, timesteps_training=None):
    """
    Computes the weights for a given constrained training set, component, formulation, reduction basis

    Parameters
    ----------
    x_training : ndarray
        training set with dimension of the constrained unreduced component
    component : amfe.component.StructuralComponent
        Structural Component to reduce
    formulation : amfe.constraint.ConstraintFormulation
        The constraint formulation of the structural component
    V : ndarray
        reduction basis
    tau : float
        tolerance for the ECSW hyperreduction

    timesteps_training : ndarray
        if timesteps are important for the training set, it can be passed here

    Returns
    -------
    weights : ndarray
        ecsw weights
    indices : ndarray
        row based indices of elements that have non-zero weights
    stats : ndarray
        convergence stats of the snnls solver

    """
    if timesteps_training is None:
        timesteps_training = np.zeros(x_training.shape[1], dtype=float)
    training_set_expanded = [formulation.u(u, t) for u, t in zip(x_training.T, timesteps_training)]
    training_set_expanded = np.array(training_set_expanded).T
    W = [formulation.u(u, 0.0) for u in V.T]
    W = np.array(W).T
    weights, indices, stats = ecsw_get_weights_by_component(component, training_set_expanded, W, timesteps_training,
                                                            tau)
    return weights, indices, stats


def poly3_get_tensors(system, V, h=1.0):
    x0 = dx0 = np.zeros(system.dimension)
    K_func = system.K
    K1 = V.T @ K_func(x0, dx0, 0.0) @ V
    K2 = compute_quadratic_force_tensor(K_func, V, h, method='central')
    K3 = compute_cubic_force_tensor(K_func, V, h)

    return K1, K2, K3


def create_poly3_hyperreduced_system(system, V, K1, K2, K3, h=1.0, constant_mass=True, constant_damping=True):

    poly3 = Poly3(K1, K2, K3)

    def M_func(x, dx, t):
        return V.T @ system.M(V.dot(x), V.dot(dx), t) @ V

    if constant_mass:
        M_func = MemoizeConstant(M_func)

    def D_func(x, dx, t):
        return V.T @ system.D(V.dot(x), V.dot(dx), t) @ V

    if constant_damping:
        D_func = MemoizeConstant(D_func)

    def f_ext_func(x, dx, t):
        return V.T @ system.f_ext(V.dot(x), V.dot(dx), t)

    f_int_func = MemoizeStiffness(poly3.K_and_f_int)
    K_func = f_int_func.derivative

    poly3_system = MechanicalSystem(V.shape[1], M_func, D_func, K_func, f_ext_func, f_int_func)

    return poly3_system
