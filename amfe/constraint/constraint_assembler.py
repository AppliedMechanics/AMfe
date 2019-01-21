#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module for assembling the constraint-objects to a set of global constraint-operators
"""

from scipy.sparse import csr_matrix, lil_matrix, identity, vstack
from scipy.linalg import null_space
from .constraint import DirichletConstraint
from math import isclose
import numpy as np


class ConstraintAssembler:
    def __init__(self):
        pass
    
    def assemble_elim_C_L_and_g(self, no_of_unconstrained_dofs, constrained_dofs, constraint_df, X, u, du, ddu, t, primary_type='u', C_preset=None, L_preset=None):
        """
        Update the global constraint-matrices C, L, g
        
        Parameters
        ----------
        no_of_unconstrained_dofs: int
            number of dofs without any constraints applied
        constraint_df: pandas dataframe
            dataframe containing all necessary information on constraint-types and constraint-objects themselves
        primary_type: string
            defines the primary type and therefore the chosen jacobian and residual 
                default value: underived primary 'u' 
                other choices: derived primary 'du', double derived primary 'ddu'

        Returns
        -------
        C: csr-matrix
            matrix containing jacobians of constraint functions
        L: csr-matrix
            nullspace of C_elim and containing all information on how to eliminate dofs and changing their relations
        g: csr-vector
            residual vector for constraints enforced by lagrange-multipliers
        """
        C, g, C_is_boolean = self._assemble_constraints(no_of_unconstrained_dofs, constraint_df, X, u, du, ddu, t, 'elim', primary_type, C_preset)

        if L_preset is not None:
            L = L_preset
        else:
            if C is not None:
                L = self._update_l(C, C_is_boolean, constrained_dofs)
            else:
                L = None
        
        return C, L, g
    
    def assemble_lagr_C_g(self, no_of_unconstrained_dofs, constraint_df, X, u, du, ddu, t, primary_type='u', C_preset=None):
        C, g, C_is_boolean = self._assemble_constraints(no_of_unconstrained_dofs, constraint_df, X, u, du, ddu, t, 'lagrmult', primary_type, C_preset)
        return C, g

    
    def _assemble_constraints(self, no_of_unconstrained_dofs, constraint_df, X, u, du, ddu, t, strategy='elim', primary_type='u', C_preset=None):
        ndof = no_of_unconstrained_dofs
        
        g = self._build_constraint_func(constraint_df, X, u, du, ddu, t, strategy)

        if C_preset is not None:
            C = C_preset
        else:
            if g is not None:
                C, C_is_boolean = self._build_C(ndof, constraint_df, g.shape[0], primary_type, X, u, du, ddu, t, strategy)
            else:
                C = None
        
        if g is not None and C is not None:        
            if g.shape[0] is not C.shape[0]:
                print('Missmatch of g- and C-dimension! Maybe an error in constraint-definition?')
        
        return C, g, C_is_boolean
    
    def _build_C(self, ndof, constraint_df, no_constr, primary_type, X, u, du, ddu, t, strategy):
        """
        Update the assembled jacobians of constraint-functions

        Returns
        -------
        csr-matrix
        """

        if no_constr > 0:
            C = csr_matrix((0, ndof), dtype=float)
        else:
            C = None
            
        C_is_boolean = True


        for iter, const in constraint_df.iterrows():
            if const['strategy'] == strategy:
                dofs = const['dofids']
                C_helper = csr_matrix(const['constraint_obj'].jacobian(X[dofs], u[dofs], du[dofs], ddu[dofs], t, primary_type))
                C_expanded = lil_matrix((C_helper.shape[0], ndof), dtype=float)
                C_expanded[:,dofs] = C_helper
                C_expanded = csr_matrix(C_expanded)

                C = vstack([C, C_expanded])
                
                if not isinstance(const['constraint_obj'], DirichletConstraint):
                    C_is_boolean = False
                
        return C, C_is_boolean

    
    def _build_constraint_func(self, constraint_df, X, u, du, ddu, t, strategy):
        """
        Update the residual of constraints for dual enforcement of constraints with Lagrange-multipliers

        Returns
        -------
        ndarray
        """
        g = np.array([])       
        for iter, const in constraint_df.iterrows():
            if const['strategy'] == strategy:
                dofs = const['dofids']
                g = np.append(g, const['constraint_obj'].constraint_func(X[dofs], u[dofs], du[dofs], ddu[dofs], t))
            
        return g
    
    def _update_l(self, C, C_is_boolean=None, one_idxs=None):
        """
        Update the L matrix that eliminates the dofs that are eliminated by constraints.

        Returns
        -------
        csr-matrix
        """
        if C_is_boolean:
            l = self._get_nullspace_from_identity(C, one_idxs)
        else:
            l = csr_matrix(null_space(C.todense()))
        return l
    
    def _get_nullspace_from_identity(self, C, col_idxs):
        """
        Calculates a(!) nullspace of C if C is boolean by deleting the rows of C from an identity matrix
        
        Parameters
        ----------
        C: csr_matrix
        col_idxs: ndarray
            Indices of ones in C-matrix => provides columns, which have to be deleted from L-matrix
        
        Returns
        -------
        L: csr_matrix
        """
        L = identity(C.shape[1])
        L = L.tocsr()
        col_idxs_not_to_remove = np.arange(0,C.shape[1])
        col_idxs_not_to_remove = np.delete(col_idxs_not_to_remove, col_idxs)
        return L[:,col_idxs_not_to_remove]
    
    
            