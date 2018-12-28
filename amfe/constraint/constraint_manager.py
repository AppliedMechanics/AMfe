#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module having a base class for ConstraintManager

On constraint-implementation see literature:
Gould, N.I.M. e.a. (1998) - On the Solution of Equality Constrained Quadratic Programming Problems Arising in Optimization
"""

import pandas as pd
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from .constraint_assembler import ConstraintAssembler
from .constraint import DirichletConstraint, FixedDistanceConstraint
    
class ConstraintManager:
    
    # Strategies to apply constraints
    # currently: elimination and lagrange multiplier
    STRATEGIES = [
        'elim',
        'lagrmult'
    ]
    
    def __init__(self, ndof_unconstrained_system=0):
        """
        Parameters
        ----------
        ndof_unconstrained_system : int
            number of dofs of the unconstrained system.
        """
        super().__init__()
        self._no_of_unconstrained_dofs = ndof_unconstrained_system
        self._constraints_df = pd.DataFrame(columns=['name', 'constraint_obj', 'dofids', 'strategy'])
        self._constraints_df['name'] = self._constraints_df['name'].astype('object')
        self._constraints_df['dofids'] = self._constraints_df['dofids'].astype('int64')
        self._constraints_df['strategy'] = self._constraints_df['strategy'].astype('object')
        
        self._constraint_assembler = ConstraintAssembler()
        self._L = None
        self._C_elim = None
        self._C_lagr = None
        self._g_elim = None
        self._g_lagr = None
        return
    
    @staticmethod
    def create_fixed_distance_constraint():
        """
        Create a constraint that preserves a fixed distance

        Returns
        -------
        constraint: amfe.constraint.FixedDistanceConstraint
        """
        return FixedDistanceConstraint()

    @staticmethod
    def create_dirichlet_constraint(no_of_dofs, U=lambda t: 0., dU=lambda t: 0., ddU=lambda t: 0.):
        """
        Create a Dirichlet constraint

        Parameters
        ----------
        no_of_dofs: int
            number of dofs that shall be constrained
        U: func
            function f(t) describing the prescribed field
        dU: func
            function f(t) describing the first time derivative of the prescribed field
        ddU: func
            function f(t) describing the second time derivative of the prescribed field

        Returns
        -------
        constraint: amfe.constraint.DirichletConstraint
        """
        return DirichletConstraint(no_of_dofs, U, dU, ddU)
    
    def add_constraint(self, name, constraint_obj, dofids, strategy):
        """
        Method for adding a constraint

        Parameters
        ----------
        name: str
            A string describing the name of the constraint (can be any name)
        constraint : amfe.constraint.ConstraintBase
            constraint object, describing the constraint
        dofsarg : tuple
            dofs' indices that must be passed to the constraint
        strategy : str {'elim', 'lagrmult'}
            strategy how the constraint shall be applied (e.g. via elimination or lagrange multiplier)

        Returns
        -------
        None
        """

        if strategy not in self.STRATEGIES:
            raise ValueError('strategy must be \'elim\' or \'lagrmult\'')

        # Create new rows for constraints_df
        df = pd.DataFrame(
            {'name': name, 'constraint_obj': constraint_obj, 'dofids': [dofids], 'strategy': strategy})
        self._constraints_df = self._constraints_df.append(df, ignore_index=True)
        return
    
    def remove_constraint_by_name(self, name):
        indices = self._constraints_df.index[self._constraints_df['name'] == name].tolist()
        self._remove_constraint_by_indices(indices)
        
    def remove_constraint_by_dofids(self, dofids):
        """
        Removes constraints from dataframe, that are applied to chosen dofs
        WARNING: This removes the whole row! So if the constraint is applied to further dofs, where you want to keep the constraint, 
        you have to reapply the constraint 
         
        Parameters
        ----------
        dofids: int64 list
        
        Returns
        -------
        None
        """
        indices = []
        for dof in dofids:
            for indx, constr in self._constraints_df.iterrows():
                if [dof] == constr['dofids']:
                    indices.append(indx)
                    
        print(indices)
                
        self._remove_constraint_by_indices(indices)
        
    def _remove_constraint_by_indices(self, indices):
        self._constraints_df = self._constraints_df.drop(indices)
        self._constraints_df = self._constraints_df.reset_index(drop=True)
    
    def constrain_matrix(self, matrix):
        if self.L is None:
            return matrix
        return self.L.T @ matrix @ self.L
    
    def constrain_vector(self, vector):
        if self.L is None:
            print('No elimination constraint set')
            return vector
        else:
            return self.L.T @ vector
    
    def unconstrain_vector(self, free_vector, constrained_vector=None):
        if self.C_elim is None:
            return free_vector
        else:
            if constrained_vector is None:
                constrained_vector = self.constrained_dofs
                
            return self.L @ free_vector + self.C_elim.T @ constrained_vector
        
    def get_constrained_coupling_quantities(self, matrix):
        """
        Calculate coupling terms of off-diagonals in system matrix, which couples constraints and free unknowns
        
        Parameters
        ----------
        matrix: csr-matrix
            tangent system-matrix of unconstrained linearized system of equations
            
        Returns
        -------
        coupling-forces: csr-matrix
            forces of system-matrix caused by constraints projected into free solution-space of L
        """
        return csr_matrix((self.L.T @ matrix @ self.C_elim.T) @ self.constrained_dofs).T
    
    def update_no_of_unconstrained_dofs(self, new_no_of_unconstrained_dofs):
        self._no_of_unconstrained_dofs = new_no_of_unconstrained_dofs
    
    def update_constraints(self, X=None, u=None, du=None, ddu=None, t=0):
        self._update_elim_constraints(X, u, du, ddu, t)
        self._update_lagr_constraints(X, u, du, ddu, t)
        
    @property
    def constrained_dofs(self):
        """
        Values of eliminated dofs
        
        Parameters
        ----------
        var: ndarray
            the whole unconstrained vector of primary solution
        
        Returns
        -------
        constrained_dofs: ndarray
            the part of the solution in the constrained subspace
        """
        self._constrained_dofs = spsolve(self.C_elim @ self.C_elim.T, -self._g_elim)
        return self._constrained_dofs

    @property
    def L(self):
        """
        Returns
        -------
        L : ndarray
            Retuns the L matrix that eliminates the dofs that are eliminated by constraints
        """

        return self._L
    
    @property
    def C_elim(self):
        """
        Returns
        -------
        C : ndarray
            Retuns the C matrix containing linearized constraints for applying lagrange multipliers
        """
        return self._C_elim
    
    def residual_elim(self, var):
        """
        Parameters
        ----------
        var: ndarray
            Full, unconstrained dof-values
            
        Returns
        -------
        g : ndarray
            Retuns the g vector containing the constraint-residuals when applying lagrange multipliers
        """
        return self.C_elim @ var - self._g_elim
    
    @property
    def C_lagr(self):
        """
        Returns
        -------
        C : ndarray
            Retuns the C matrix containing linearized constraints for applying lagrange multipliers
        """

        return self._C_lagr
    
    def residual_lagr(self, var):
        """
        Returns
        -------
        g : ndarray
            Retuns the g vector containing the constraint-residuals when applying lagrange multipliers
        """

        return self.C_lagr @ var - self._g_lagr

    @property
    def no_of_constrained_dofs(self):
        """
        Gives the number of dofs of the constrained system

        Returns
        -------
        no_of_constrained_dofs : int
            Number of dofs of the constrained system
        """
        if self.C_elim is None:
            return self._no_of_unconstrained_dofs
        else:
            return self._no_of_unconstrained_dofs - self.C_elim.shape[0]
        
    def _update_elim_constraints(self, X, u, du, ddu, t):
        self._C_elim, self._L, self._g_elim = self._constraint_assembler.assemble_elim_C_L_and_g(self._no_of_unconstrained_dofs, self._constraints_df, X, u, du, ddu, t)
        
    def _update_lagr_constraints(self, X, u, du, ddu, t):
        self._C_lagr, self._g_lagr = self._constraint_assembler.assemble_lagr_C_g(self._no_of_unconstrained_dofs, self._constraints_df, X, u, du, ddu, t)
        if self._L is not None and self._C_lagr is not None:
            self._C_lagr = self._C_lagr @ self._L
        
    def _get_dofids_by_strategy(self, strategy):
        if strategy in self._constraints_df['strategy'].values:
            dofids_per_strat = self._constraints_df.groupby('strategy')['dofids'].aggregate(lambda dof: dof)
            dofids = np.unique(np.array(dofids_per_strat[strategy]))
        else:
            dofids = np.empty([0,0])
            
        return dofids
