#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module providing a ConstraintManager

On constraint-implementation see literature:
Gould, N.I.M. e.a. (1998) - On the Solution of Equality Constrained Quadratic Programming Problems Arising in
Optimization
"""

import numpy as np
import pandas as pd

from .constraint_assembler import ConstraintAssembler
from .constraint import DirichletConstraint, FixedDistanceConstraint


class ConstraintManager:
    """
    The Constraint Manager couples constraint objects based from HolonomicConstraintClass
    with the Assembler.
    It thus provides functions:
        - creation methods to create constraint objects that can be passed to the add method of the manager
        - to add and remove constraint objects
        - to ask for global vectors/matrices coming from algebraic constraint equations
        - general state properties to ask the state of the constraint manager, e.g. no_of_constraints

    Attributes
    ----------
    _no_of_dofs_unconstrained: int
        number of dofs of the unconstrained system (length of global vectors that are provided when an entity is asked
        for
    _constraints_df: pandas.DataFrame
        pandas DataFrame containing information about applied constraints: name, objects, Xidxs to be passed,
        dofidxs to be passed
    _constraint_assembler: amfe.constraint_assembler.ConstraintAssembler
        an assembler object that is able to assemble the global entities
    _B: csr_matrix
        preallocated csr_matrix to assemble global B
    _g: numpy.array
        preallocated numpy.array to assemble global g (holonomic constraint function)
    _update_flag: bool
        internal flag that indicates if some things must be updated when they are asked for or not
    """

    def __init__(self, ndof_unconstrained_system=0):
        """
        Parameters
        ----------
        ndof_unconstrained_system : int
            number of dofs of the unconstrained system.
        """
        super().__init__()
        self._no_of_dofs_unconstrained = ndof_unconstrained_system
        self._update_flag = True
        self._constraints_df = pd.DataFrame(columns=['name', 'constraint_obj', 'Xidxs', 'dofidxs'])
        self._constraints_df['name'] = self._constraints_df['name'].astype('object')
        self._constraints_df['Xidxs'] = self._constraints_df['Xidxs'].astype('object')
        self._constraints_df['dofidxs'] = self._constraints_df['dofidxs'].astype('object')
        
        self._constraint_assembler = ConstraintAssembler()

        self._B = None
        self._g = None
        self._update_flag = True
        return

    @property
    def no_of_constraint_definitions(self):
        """
        Returns the number of constraints that have different names

        Returns
        -------
        n: int
            number of constraints that have different names
        """
        return len(self._constraints_df['name'].unique())

    @property
    def no_of_constraints(self):
        """
        Returns the total number of degrees of freedom that will be constrained by all constraints together

        Returns
        -------
        n: int
            number of degrees of freedom that will be constrained by all constraints
        """
        return np.sum(self._no_of_constraints_by_object())

    @property
    def no_of_dofs_unconstrained(self):
        """
        Returns the number of dofs of the unconstrained system, i.e. this defines the length of the global vectors
        that are passed to the global vector and matrix getter functions

        Returns
        -------
        n: int
            number of dofs of the unconstrained system that is assumed
        """
        return self._no_of_dofs_unconstrained

    @no_of_dofs_unconstrained.setter
    def no_of_dofs_unconstrained(self, new_no_of_dofs_unconstrained):
        """
        Setter for number of dofs of the unconstrained system that is assumed

        Parameters
        ----------
        new_no_of_dofs_unconstrained: int
            new number of dofs of the unconstrained system that is assumed

        Returns
        -------
        None
        """
        self._no_of_dofs_unconstrained = new_no_of_dofs_unconstrained
        self._update_flag = True
    
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
    def create_dirichlet_constraint(U=lambda t: 0., dU=lambda t: 0., ddU=lambda t: 0.):
        """
        Create a Dirichlet constraint

        Parameters
        ----------
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
        return DirichletConstraint(U, dU, ddU)

    def add_constraint(self, name, constraint_obj, dofidxs, Xidxs=()):
        """
        Method for adding a constraint

        Parameters
        ----------
        name: str
            A string describing the name of the constraint (can be any name)
        constraint_obj : amfe.constraint.ConstraintBase
            constraint object, describing the constraint
        dofidxs : tuple
            dofs' indices that must be passed to the constraint
        Xidxs : tuple
            indices of the reference coordinates that might have to be passed to the constraint

            ATTENTION: Whether this is needed or not depends on the constraint's type. Take a look at the constraint-
            classes' documentation!
        """
        print('Adding constraint {} to dofs {} and nodes {}'.format(name, dofidxs, Xidxs))

        # Create new rows for constraints_df
        df = pd.DataFrame(
            {'name': name, 'constraint_obj': constraint_obj,
             'dofidxs': [np.array([dofidxs], dtype=int).reshape(-1)],
             'Xidxs': [np.array([Xidxs], dtype=int).reshape(-1)]},
            )

        self._constraints_df = self._constraints_df.append(df, ignore_index=True)
        constraint_obj.after_assignment(dofidxs)

        self._update_flag = True
        
        return
    
    def remove_constraint_by_name(self, name):
        """
        Removes a constraint by its given name

        Parameters
        ----------
        name : str
            name of the constraint
        """
        indices = self._constraints_df.index[self._constraints_df['name'] == name].tolist()
        self._remove_constraint_by_indices(indices)
        self._update_flag = True
        
    def remove_constraint_by_dofidxs(self, dofidxs):
        """
        Removes constraints from dataframe, that are applied to chosen dofs
        WARNING: This removes the whole row! So if the constraint is applied to further dofs, where you want to keep the
        constraint,
        you have to reapply the constraint 
         
        Parameters
        ----------
        dofidxs: int64 list
        
        Returns
        -------
        None
        """
        indices = []
        for dof in dofidxs:
            for indx, constr in self._constraints_df.iterrows():
                if [dof] == constr['dofidxs']:
                    indices.append(indx)

        self._remove_constraint_by_indices(indices)
        self._update_flag = True
        
    def _remove_constraint_by_indices(self, indices):
        self._constraints_df = self._constraints_df.drop(indices)
        self._constraints_df = self._constraints_df.reset_index(drop=True)

    def update(self):
        """
        Function that is called by observers or called internally when something is asked and the update flag is True

        Returns
        -------
        None
        """
        self._g, self._B = self._constraint_assembler.preallocate_g_and_B(self._no_of_dofs_unconstrained,
                                                                          self._dofidxs(),
                                                                          self._no_of_constraints_by_object())

    def _no_of_constraints_by_object(self):
        """
        Returns the number of dofs that are constrained the constraint objects defined in the DataFrame

        Returns
        -------
        no_of_constraints_by_object: list
            list of ints that containt the number of dofs 'removed' by the constrained
        """
        return [const['constraint_obj'].NO_OF_CONSTRAINTS for i, const in self._constraints_df.iterrows()]

    def _dofidxs(self):
        """
        Returns a list with ndarrays containing the local dofidxs that must be passed to the constraints

        Returns
        -------
        dofidxs: list
            list of ndarrays containing the local dofidxs that must be passed to the constraints
        """
        return [const['dofidxs'] for i, const in self._constraints_df.iterrows()]

    def _Bs(self, X, t):
        """
        Returns the B functions that now only have u as input function because u must be separated to local dofs
        defined by the dofidxs

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        t: float
            time

        Returns
        -------
        jacs: generator
            generator object that yields the B function with correct signature for the assembler
        """
        for i, const in self._constraints_df.iterrows():
            X_local = X[const['Xidxs']]

            def B(u):
                return const['constraint_obj'].B(X_local, u, t)
            yield B

    def _gs(self, X, t):
        """
        Returns the holonomic g functions that now only have u as input function because u must be separated to local
        dofs defined by the dofidxs

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        t: float
            time

        Returns
        -------
        ress: generator
            generator object that yields the g functions with correct signature for the assembler
        """
        for i, const in self._constraints_df.iterrows():
            X_local = X[const['Xidxs']]

            def g(u):
                return const['constraint_obj'].g(X_local, u, t)

            yield g

    def _as(self, X, t):
        """
        Returns the a functions that now only have u and du as input function because these must be separated to local
        dofs defined by the dofidxs

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        t: float
            time

        Returns
        -------
        ress: generator
            generator object that yields the g functions with correct signature for the assembler
        """
        for i, const in self._constraints_df.iterrows():
            X_local = X[const['Xidxs']]

            def a(u, du):
                return const['constraint_obj'].a(X_local, u, du, t)

            yield a

    def _bs(self, X, t):
        """
        Returns the b functions that now only have u as input function because u must be separated to local
        dofs defined by the dofidxs

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        t: float
            time

        Returns
        -------
        ress: generator
            generator object that yields the g functions with correct signature for the assembler
        """
        for i, const in self._constraints_df.iterrows():
            X_local = X[const['Xidxs']]

            def b(u):
                return const['constraint_obj'].b(X_local, u, t)

            yield b

    def g_and_B(self, X, u, t):
        """
        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        u: numpy.array
            numpy.array containing the global displacements of the dofs of the system
        t: float
            time

        Returns
        -------
        g : ndarray
            the global holonomic constraint function residual
        B : csr_matrix
            Returns the B matrix containing linearized constraints for applying lagrange multipliers
        """
        if self._update_flag:
            self.update()
        g, B = self._constraint_assembler.assemble_g_and_B(self._gs(X, t), self._Bs(X, t),
                                                           self._dofidxs(), (u,), self._g, self._B)
        return g, B

    def B(self, X, u, t):
        """

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        u: numpy.array
            numpy.array containing the global displacements of the dofs of the system
        t: float
            time

        Returns
        -------
        B : csr_matrix
            Returns the B matrix containing linearized constraints for applying lagrange multipliers
        """
        if self._update_flag:
            self.update()
        return self._constraint_assembler.assemble_B(self._Bs(X, t, ), self._dofidxs(), (u,), self._B)

    def g(self, X, u, t):
        """

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        u: numpy.array
            numpy.array containing the global displacements of the dofs of the system
        t: float
            time

        Returns
        -------
        g : ndarray
            the global holonomic constraint function residual
        """
        if self._update_flag:
            self.update()
        return self._constraint_assembler.assemble_g(self._gs(X, t), self._dofidxs(), (u,), self._g)

    def a(self, X, u, du, t):
        """

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        u: numpy.array
            numpy.array containing the global displacements of the dofs of the system
        du: numpy.array
            numpy.array containing the global velocities of the dofs of the system
        t: float
            time

        Returns
        -------
        a : ndarray
            the inhomogeneous part of the constraint function on acceleration level
        """
        if self._update_flag:
            self.update()
        a = self._g.copy()
        a *= 0.0
        a = self._constraint_assembler.assemble_g(self._as(X, t), self._dofidxs(), (u, du), a)
        return a

    def b(self, X, u, t):
        """

        Parameters
        ----------
        X: numpy.array
            numpy.array containing the coordinates of the nodes of the system
        u: numpy.array
            numpy.array containing the global displacements of the dofs of the system
        t: float
            time

        Returns
        -------
        b : ndarray
            the inhomogeneous part of the constraint function on velocity level
        """
        if self._update_flag:
            self.update()
        b = self._g.copy()
        b *= 0.0
        b = self._constraint_assembler.assemble_g(self._bs(X, t), self._dofidxs(), (u, ), b)
        return b
