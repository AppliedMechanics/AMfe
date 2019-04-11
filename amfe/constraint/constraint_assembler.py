#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Module for assembling the constraint-objects to a set of global constraint-operators
"""

from scipy.sparse import csr_matrix
import numpy as np


class ConstraintAssembler:
    def __init__(self):
        return

    @staticmethod
    def preallocate_g_and_B(no_of_dofs_unconstrained, dofs_by_object, no_of_constraints_by_object):
        """
        Preallocates B array and g vector for given dofs concerning constraints

        Parameters
        ----------
        no_of_dofs_unconstrained : int
            number of dofs of the unconstrained system (number of columns of B)
        dofs_by_object : list of ndarrays with dtype int
            list containing indices (position based) of dofs that are concerned by constraints
            dofs[i] = np.array([dof1, dof2, ...], dtype=int)
            where dof1, dof2,... are all dofs that are concerned by the i-th constraint
        no_of_constraints_by_object : tuple
            containing the number of constraints by later passed constraint object

        Returns
        -------
        g : ndarray
            ndarray for storing the holonomic constraint function
        B : csr_matrix
            csr_matrix for storing the Jacobian of the holonomic constraint function g(x) = 0
        """
        # Initialize indptr and indices
        indptr = [0]
        indices = list()

        # Generate indptr and indices values
        for dofs, no_of_constraints in zip(dofs_by_object, no_of_constraints_by_object):
            for _ in np.arange(no_of_constraints):
                indptr.append(indptr[-1] + len(dofs))
                indices.extend(dofs.tolist())

        # Make empty data array of correct size
        data = np.zeros(len(indices))

        no_of_constraint_rows = len(indptr) - 1
        B = csr_matrix((data, indices, indptr), shape=(no_of_constraint_rows, no_of_dofs_unconstrained))
        g = np.zeros(no_of_constraint_rows)
        return g, B

    def assemble_g(self, residuals, dofs, args, g):
        """
        Assemble the holonomic constraint function

        Parameters
        ----------
        residuals: list
            list of function handles returning the residual for each constraint
        dofs : list
            list of coordinate indices that must be picked from the global arrays in args for each residual function
        args : list
            list of full global arrays whose local coordinates needs to be passed to the residuals
        g : ndarray
            preallocated ndarray where the result shall be written to

        Returns
        -------
        g : ndarray
            assembled holonomic constraint function

        """
        return self._build_g(residuals, dofs, args, g)

    def assemble_B(self, jacobians, dofs, args, B):
        """
        Assembles the Jacobians of a holonomic constraint function

        Parameters
        ----------
        jacobians : list
            list of function handles that return the local B for each constraint
        dofs : list
            list of ndarrays (dtype int) containing the indices of the dofs that are passed to the B functions
        args : list
            list of full global arrays whose local coordinates needs to be passed to the jacobians
        B : csr_matrix
            preallocated ndarray where the result shall be written to

        Returns
        -------
        B : csr_matrix
            assembled B
        """
        return self._build_B(jacobians, dofs, args, B)

    def assemble_g_and_B(self, residuals, jacobians, dofs, args, g, B):
        """
        Assemble the holonomic constraint function

        Parameters
        ----------
        residuals: list
            list of function handles returning the residual for each constraint
        jacobians : list
            list of function handles that return the local B for each constraint
        dofs : list
            list of coordinate indices that must be picked from the global arrays in args for each residual function
        args : list
            list of full global arrays whose local coordinates needs to be passed to the residuals
        g : ndarray
            preallocated ndarray where the assembled residual shall be written to
        B : csr_matrix
            preallocated csr_matrix where the assembled B shall be written to

        Returns
        -------
        g : ndarray
            assembled holonomic constraint function
        B : csr_matrix
            assembled B

        """
        # Assemble B
        self._build_B(jacobians, dofs, args, B)

        # Assemble holonomic constraint function
        self._build_g(residuals, dofs, args, g)

        return g, B

    @staticmethod
    def _build_B(jacobians, dofs, args, B):
        """
        Assembles the Jacobians of a holonomic constraint function

        Parameters
        ----------
        jacobians : list
            list of function handles that return the local B for each constraint
        dofs : list
            list of ndarrays (dtype int) containing the indices of the dofs that are passed to the B functions
        args : list
            list of full global arrays whose local coordinates needs to be passed to the jacobians
        B : csr_matrix
            preallocated ndarray where the result shall be written to

        Returns
        -------
        B : csr_matrix
            assembled B
        """
        B.data *= 0
        B.data = np.concatenate([jac(*(arg[dofs_i] for arg in args)).reshape(-1) for jac, dofs_i in zip(jacobians,
                                                                                                        dofs)])
        return B

    @staticmethod
    def _build_g(residuals, dofs, args, g):
        """
        Assemble the holonomic constraint function

        Parameters
        ----------
        residuals: list
            list of function handles returning the residual for each constraint
        dofs : list
            list of coordinate indices that must be picked from the global arrays in args for each residual function
        args : list
            list of full global arrays whose local coordinates needs to be passed to the residuals
        g : ndarray
            preallocated ndarray where the result shall be written to

        Returns
        -------
        g : ndarray
            assembled holonomic constraint function

        """
        g[:] = np.concatenate([res(*(arg[dofs_i] for arg in args)) for res, dofs_i in zip(residuals, dofs)])
        return g
