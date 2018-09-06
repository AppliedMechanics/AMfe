#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Constrained mechanical system.
"""

import numpy as np
from scipy import zeros
from scipy.linalg import eigvalsh
from scipy.sparse import vstack, hstack, csr_matrix

from .mechanical_system import MechanicalSystem
from ..assembly import AssemblyConstraint

__all__ = [
    'ConstrainedMechanicalSystem'
]


class ConstrainedMechanicalSystem(MechanicalSystem):
    '''
    Mechanical System with constraints
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # properties for constraints
        self.number_of_constraints = 0
        self.scaling = 1
        self.penalty = 1
        self.assembly_class = AssemblyConstraint(self.mesh_class)
        self.lambda_output = []
        self.characteristic_length = 1

    def extend_K(self, K, B, dt=1):
        """
        Function that receives the K matrix and the Jacobian of the Constraints
        (B) and returns the extended matrix K_extended already scaled when
        necessary.
        """
        # use sp.sparse.hstack and sp.sparse.vstack for better speed
        K = K + B.T.dot(B) * self.scaling * self.penalty
        return vstack([hstack([K, B.T*self.scaling], format='csr'),
                       hstack([B*self.scaling, csr_matrix((B.shape[0], B.shape[0]))], format='csr')],
                      format='csr')  # FIXME: CSR or CSC matrix !?

    def M(self, u=None, t=0):
        '''
        Compute the extended mass matrix of the dynamical system.

        This method computes:

            -           -
            |  M  |  0  |
        M = |-----+-----|
            |  0  |  0  |
            -           -

        '''

        ndof = self.dirichlet_class.no_of_constrained_dofs
        ndof_const = self.number_of_constraints

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)

        M = MechanicalSystem.M(self, u[:ndof], t)

        return vstack([hstack([M, csr_matrix((ndof, ndof_const))], format='csr'),
                       hstack([csr_matrix((ndof_const, ndof)), csr_matrix((ndof_const, ndof_const))], format='csr')],
                      format='csr')  # FIXME: CSR or CSC matrix !?

    def K(self, u=None, t=0):
        '''
        Compute the extended stiffness matrix of the mechanical system

        This method computes:
            -                                -
            | K  +  s p B^T B   |   s B^T    |
        K = |-------------------+------------|
            |       s B         |     0      |
            -                                -


        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation
        t : float, optional
            Time

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse csr-format
        '''
        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)

        ndof = self.dirichlet_class.no_of_constrained_dofs
        ndof_const = self.number_of_constraints

        K = MechanicalSystem.K(self, u[:ndof], t)

        # extend K if needed
        if ndof_const is not 0:
            B = self.B(u[:ndof], t)
            K_extended = self.extend_K(K, B)
        else:
            K_extended = K

        return K_extended

    def f_int(self, u, t=0, dt=1):
        '''
        Return the extended elastic restoring force of the system


        This method returns:

        -                                   -
        |f_int + s B^T ( lambda + p C(u) )  |
        |  s C(u)                           |
        -                                   -

        '''
        ndof = self.dirichlet_class.no_of_constrained_dofs
        ndof_const = self.number_of_constraints

        f_int = MechanicalSystem.f_int(self, u[:ndof], t)
        f_int_extended = np.zeros(ndof+ndof_const)
        f_int_extended[:ndof] = f_int
        if ndof_const is not 0:
            B = self.B(u[:ndof],t)
            C = self.C(u[:ndof],t)

            f_int_extended[:ndof] += B.T.dot(u[ndof:]+ \
                                             self.penalty*C)*self.scaling
            f_int_extended[ndof:] += self.scaling*C
        return f_int_extended

    def f_ext(self, u, du, t):
        '''
        Return the extended external force of the system


        This method returns:

        -       -
        |f_ext  |
        | sC(u) |
        -       -

        '''
        ndof = self.dirichlet_class.no_of_constrained_dofs
        ndof_const = self.number_of_constraints

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)
            du = np.zeros_like(u)

        f_ext = MechanicalSystem.f_ext(self, u[:ndof], du[:ndof], t)
        f_ext_extended = np.zeros(ndof+ndof_const)
        f_ext_extended[:ndof] = f_ext

        return f_ext_extended

    def K_and_f(self, u=None, t=0, dt=1):
        '''
        Returns tangential stiffness K and f_int of constrained system

        This system returns the K and f_int according to the methods K  and f_int
        See documentation there
        '''
        ndof = self.dirichlet_class.no_of_constrained_dofs
        ndof_const = self.number_of_constraints
        # pass a right sized u for the original K_and_f function
        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs +
                         ndof_const)
        K, f = MechanicalSystem.K_and_f(self, u[:ndof], t)

        # extend K and f if needed
        if ndof_const is not 0:
            B = self.B(u[:ndof], t)
            C = self.C(u[:ndof],t)

            K_extended = self.extend_K(K, B, dt)
            f_extended = np.zeros(ndof+ndof_const)
            f_extended[:ndof] = f + B.T.dot(u[ndof:]+\
                                                  self.penalty*C)*self.scaling
            f_extended[ndof:] = self.scaling*C
        else:
            K_extended = K
            f_extended = f

        return K_extended, f_extended

    def C(self, u, t):
        '''
        Return the residual of the constraints.

        The constraints are given in the canonical form C=0. This function
        returns the residual of the constraints, i.e. C=res.

        Parameters
        ----------
        u : ndarray
            generalized position
        du : ndarray
            generalized velocity
        t : float
            current time

        Returns
        -------
        C : ndarray
            residual vector of the constraints

        '''

        return self.assembly_class.assemble_C(self.unconstrain_vec(u), t)

    def B(self, u, t):
        '''
        Return the Jacobian B of the constraints.

        The Jacobian matrix of the constraints B is the partial derivative of
        the constraint vector C with respect to the generalized coordinates q,
        i.e. B = dC/dq

        Parameters
        ----------
        u : ndarray
            generalized position
        du : ndarray
            generalized velocity
        t : float
            current time

        Returns
        -------
        B : ndarray
            Jacobian of the constraint vector with respect to the generalized
            coordinates
        '''
        B_unconstr = self.assembly_class.assemble_B(self.unconstrain_vec(u), t)
        B_constr_t = self.constrain_vec(B_unconstr.T)
        return B_constr_t.T

    def calculate_scaling_factor(self, dt=1.0):
        """
        Calculates the scaling factor and get the number of degrees of freedom
        of the system.

        Parameters
        ----------
        dt : float
            time step width

        Returns
        -------
        scaling : float
            scaling factor for the constraints to have the same order of
            magnitude as the matrices M, D and K.
        """
        self.ndof = self.dirichlet_class.no_of_constrained_dofs
        # or use self.K().shape[0]

        u0 = zeros(self.ndof)

        M = MechanicalSystem.M(self, u0) #self.M(u=u0)
        mr = M.diagonal().sum()/self.ndof # is faster than calculating the norm
        # mr = sp.sparse.linalg.norm(M)

        K = MechanicalSystem.K(self, u0)
        kr = K.diagonal().sum()/self.ndof
        # kr = sp.sparse.linalg.norm(K)

        self.scaling = kr + mr/(dt**2) # + dr/dt
        self.penalty = 1

    def apply_constraint(self, constraint_name, key1, key2=0, coord='xyz',
                         mesh_prop1='phys_group',
                         mesh_prop2='phys_group'):
        """
        Applies specific types of constraints depending on their name, given
        keys, coordinates and mesh properties.
        """
        self.mesh_class.create_constraint(constraint_name, key1, key2, coord,
                                          mesh_prop1, mesh_prop2)
        self.number_of_constraints = len(self.mesh_class.constraints_list)
        self.assembly_class.preallocate_B_csr()

    def are_the_constraints_li(self, u=None, t=0, tol=1E-10):
        """
        Function to find out if the constraints are linearly dependent. If the
        constraints are linearly dependent then the system cannot be solved.
        """
        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)

        # Gilbert Strang singular value decomposition
        B = self.B(u, t)
        B_BT = B @ B.T
        omega_2 = eigvalsh(B_BT.A)

        if not np.any(omega_2 < tol):
            print("YES. The constraints are not linearly dependent within this",
                  "tolerance. That means the system most likely can be solved.")

        else:
            print("NO, WARNING! There are", np.sum(omega_2 < 1E-10),
                  "constraints that are linearly dependent within",
                  "this tolerance. That means the system cannot be solved",
                  "with these constraints and this tolerance.")

    def write_timestep(self, t, u):
        '''
        write the timestep to the mechanical_system class
        '''
        MechanicalSystem.write_timestep(self, t, u[:self.dirichlet_class.no_of_constrained_dofs])
        if self.number_of_constraints is not 0:
            self.lambda_output.append(u[self.dirichlet_class.no_of_constrained_dofs:])
