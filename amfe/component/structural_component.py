# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
from scipy.sparse import csc_matrix

from .mesh_component import MeshComponent
from amfe.component.constants import ELEPROTOTYPEHELPERLIST
from amfe.mesh import Mesh


class StructuralComponent(MeshComponent):
        
    TYPE = 'StructuralComponent'
    ELEMENTPROTOTYPES = dict(((element[0], element[1]()) for element in ELEPROTOTYPEHELPERLIST
                              if element[1] is not None))
    BOUNDARYELEMENTFACTORY = dict(((element[0], element[2]) for element in ELEPROTOTYPEHELPERLIST
                                   if element[2] is not None))
    VALID_GET_MAT_NAMES = ('K', 'M', 'D')

    def __init__(self, mesh=Mesh()):
        super().__init__(mesh)
        self.rayleigh_damping = None
        self.mesh = mesh

    def M(self, u=None, t=0, force_update=False):
        """
        Compute and return the mass matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Array of the displacement.
        t : float
            Time.
        force_update : bool
            Flag to force update of M otherwise already calculated M is returned. Default is False.

        Returns
        -------
        M : sp.sparse.sparse_matrix
            Mass matrix with applied constraints in sparse CSC format.
        """

        if self.M_constr is None or force_update:
            if u is not None:
                u_unconstr = self._constraints.unconstrain_vec(u)
            else:
                u_unconstr = None

            M_unconstr = self._assembly.assemble_m(u_unconstr, t)
            self.M_constr = self._constraints.constrain_m(M_unconstr)
        return self.M_constr

    def D(self, u=None, t=0, force_update=False):
        """
        Compute and return the damping matrix of the mechanical system. At the moment either no damping
        (rayleigh_damping = False) or simple Rayleigh damping applied to the system linearized around zero
        displacement (rayleigh_damping = True) are possible. They are set via the functions apply_no_damping() and
        apply_rayleigh_damping(alpha, beta).

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.
        force_update : bool
            Flag to force update of D otherwise already calculated D is returned. Default is False.

        Returns
        -------
        D : scipy.sparse.sparse_matrix
            Damping matrix with applied constraints in sparse CSC format.
        """

        if self.D_constr is None or force_update:
            if self.rayleigh_damping:
                self.D_constr = self.rayleigh_damping[0] * self.M() + self.rayleigh_damping[1] * self.K()
            else:
                self.D_constr = csc_matrix(self.M().shape)
        return self.D_constr

    def K(self, u=None, t=0):
        """
        Compute and return the stiffness matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        """

        if u is None:
            u = np.zeros(self._constraints.no_of_constrained_dofs)

        K_unconstr = self._assembly.assemble_k_and_f(self._constraints.unconstrain_u(u), t)[0]
        return self._constraints.constrain_matrix(K_unconstr)
