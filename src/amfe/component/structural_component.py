# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

from scipy.sparse import csc_matrix
import numpy as np
import logging
from .mesh_component import MeshComponent
from amfe.assembly.structural_assembly import StructuralAssembly
from amfe.component.constants import ELEPROTOTYPEHELPERLIST, SHELLELEPROTOTYPEHELPERLIST
from amfe.mesh import Mesh


class StructuralComponent(MeshComponent):
    TYPE = 'StructuralComponent'
    ELEMENTPROTOTYPES = dict(((element[0], element[1]()) for element in ELEPROTOTYPEHELPERLIST
                              if element[1] is not None))
    SHELLELEMENTPROTOTYPES = dict(((element[0], element[1]()) for element in SHELLELEPROTOTYPEHELPERLIST
                              if element[1] is not None))
    BOUNDARYELEMENTFACTORY = dict(((element[0], element[2]) for element in ELEPROTOTYPEHELPERLIST
                                   if element[2] is not None))
    VALID_GET_MAT_NAMES = ('K', 'M', 'D')

    def __init__(self, mesh=Mesh()):
        super().__init__(mesh)
        self.rayleigh_damping = None
        self._assembly = StructuralAssembly()
        self._M_constr = None
        self._D_constr = None
        self._C_csr = None
        self._M_csr = None
        self._f_glob_int = None
        self._f_glob_ext = None
        self._stresses = None
        self._strains = None

    def g_holo(self, q, t):
        """
        Return the residual of the holonomic constraint function on displacement level

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        t : float
            time

        Returns
        -------
        res : ndarray
            Residual of the holonomic constraint function

        """
        return self._constraints.g(self._mesh.nodes.reshape(-1), q, t)

    def B(self, q, t):
        r"""
        Constraint matrix B such that :math:`B(q, t) \dot{q} + b(q, t) = 0`


        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation
        t : float
            time

        Returns
        -------
        B : csr_matrix
            Constraint matrix B
        """
        return self._constraints.B(self._mesh.nodes.reshape(-1), q, t)

    def b(self, q, t):
        r"""
        Rheonomic part of :math:`b` the constraint equation :math:`B(q, t) \dot{q} + b(q, t) = 0`


        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation
        t : float
            time

        Returns
        -------
        b : ndarray
            Rheonomic part b on velocity level
        """
        return self._constraints.b(self._mesh.nodes.reshape(-1), q, t)

    def a(self, q, dq, t):
        r"""
        Inhomogeneous part :math:`a` of the constraint equation on acceleration level
        such that :math:`B(q, t) \ddot{q} + a(q, \dot{q}, t) = 0`


        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            time

        Returns
        -------
        a : ndarray
            Part a of the constraint equation above
        """
        return self._constraints.a(self._mesh.nodes.reshape(-1), q, dq, t)

    def M(self, q, dq, t):
        """
        Compute and return the unconstrained mass matrix of the structural component.

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            Time.

        Returns
        -------
        M : sp.sparse.sparse_matrix
            Mass matrix without applied constraints in sparse CSR format.

        Notes
        -----
            M is by definition independent of ddq
        """
        self._M_csr = self._assembly.assemble_m(self._mesh.nodes, self.ele_obj,
                                                self._mesh.get_iconnectivity_by_elementids(
                                                    self._ele_obj_df['fk_mesh'].values),
                                                self._mapping.get_dofs_by_ids(self._ele_obj_df['fk_mapping'].values),
                                                q, t, self._M_csr)
        return self._M_csr

    def D(self, q, dq, t):
        """
        Compute and return the unconstrained damping matrix of the mechanical system. At the moment either no damping
        (rayleigh_damping = False) or simple Rayleigh damping applied to the system linearized around zero
        displacement (rayleigh_damping = True) are possible. They are set via the functions apply_no_damping() and
        apply_rayleigh_damping(alpha, beta).

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            Time.

        Returns
        -------
        D : scipy.sparse.sparse_matrix
            Damping matrix with applied constraints in sparse CSR format.
        """
        if self.rayleigh_damping:
            self._D_constr = self.rayleigh_damping[0] * self.M(q, dq, t) + self.rayleigh_damping[1] * self.K(q, dq, t)
        else:
            self._D_constr = csc_matrix(
                (self._constraints.no_of_dofs_unconstrained, self._constraints.no_of_dofs_unconstrained))

        return self._D_constr

    def f_int(self, q, dq, t):
        """
        Compute and return the nonlinear unconstrained internal force vector of the structural component.

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            Time.

        Returns
        -------
        f_int : ndarray
            Nonlinear internal force vector after constraints have been applied
        """
        self._f_glob_int = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                           self._mesh.get_iconnectivity_by_elementids(
                                                           self._ele_obj_df['fk_mesh'].values),
                                                           self._mapping.get_dofs_by_ids(
                                                           self._ele_obj_df['fk_mapping'].values),
                                                           q, t,
                                                           self._C_csr, self._f_glob_int)[1]
        return self._f_glob_int + self.D(q, dq, t).dot(dq)

    def K(self, q, dq, t):
        """
        Compute and return the unconstrained stiffness matrix of the structural component

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation
        t : float
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSR format.
        """
        self._C_csr = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                      self._mesh.get_iconnectivity_by_elementids(
                                                          self._ele_obj_df['fk_mesh'].values),
                                                      self._mapping.get_dofs_by_ids(
                                                          self._ele_obj_df['fk_mapping'].values), q, t,
                                                      self._C_csr, self._f_glob_int)[0]
        return self._C_csr

    def K_and_f_int(self, q, dq, t):
        """
        Compute and return the unconstrained tangential stiffness matrix and internal force vector of the structural
        component.

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSR format.
        f : ndarray
            Internal nonlinear force vector after constraints have been applied
        """
        self._C_csr, self._f_glob_int = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                                        self._mesh.get_iconnectivity_by_elementids(
                                                                        self._ele_obj_df['fk_mesh'].values),
                                                                        self._mapping.get_dofs_by_ids(
                                                                        self._ele_obj_df['fk_mapping'].values),
                                                                        q, t,
                                                                        self._C_csr, self._f_glob_int)
        return self._C_csr, self._f_glob_int + self.D(q, dq, t).dot(dq)

    def f_ext(self, q, dq, t):
        """
        Compute and return external unconstrained force vector

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            time

        Returns
        -------
        f_ext : ndarray
            external force vector after contraints have been applied
        """
        neumann_elements, neumann_mesh_fk, neumann_mapping_fk = self._neumann.get_ele_obj_fk_mesh_and_fk_mapping()
        neumann_connectivities = self._mesh.get_iconnectivity_by_elementids(neumann_mesh_fk)
        neumann_dofs = self._mapping.get_dofs_by_ids(neumann_mapping_fk)
        self._f_glob_ext = self._assembly.assemble_f_ext(self._mesh.nodes, neumann_elements,
                                                         neumann_connectivities, neumann_dofs, q, t,
                                                         f_glob=self._f_glob_ext)
        return self._f_glob_ext

    def strains_and_stresses(self, q, dq, t):
        """
        Update strain- and stress-fields

        Parameters
        ----------
        q : ndarray
            Displacement field in voigt notation.
        dq : ndarray
            Velocity field in voigt notation.
        t : float
            time

        Returns
        -------
        strains : ndarray
            nodal strains for each node

        stresses : ndarray
            nodal stresses for each node
        """
        logger = logging.getLogger(__name__)
        logger.info('assembling strains and stresses...')

        K_csr, f_glob, self._stresses, self._strains = self._assembly.assemble_k_f_S_E(self._mesh.nodes, self.ele_obj,
                                                      self._mesh.get_iconnectivity_by_elementids(
                                                          self._ele_obj_df['fk_mesh'].values),
                                                      self._mapping.get_dofs_by_ids(
                                                          self._ele_obj_df['fk_mapping'].values), self._elements_on_nodes, q, t,
                                                      self._C_csr, self._f_glob_int)

        return self._strains, self._stresses
