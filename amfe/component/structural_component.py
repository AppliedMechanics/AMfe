# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
from scipy.sparse import csc_matrix

from .mesh_component import MeshComponent
from amfe.constraint.constraint_manager import  ConstraintManager
from amfe.assembly.structural_assembly import StructuralAssembly
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
        if mesh.dimension == 3:
            self._fields = ('ux', 'uy', 'uz')
        elif mesh.dimension == 2:
            self._fields = ('ux', 'uy')
        self._assembly = StructuralAssembly()
        self._M_constr = None
        self._D_constr = None
        self._C_csr = None
        self._M_csr = None
        self._f_glob = None

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

        u_unconstr = self._get_unconstrained_u(u)
        
        self._constraints.update_constraints(self.X, u=u_unconstr, du=u_unconstr, ddu=u_unconstr, t=t)
            

        self._M_csr = self._assembly.assemble_m(self._mesh.nodes, self.ele_obj,
                                                self._mesh.get_iconnectivity_by_elementids(self._ele_obj_df['fk_mesh'].values),
                                                self._mapping.get_dofs_by_ids(self._ele_obj_df['fk_mapping'].values), u_unconstr, t, self._M_csr)
        self._M_constr = self._constraints.constrain_matrix(self._M_csr)
        return self._M_constr

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

        if self._D_constr is None or force_update:
            if self.rayleigh_damping:
                self._D_constr = self.rayleigh_damping[0] * self.M() + self.rayleigh_damping[1] * self.K()
            else:
                self._D_constr = csc_matrix(self.M().shape)
        return self._D_constr

    def f_int(self, u=None, t=0):
        """
        Compute and return the nonlinear internal force vector of the structural component.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        f_int : ndarray
            Nonlinear internal force vector after constraints have been applied
        """

        u_unconstr = self._get_unconstrained_u(u)
        
        self._constraints.update_constraints(self.X, u=u_unconstr, du=u_unconstr, ddu=u_unconstr, t=t)

        self._f_glob = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                       self._mesh.get_iconnectivity_by_elementids(self._ele_obj_df['fk_mesh'].values),
                                                          self._mapping.get_dofs_by_ids(self._ele_obj_df['fk_mapping'].values),
                                                          u_unconstr, t,
                                                          self._C_csr, self._f_glob)[1]
        return self._constraints.constrain_vector(self._f_glob)

    def K(self, u=None, t=0):
        """
        Compute and return the stiffness matrix of the structural component

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        """

        u_unconstr = self._get_unconstrained_u(u)
        
        self._constraints.update_constraints(self.X, u=u_unconstr, du=u_unconstr, ddu=u_unconstr, t=t)

        self._C_csr = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                      self._mesh.get_iconnectivity_by_elementids(self._ele_obj_df['fk_mesh'].values),
                                                      self._mapping.get_dofs_by_ids(
                                                          self._ele_obj_df['fk_mapping'].values), u_unconstr, t,
                                                      self._C_csr, self._f_glob)[0]
        return self._constraints.constrain_matrix(self._C_csr)

    def K_and_f_int(self, u=None, t=0):
        """
        Compute and return the tangential stiffness matrix and internal force vector of the structural component.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        f : ndarray
            Internal nonlinear force vector after constraints have been applied
        """

        u_unconstr = self._get_unconstrained_u(u)
        
        self._constraints.update_constraints(self.X, u=u_unconstr, du=u_unconstr, ddu=u_unconstr, t=t)

        self._C_csr, self._f_glob = self._assembly.assemble_k_and_f(self._mesh.nodes, self.ele_obj,
                                                                    self._mesh.get_iconnectivity_by_elementids(self._ele_obj_df['fk_mesh'].values),
                                                                    self._mapping.get_dofs_by_ids(self._ele_obj_df['fk_mapping'].values), u_unconstr, t,
                                                                    self._C_csr, self._f_glob)
        return self._constraints.constrain_matrix(self._C_csr), self._constraints.constrain_vector(self._f_glob)

    def f_ext(self, u=None, du=None, t=0):
        """
        Compute and return external force vector

        Parameters
        ----------
        u : ndarray
            displacement field in voigt noation. len(u)  is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            time

        Returns
        -------
        f_ext : ndarray
            external force vector after contraints have been applied
        """

        u_unconstr = self._get_unconstrained_u(u)

        neumann_elements, neumann_mesh_fk, neumann_mapping_fk = self._neumann.get_ele_obj_fk_mesh_and_fk_mapping()
        neumann_connectivities = self._mesh.get_iconnectivity_by_elementids(neumann_mesh_fk)
        neumann_dofs = self._mapping.get_dofs_by_ids(neumann_mapping_fk)
        self._f_glob = self._assembly.assemble_f_ext(self._mesh.nodes, neumann_elements,
                                      neumann_connectivities, neumann_dofs, u_unconstr, t,
                                      f_glob=self._f_glob)
        return self._constraints.constrain_vector(self._f_glob)
    
    def _get_unconstrained_u(self, u):
        if u is None:
            u_unconstr = np.zeros(self._mapping.no_of_dofs)
        else:
            u_unconstr = self._constraints.unconstrain_vector(u)
            
        return u_unconstr
