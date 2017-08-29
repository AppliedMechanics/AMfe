"""
Discrete Empirical Interpolation Method (DEIM) hyper-reduction

"""

import time
import copy
import numpy as np
import scipy as sp

from ..mechanical_system import ReducedSystem

__all__ = ['DEIMSystem',
           'reduce_mechanical_system_deim',
          ]



def force_surrogate(F_u, no_of_elements):
    '''
    Compute a surrogate force value for the unassembled forces F_u.

    Parameters
    ----------
    F_u : ndarray, dimension (ndof_unassembled, no_of_snapshots)
        unassembled force snapshots
    no_of_elments : int

    Returns
    -------
    F_s : ndarray, dimension (no_of_elements, no_of_snapshots)
        Surrogate force snapshots

    '''
    ndim_unassembled, no_of_snapshots = F_u.shape
    ele_dofs = ndim_unassembled // no_of_elements
    F_s = np.zeros((no_of_elements, no_of_snapshots))
    for i in range(no_of_elements):
        F_squared = F_u[i*ele_dofs:(i+1)*ele_dofs,:]**2
        F_s[i,:] = np.sum(F_squared, axis=0)
    return F_s


class DEIMSystem(ReducedSystem):
    r'''
    Hyper-reduction technique inherited from ReducedSystem class.
    Currently only DEIM based on POD is implemented.

    This class can handle UDEIM, SUDEIM and their symmetric variants
    and also the other variants of the above based on choosing one dof or nodal
    dofs or the entire elements dofs (refered to as collocaiton variants).

    Attributes
    ----------
    DEIM_type : str,
        String indicating the DEIM type.
    E_tilde : ndarray, ndim (no_of_active_elements)
        Indices of all active elements
    oblique_proj : ndarray, ndim (no_of_dofs, no_of_force_modes)
        oblique projection matrix formed by V.T @ U @ inv(P.T @ U)
        This matrix is compact and carries the oblique projection of the
        nonlinear force after the collocation.
    K0_deim : ndarray, ndim (nred, nred)
        Reduced linear stiffness matrix which corrects the DEIM nonlinear
        force and tangential stiffness matrix. It is computed as the difference
        of the linear stiffness matrix minus the linear effects coming from the
        DEIM procedure. Hence, this matrix has to be added to the nakedly
        assembled DEIM tangential stiffness matrix to obtain the tangential
        stiffness matrix of DEIM sytem.
    proj_list : ndarray, ndim (no_of_active_elements, no_of_dofs,
                               no_of_dofs_per_element)
        This tensor is used as a list. It contains all oblique projectors of
        the active elements onto the kinematic subspace. The i-th element has
        the projector proj_list[i] as oblique projector to give the
        contribution of the nonlinear force.

    Note
    ----
    The key approximation in DEIM is the approximation of the full nonlinear
    force vector f_int through the unassembled force vector f_u:

    .. math::   f  \approx U (P^T U)^{-1} P^T f_u \\
                f_{red} \approx \underbrace{V^T U (P^T U)^{-1}P^T}_{X} f_u

    It is important to note, that the nonlinear force given here is a split-off
    part of the nonlinear restoring force. Here the nolinear force is the force
    without the linear part.

    '''

    def __init__(self, *args, **kwargs):

        '''
        Initializes the Hyperreduced system

        Parameters
        ----------
        None

        Returns
        -------
        None


        '''
#        ReducedSystem.__init__(self, *args, **kwargs)
        super().__init__(self, *args, **kwargs)

        # Global variables used in DEIM
        # Set of all active elements
        self.E_tilde = np.array([], dtype=int)
#        self.K0_red = None
        self.K0_deim = None
        self.oblique_proj = None
        self.proj_list = None
        self.V_unconstr = None
        self.DEIM_type = ''


    def preprocess_DEIM(self, DEIM_type='unassem-deim-dof'):
        '''
        Setting the flags necessary for the force basis and collocation
        procedure.

        Parameters
        ----------
        DEIM_type : str, optional
            string indicating the technique used for DEIM reduction

        '''
        deim_type = DEIM_type.lower()
        self.DEIM_type = deim_type
        # Prerequisites for Alg_UDEIM
        element_indices = self.assembly_class.element_indices
        dofs_per_element = element_indices[0].shape[0]
        dofs_per_node = self.mesh_class.no_of_dofs_per_node

        ele_loc_ind = np.arange(dofs_per_element)
        self.ele_colloc_dict = {'x': ele_loc_ind[0::dofs_per_node],
                                'y': ele_loc_ind[1::dofs_per_node],
                                'z': ele_loc_ind[2::dofs_per_node],
                                'ele': ele_loc_ind }

        self.surr_flag = False
        self.unassem_flag = False
        self.sym_flag = False

        if 'surr' in deim_type:
            self.surr_flag = True

        if ('unass' in deim_type) or ('udeim' in deim_type):
            self.unassem_flag = True

        if 'sym' in deim_type:
            self.sym_flag = True

        # This is a little verbose but clear; ele collocation is default.
        if 'ele' in deim_type:
            self.collocation_type = 'ele'
        elif 'dof' in deim_type:
            self.collocation_type = 'dof'
        elif 'node' in deim_type:
            self.collocation_type = 'node'
        elif 'component' in deim_type:
            self.collocation_type = 'component'
        elif 'z' in deim_type:
            self.collocation_type = 'z'
        elif 'x' in deim_type:
            self.collocation_type = 'x'
        elif 'y' in deim_type and not 'sym' in deim_type:
            self.collocation_type = 'y'
        else:
            self.collocation_type = 'dof'

        return


    def reduce_mesh(self, U_snapshots, no_of_force_modes=10,
                          DEIM_type='unassem-deim-dof'):
        '''
        Reduce the mesh using an interpolation gathered from the snapshots.

        Parameters
        ----------
        U_snapshots : ndarray
            Constrained training snapshots of the system
        no_of_force_modes : int, optional
            First 'm' number of force modes to be chosen
        DEIM_type : str, optional
            String indicating the technique used for DEIM reduction.
            The string can be composed of different keywords indicating the
            DEIM technique:

            - collocation_type: {'dof', 'ele', 'node', 'x', 'y', 'z'}
                The element is collocated only in the selected direction
                ('dof'), all dofs of the whole element are collocated
                ('ele'), all dofs of the node are chosen ('node') or a direction
                is additionally chosen ('x', 'y', 'z')
            - symmetric: {'symm'}
                If this keyword is in the DEIM_type, the symmetric DEIM
                technique is applied.
            - surrogate: {'surr'}
                If this keyword is there, a surrogate technique for avoiding a
                large SVD is used. This is the so-called surrogate DEIM.
            The string can then be composed in arbitrary order, ie.
            'dof-symm-surr' to make a surrogate symemtric DEIM with dof
            collocation.

        Returns
        -------
        None

        References
        ----------
        DEIM:
        Chaturantabut, S. and Sorensen, D.C., 2010. Nonlinear model reduction
        via discrete empirical interpolation. SIAM Journal on Scientific
        Computing, 32(5), pp.2737-2764.

        UDEIM, SUDEIM:
        Tiso, P. and Rixen, D.J., 2013. Discrete empirical interpolation method
        for finite element structural dynamics. In Topics in Nonlinear Dynamics,
        Volume 1 (pp. 203-212). Springer New York.

        symmetric DEIM:
        Chaturantabut, S., Beattie, C. and Gugercin, S., 2016. Structure-Preserving
        Model Reduction for Nonlinear Port-Hamiltonian Systems. arXiv preprint
        arXiv:1601.00527.

        symmetric UDEIM:
        Ravichandran, T.K., Investigation of accuracy, speed and stability of
        hyper-reduction techniques for nonlinear FE. Masters Thesis. TU Delft,
        Delft University of Technology, 2016.

        '''
        self.preprocess_DEIM(DEIM_type)

        print('*'*80)
        if self.unassem_flag:
            print('Run unassembled DEIM (UDEIM) using ' +
                  '{}-collocatiion'.format(self.collocation_type))
        else:
            print('Run assembled DEIM using ' +
                  '{}-collocatiion'.format(self.collocation_type))
        if self.sym_flag:
            print('Symmetric DEIM is enabled.')
        print('*'*80)

        # Initialize the needed quantities

        element_indices = self.assembly_class.element_indices
        no_of_elements = self.mesh_class.no_of_elements
        dofs_per_element = element_indices[0].shape[0]
        no_of_modes = self.V.shape[1]

        self.V_unconstr = self.dirichlet_class.unconstrain_vec(self.V)

        self.C_deim = self.assembly_class.compute_c_deim()

        # Compute the force basis
        U_snapshots_unconstr = self.unconstrain_vec(U_snapshots)
        if self.unassem_flag: # UDEIM
            U_f_u = self.force_basis(U_snapshots_unconstr,no_of_force_modes,
                                     unassembled=True)
            P_u, self.E_tilde = self.collocate_UDEIM(U_f_u)
            self.oblique_proj = self.V_unconstr.T @ self.C_deim @ U_f_u \
                                @ sp.linalg.pinv(P_u.T @ U_f_u)
            self.P = P_u
        else: # DEIM
            U_f = self.force_basis(U_snapshots_unconstr,no_of_force_modes,
                                     unassembled=False)
            P, self.E_tilde = self.collocate_DEIM(U_f)
            self.oblique_proj = self.V_unconstr.T @ U_f \
                                @ sp.linalg.pinv(P.T @ U_f)
            P_u = self.C_deim.T @ P

            self.P = P # just some postprocessing

        # this is a little hacky but should work...
        proj_full = self.oblique_proj @ P_u.T
        proj_full = proj_full.T.reshape((no_of_elements, dofs_per_element,
                                         no_of_modes))

        self.proj_list = proj_full[self.E_tilde, :, :].transpose((0,2,1))

        print('Finished (U)DEIM mesh reduction. \n' +
              '{} collocation nodes were selected for '.format(self.P.shape[1]) +
              '{} force basis vectors \n'.format(no_of_force_modes) +
              'using the {}-collocation technique.'.format(self.collocation_type))
        print('*'*80)

        return


    def force_basis(self, U_snapshots_unconstr, no_of_force_modes,
                    unassembled=True):
        '''
        Compute the unassembled or assembled force modes based on an SVD of
        the force snapshots generated by the displacement vectors.

        Parameters
        ----------
        U_snapshots : ndarray
            Training snapshots of the system (Unconstrained full solution)
        no_of_force_modes : int, optional
            First 'm' number of force modes to be chosen
        unassembled : bool, optional
            Flag setting, if unassembled (True) or assembled (False) force basis
            is computed


        Returns
        -------
        U_f : ndarray
            Force mode either unassembled or assembled depending on the option

        '''
        # get all dimensions
        no_of_assbld_dofs, no_of_snapshots = U_snapshots_unconstr.shape
        no_of_elements = self.mesh_class.no_of_elements
        no_of_unassbld_dofs = \
            len(np.concatenate(self.assembly_class.element_indices))

        t1 = time.time()

        # compute forces corresponding to displacements
        F_snapshots = np.zeros((no_of_unassbld_dofs, no_of_snapshots))
        for i, u in enumerate(U_snapshots_unconstr.T):
            F_snapshots[:,i] = self.assembly_class.f_nl_unassembled(u, t=0)

        # assemble snapshots if necessary
        if not unassembled:
            C_deim = self.assembly_class.compute_c_deim()
            F_snapshots = C_deim @ F_snapshots

        t2 = time.time()

        # perform the SVD
        if not self.surr_flag: # regular SVD
            F_mode, sigma, _ = sp.linalg.svd(F_snapshots, full_matrices=False)

        elif self.unassembled: # surrogate SVD
            F_s_snapshots = force_surrogate(F_snapshots, no_of_elements)
            F_s_mode, sigma, V_s = sp.linalg.svd(F_s_snapshots,
                                                 full_matrices=False)
            F_mode, _ = sp.linalg.qr(F_snapshots @ V_s, mode='economic')
        else:
            raise ValueError('Surrogate DEIM is not possible. ' +
                             'Try surrogate UDEIM instead.')

        U_f = F_mode[:,:no_of_force_modes] # Choose m modes

        t3 = time.time()

        print('Force basis for (U)DEIM built.\nTime for assembly: ' +
              '{:2.2f} s, Time for SVD: {:2.2f} s.'.format(t2-t1, t3-t2))

        return U_f

    def collocate_DEIM(self, U_f):
        '''
        Build the collocation matrix P for the DEIM algorithm

        Parameters:
        -----------
        U_f : ndarray
            Assembled force basis

        Returns
        -------
        E_tilde : ndarray
            vector containing the indices of the reduced element set
        P : sparse matrix
            Collocation matrix

        '''

        ndim, no_of_force_modes = U_f.shape
        dofs_per_node = self.mesh_class.no_of_dofs_per_node
        dofs_per_element = self.assembly_class.element_indices[0].shape[0]
        C_deim = self.C_deim

        # this is more or less some nonsense but the linter is quiet then
        P = None
        colloc_dofs = np.array([], dtype=int)

        # Selection loop
        for l in range(no_of_force_modes):

            # Compute the unassembled residual res by doing a least square
            # take care of l == 0:
            if l == 0:
                res = U_f[:,l]
            else:
                A = P.T @ U_f[:,:l]
                b = P.T @ U_f[:,l]

                c = np.linalg.solve(A.T @ A, A.T @ b)
                res = U_f[:,l] - U_f[:,:l] @ c

            p = np.argmax(np.absolute(res))

            # select all dofs of the selected node
            if self.collocation_type == 'node':
                node = p // dofs_per_node
                p = node * dofs_per_node + np.arange(dofs_per_node)

            colloc_dofs = np.append(colloc_dofs, p)
            colloc_dofs = np.unique(colloc_dofs)

            # Build P:
            n_coll = len(colloc_dofs)
            P = sp.sparse.csr_matrix((np.ones(n_coll, dtype=bool),
                                      (colloc_dofs, np.arange(n_coll))),
                                     shape=(ndim, n_coll), dtype=bool)

        # Handle element set
        E_tilde = np.array([], dtype=int)

        for dof in colloc_dofs:
            p_all = C_deim.indices[C_deim.indptr[dof]:C_deim.indptr[dof+1]]
            E_tilde = np.append(E_tilde, p_all // dofs_per_element)

        E_tilde = np.unique(E_tilde)

        return P, E_tilde


    def collocate_UDEIM(self, U_f_u):
        '''
        Run collocation with UDEIM algorithms
        '''
        ndim, no_of_force_modes = U_f_u.shape
        dofs_per_node = self.mesh_class.no_of_dofs_per_node
        dofs_per_element = self.assembly_class.element_indices[0].shape[0]
        nodes_per_element = dofs_per_element // dofs_per_node
        # this is more or less some nonsense but the linter is quiet then
        P = None

        E_tilde = np.array([], dtype=int)
        colloc_dofs = np.array([], dtype=int)

        for l in range(no_of_force_modes):

            # Compute the unassembled residual res by doing a least square
            # take care of l == 0:
            if l == 0:
                res = U_f_u[:,l]
            else:
                A = P.T @ U_f_u[:,:l]
                b = P.T @ U_f_u[:,l]

                c = np.linalg.solve(A.T @ A, A.T @ b)
                res = U_f_u[:,l] - U_f_u[:,:l] @ c

            p = np.argmax(np.absolute(res))
            element = p // dofs_per_element

            E_tilde = np.append(E_tilde, element)

            # define the loc_in depending on the collocation method
            if self.collocation_type == 'dof':
                # do the inverse of what is done below to keep colloc_dofs_temp
                # equal to p
                loc_ind = p - element*dofs_per_element

            elif self.collocation_type == 'node':
                loc_dof = p - element*dofs_per_element
                node_num = loc_dof // dofs_per_node
                loc_ind = node_num*dofs_per_node + np.arange(dofs_per_node)

            elif self.collocation_type == 'component':
                loc_dof = p - element*dofs_per_element
                component = loc_dof % dofs_per_node
                loc_ind = np.arange(nodes_per_element)*dofs_per_node + component

            else:
                loc_ind = self.ele_colloc_dict[self.collocation_type]

            colloc_dofs_temp = element*dofs_per_element + loc_ind
            colloc_dofs = np.unique(np.append(colloc_dofs, colloc_dofs_temp))

            n_coll = len(colloc_dofs)
            P = sp.sparse.csr_matrix((np.ones(n_coll, dtype=bool),
                                      (colloc_dofs, np.arange(n_coll))),
                                     shape=(ndim, n_coll), dtype=bool)

        # Element set has to be unique
        E_tilde = np.unique(E_tilde)

        return P, E_tilde

    def K_and_f(self, u=None, t=0):
        '''
        Compute tangential stiffness matrix and nonlinear force vector
        with the help of the computed interpolation and selective evaluation
        based on the selected DEIM elements

        Parameters
        ----------
        u : ndarray
            Initial solution or solution from previous iteration
        t : float
            Time

        Returns
        -------
        K     : ndarray
                Hyper-reduced stiffness matrix of reduced system
        f_int : ndarray
                Hyper-reduced internal force of reduced system
        '''

        if u is None:
            u = np.zeros(self.V_unconstr.shape[1])

        if self.K0_deim is None:
            K0_red, _ = ReducedSystem.K_and_f(self,u=None,t=0)
            K0_deim_diff, _ = self.assembly_class.assemble_k_and_f_DEIM(
                                    self.E_tilde, self.proj_list,
                                    self.V_unconstr, u_red=u*0, t=t,
                                    symmetric=self.sym_flag)

            self.K0_deim = K0_red - K0_deim_diff

        K_deim, f_deim = self.assembly_class.assemble_k_and_f_DEIM(
                                    self.E_tilde, self.proj_list,
                                    self.V_unconstr, u_red=u, t=t,
                                    symmetric=self.sym_flag)

        K = K_deim + self.K0_deim
        f_int = f_deim + self.K0_deim @ u
        return K, f_int


    def export_paraview(self, filename, field_list=None):
        '''
        Export the produced results to ParaView via XDMF format.

        '''
        if field_list is None:
            new_field_list = []
        else:
            new_field_list = field_list.copy()

        h5_xi_dict = {'ParaView':True,
             'AttributeType':'Scalar',
             'Center':'Cell',
             'Name':'ele_set',
             'NoOfComponents':1,
             }

        xi = np.zeros(self.mesh_class.no_of_elements)
        xi[self.E_tilde] = np.ones(len(self.E_tilde))

        new_field_list.append((xi, h5_xi_dict))
        ReducedSystem.export_paraview(self, filename, new_field_list)
        return


def reduce_mechanical_system_deim(mechanical_system, V, overwrite=False,
                                        assembly='indirect'):
    '''
    Reduce the given mechanical system with the linear basis V and the given
    weights.

    Parameters
    ----------
    mechanical_system : instance of MechanicalSystem
        Mechanical system which will be transformed to a ReducedSystem.
    V : ndarray, shape (N_constrained, n_red)
        Reduction Basis for the reduced system
    overwrite : bool, optional
        switch, if mechanical system should be overwritten (is less memory
        intensive for large systems) or not.

    Returns
    -------
    reduced_system : instance of ReducedSystem
        Reduced system with same properties of the mechanical system and
        reduction basis V

    Example
    -------

    '''

    if overwrite:
        reduced_sys = mechanical_system
    else:
        reduced_sys = copy.deepcopy(mechanical_system)

    reduced_sys.__class__ = DEIMSystem
    reduced_sys.V = V.copy()
    reduced_sys.V_unconstr = reduced_sys.dirichlet_class.unconstrain_vec(V)
    reduced_sys.u_red_output = []
    reduced_sys.M_constr = None

    reduced_sys.E_tilde = None
    reduced_sys.K0_deim = None
    reduced_sys.DEIM_type = None
    reduced_sys.assembly_type = assembly


    print('The system is hyper reduced now. It still needs to build the ' +
          'reduced mesh.')
    return reduced_sys