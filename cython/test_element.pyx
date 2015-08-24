# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:37 2015

@author: johannesr
"""

import numpy as np
cimport numpy as cnp
cimport cython

from cython.view cimport array as cvarray
from cpython cimport bool

cpdef scatter_geometric_matrix(cnp.ndarray[double, ndim=2] Mat, int ndim):

    '''
    Scatter the symmetric geometric stiffness matrix to all dofs.

    What is basically done is to perform the kron(Mat, eye(ndof))

    Parameters
    ----------
    Mat : ndarray
        Matrix that should be scattered
    ndim : int
        number of dimensions of finite element. If it's a 2D element, ndim=2,
        if 3D, ndim=3

    Returns
    -------
    Mat_scattered : ndarray
        scattered matrix

    '''
    cdef int dof_small_row = Mat.shape[0]
    cdef int dof_small_col = Mat.shape[1]
    cdef cnp.ndarray[double, ndim=2] Mat_scattered = np.zeros((dof_small_row*ndim, dof_small_col*ndim))
    cdef:
        int i
        int j
        int k
    for i in range(dof_small_row):
        for j in range(dof_small_col):
            for k in range(ndim):
                Mat_scattered[ndim*i+k,ndim*j+k] = Mat[i, j]
    return Mat_scattered

# giving how many voigt-dofs do you have given a dof.
voigt_dof_dict = {1: 1, 2: 3, 3: 6}

cpdef compute_B_matrix(cnp.ndarray[double, ndim=2] F, cnp.ndarray[double, ndim=2] B_tilde):
    '''
    Compute the B-matrix used in Total Lagrangian Finite Elements.

    Parameters
    ----------
    F : ndarray
        deformation gradient
    B_tilde : ndarray
        Matrix of the spatial derivative of the shape functions

    Returns
    -------
    B : ndarray
        B matrix such that {dE} = B @ {u^e}

    Notes
    -----
    When the Voigt notation is used in this Reference, the variables are denoted with curly brackets.
    '''
    cdef:
        int no_of_nodes, no_of_dims, i
        double F11, F12, F21, F22, F13, F31, F23, F32, F33
        cnp.ndarray[double, ndim=2] b, B

    no_of_nodes = B_tilde.shape[1]
    no_of_dims = B_tilde.shape[0] # spatial dofs per node, i.e. 2 for 2D or 3 for 3D
    b = B_tilde
    B = np.zeros((voigt_dof_dict[no_of_dims], no_of_nodes*no_of_dims))
    F11 = F[0,0]
    F12 = F[0,1]
    F21 = F[1,0]
    F22 = F[1,1]
    if no_of_dims == 3:

        F13 = F[0,2]
        F31 = F[2,0]
        F23 = F[1,2]
        F32 = F[2,1]
        F33 = F[2,2]
    for i in range(no_of_nodes):
        if no_of_dims == 2:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
            [F11*b[0,i], F21*b[0,i]],
            [F12*b[1,i], F22*b[1,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]]]
        else:
            B[:, i*no_of_dims : (i+1)*no_of_dims] = [
            [F11*b[0,i], F21*b[0,i], F31*b[0,i]],
            [F12*b[1,i], F22*b[1,i], F32*b[1,i]],
            [F13*b[2,i], F23*b[2,i], F33*b[2,i]],
            [F11*b[1,i] + F12*b[0,i], F21*b[1,i] + F22*b[0,i]], F31*b[1,i]+F32*b[0,i],
            [F12*b[2,i] + F13*b[1,i], F22*b[2,i] + F23*b[1,i]], F32*b[2,i]+F33*b[1,i],
            [F13*b[0,i] + F11*b[2,i], F23*b[0,i] + F21*b[2,i]], F33*b[0,i]+F31*b[2,i]]
    return B

# this dict contains a list with the voigtize index triples. Giving the indices
# i, j, k, i is the index of the voigt vector whereas j, k are the indices of the full matrix.
voigtize_index_pairs = {
2 : ((0, 0,0), (1, 1,1), (2, 0,1)),
3 : ((0, 0,0), (1, 1,1), (2, 2,2), (3, 1, 2), (4, 0, 2), (5, 0, 1)),
}

cpdef voigtize(cnp.ndarray[double, ndim=2] Mat, bool kinematic=False):
    '''
    Voigtize a symmetric matrix.

    Parameters
    ----------


    Returns
    -------
    '''
    cdef int ndim = Mat.shape[0]
    cdef cnp.ndarray[double, ndim=1] Mat_v = np.zeros(voigt_dof_dict[ndim])
    cdef:
        int i
        int j
        int k
    for i, j, k in voigtize_index_pairs[ndim]:
        Mat_v[i] = Mat[j, k]
    if kinematic:
        for i in range(voigt_dof_dict[ndim] - ndim):
            i += 1 # correct indexing as the index is negative; the last element is -1, the one before -2 etc.
            Mat_v[-i] *= 2.
    return Mat_v


cpdef unvoigtize(cnp.ndarray[double, ndim=2] Mat_v, bool kinematic=False):
    '''
    Make a proper symmetric array from the voigt array given.
    '''
    if Mat_v.shape[0] == 3:
        ndim = 2
    elif Mat_v.shape[0] == 6:
        ndim = 3
    else:
        raise Exception('The given matrix is no voigt array!')

    cdef:
        int i
        int j
        int k

    cdef cnp.ndarray[double, ndim=2] Mat = np.zeros((ndim, ndim))

    if kinematic:
        for i in range(voigt_dof_dict[ndim] - ndim):
            i += 1
            Mat_v[-i] /= 2

    for i, j, k in voigtize_index_pairs[ndim]:
        Mat[j, k] = Mat_v[i]
        # consider the antisymmetric part:
        if j != k:
            Mat[k, j] = Mat_v[i]

    return Mat

#cpdef mydot(A, B):
#    cdef int ndim_A = len(A.shape)
#    cdef int ndim_B = len(B.shape)
#    cdef double[][] C
#    # check, if dimensions are okay:
#    if not A.shape[-1] == B.shape[0]:
#        raise Exception('The dimensions of the two matrices do not match!')
#
#    pass


cdef class Element:
    '''
    this is the baseclass for all elements. It contains the methods needed
    for the computation of the element stuff...
    '''

    def __init__(self, float E_modul=210E9, float poisson_ratio=0.3, float density=1E4):
        pass

    def _compute_tensors(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Virtual function for the element specific implementation of a tensor
        computation routine which will be called before _k_int and _f_int
        will be called. For many computations the tensors need to be computed
        the same way.
        '''
        pass

    def _k_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        pass

    def _f_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        pass

    def _m_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Virtual function for the element specific implementation of the mass
        matrix;
        '''
        print('The function is not implemented yet...')
        pass

    def k_and_f_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Returns the tangential stiffness matrix and the internal nodal force
        of the Element.

        Parameters
        -----------
        X : ndarray
            nodal coordinates given in Voigt notation (i.e. a 1-D-Array
            of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])
        u : ndarray
            nodal displacements given in Voigt notation

        Returns
        --------
        k_int : ndarray
            The tangential stiffness matrix (ndarray of dimension (ndim, ndim))
        f_int : ndarray
            The nodal force vector (ndarray of dimension (ndim,))

        Examples
        ---------
        TODO

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u), self._f_int(X, u)

    def k_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        k_int :     The tangential stiffness matrix (numpy.ndarray of
                    type ndim x ndim)

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u)

    def f_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        f_int :     The nodal force vector (numpy.ndarray of dimension (ndim,))

        '''
        self._compute_tensors(X, u)
        return self._f_int(X, u)

    def m_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        m_int :     The consistent mass matrix of the element
                    (numpy.ndarray of dimension (ndim,ndim))

        '''
        return self._m_int(X, u)


    def k_and_m_int(self,cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Returns the stiffness and mass matrix of the Element.

        Parameters
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns
        --------
        m_int :     The consistent mass matrix of the element
                    (numpy.ndarray of dimension (ndim,ndim))

        '''
        return self._k_and_m_int(X, u)

cdef class Tri3(Element):
    '''
    Element class for a plane triangle element in Total Lagrangian formulation.
    The displacements are given in x- and y-coordinates;

    Element-properties:
    -------------------
    The Element assumes constant strain and stress over the whole element.
    Thus the approximation quality is very moderate.


    References:
    -----------
    Basis for this implementation is the Monograph of Ted Belytschko:
    Nonlinear Finite Elements for Continua and Structures.
    pp. 201 and 207.

    '''
    plane_stress = True
    cdef:
        double poisson_ratio
        double e_modul
        double lame_mu
        double lame_lambda
        double t
        double rho
        double A0

        object C_SE


#    @jit
    def __init__(self, double E_modul=210E9, double poisson_ratio=0.3, double element_thickness=1., double density=1E4):
        '''
        Definition of material properties and thickness as they are 2D-Elements.
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.t = element_thickness
        self.rho = density

        # ATTENTION: here the switch between plane stress and plane strain makes sense.
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([
                         [self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])

    def _compute_tensors(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        pass

    def _compute_everything(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Compute the tensors B0_tilde, B0, F, E and S at the Gauss Points.

        The tensors are:
            B0_tilde:   Die Ableitung der Ansatzfunktionen nach den x- und y-Koordinaten (2x3-Matrix)
                        In den Zeilein stehen die Koordinatenrichtungen, in den Spalten die Ansatzfunktionen
            B0:         The mapping matrix of delta E = B0 * u^e
            F:          Deformation gradient (2x2-Matrix)
            E:          Der Green-Lagrange strain tensor (2x2-Matrix)
            S:          2. Piola-Kirchhoff stress tensor, using Kirchhoff material (2x2-Matrix)

        The thickness information is used later for the internal forces, the mass and the stiffness matrix.
        '''
        cdef:
            double X1, Y1, X2, Y2, X3, Y3
            double A0

            # the matrices involved using memory view types
#            double u_mat[3][2]
#            double B0_tilde[2][3]
#            double H[2][2]
#            double F[2][2]
#            double E[2][2]
#            double E_v[3]
#            double S[2][2]
#            double S_v[3]
#            double B0[3][6]

            cnp.ndarray[double, ndim=2] B0_tilde
            cnp.ndarray[double, ndim=2] H
            cnp.ndarray[double, ndim=2] F
            cnp.ndarray[double, ndim=2] E
            cnp.ndarray[double, ndim=2] S
            cnp.ndarray[double, ndim=2] B0
            cnp.ndarray[double, ndim=2] K_mat
            cnp.ndarray[double, ndim=2] K_geo_small
            cnp.ndarray[double, ndim=2] K

            cnp.ndarray[double, ndim=1] f_int
            cnp.ndarray[double, ndim=1] E_v
            cnp.ndarray[double, ndim=1] S_v


        X1, Y1, X2, Y2, X3, Y3 = X
        A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        u_mat = u.reshape(-1, 2)
#        u_mat[0][:] = [u[0], u[1]]
#        u_mat[1][:] = [u[2], u[3]]
#        u_mat[2][:] = [u[4], u[5]]
        B0_tilde = np.array([[Y2-Y3, Y3-Y1, Y1-Y2], [X3-X2, X1-X3, X2-X1]])
#        B0_tilde[0][:] = [Y2-Y3, Y3-Y1, Y1-Y2]
#        B0_tilde[1][:] = [X3-X2, X1-X3, X2-X1]
        B0_tilde /= 2*A0
        H = np.dot(u_mat.T, B0_tilde.T)
        E = 0.5*(H + H.T + np.dot(H.T, H))
        E_v = np.array([E[0,0], E[1,1], 2*E[0,1]])
#        E_v = voigtize(E, kinematic=True)
        F = H.copy()
        F[0,0] += 1.
        F[1,1] += 1.
        S_v = np.dot(self.C_SE, E_v)
#        S = unvoigtize(S_v)
        S = np.array([[S_v[0], S_v[2]], [S_v[2], S_v[1]]])
        B0 = compute_B_matrix(F, B0_tilde)
        K_mat = np.dot(B0.T, np.dot(self.C_SE, B0)) * A0 * self.t
        K_geo_small = np.dot(B0_tilde.T, np.dot(S, B0_tilde)) * A0 * self.t
        K = K_mat + scatter_geometric_matrix(K_geo_small, ndim=2)
        f_int = np.dot(B0.T, S_v) * A0 * self.t

#        cdef cnp.ndarray[double, ndim=2] B0_tilde = 1/(2*A0)*[, ]
#        cdef cnp.ndarray[double, ndim=2] H        = u_mat.T.dot(B0_tilde.T)
#        cdef cnp.ndarray[double, ndim=2] F        = H + [[1., 0], [0, 1.]]
#        cdef cnp.ndarray[double, ndim=2] E        = 0.5*(H + H.T + H.T.dot(H))
#        cdef cnp.ndarray[double, ndim=1] E_v      = [E[0,0], E[1,1], 2*E[0,1]]
#        cdef cnp.ndarray[double, ndim=1] S_v      = self.C_SE.dot(E_v)
#        cdef cnp.ndarray[double, ndim=2] S        = [[S_v[0], S_[2]], [S_v[2], S_v[1]]]
#        cdef cnp.ndarray[double, ndim=2] B0       = compute_B_matrix(F, B0_tilde)
#        cdef cnp.ndarray[double, ndim=1] f_int    = B0.T.dot(S_v)*A0*self.t
#        cdef cnp.ndarray[double, ndim=2] K_mat    = B0.T.dot(self.C_SE.dot(B0))*A0*self.t
#        cdef cnp.ndarray[double, ndim=2] K_geo_small = B0_tilde.T.dot(S.dot(B0_tilde))*A0*self.t
#        cdef cnp.ndarray[double, ndim=2] K        = K_mat + scatter_geometric_matrix(K_geo_small, ndim=2)
        return K, f_int

    def _f_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Private method for the computation of the internal nodal forces without computation of the relevant tensors
        '''
        K, f = self._compute_everything(X, u)
        return f

    def _k_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''
        Private method for computation of internal tangential stiffness matrix without an update of the internal tensors

        '''
        K, f = self._compute_everything(X, u)
        return K

    def k_and_f_int(self, cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
        '''

        '''
        return self._compute_everything(X, u)


    def _m_int(self, X, u):
        '''
        Bestimmt die Massenmatrix. Erstellt die Massenmatrix durch die fest einprogrammierte Darstellung aus dem Lehrbuch.
        '''
        X1, Y1, X2, Y2, X3, Y3 = X
        A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.M = np.array([[2, 0, 1, 0, 1, 0],
                           [0, 2, 0, 1, 0, 1],
                           [1, 0, 2, 0, 1, 0],
                           [0, 1, 0, 2, 0, 1],
                           [1, 0, 1, 0, 2, 0],
                           [0, 1, 0, 1, 0, 2]])*A0/12*self.t*self.rho
        return self.M
