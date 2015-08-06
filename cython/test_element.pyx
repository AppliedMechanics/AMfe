# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:00:37 2015

@author: johannesr
"""

import numpy as np
cimport numpy as cnp
cimport cython

# decorator for the function
# @cython.boundscheck(False)

cnp.ndarray[double, ndim=1] X, cnp.ndarray[double, ndim=1] u):
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

class Element:
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

class Tri3(Element):
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
    cdef bool plane_stress = True

#    @jit
    def __init__(self, double E_modul=210E9, double poisson_ratio=0.3, double element_thickness=1., double density=1E4):
        '''
        Definition of material properties and thickness as they are 2D-Elements.
        '''
        cdef double self.poisson_ratio = poisson_ratio
        cdef double self.e_modul = E_modul
        cdef double self.lame_mu = E_modul / (2*(1+poisson_ratio))
        cdef double self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        # ATTENTION: here the switch between plane stress and plane strain makes sense.
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            cdef cnp.ndarray[double, ndim=2] self.C_SE = np.array([
                         [self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        cdef double self.t = element_thickness
        cdef double self.rho = density

#    @jit
    def _compute_tensors(self, ndarray[double, ndim=1] X, ndarray[double, ndim=1] u):
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
            double X1
            double Y1
            double X2
            double Y2
            double X3
            double Y3
        X1, Y1, X2, Y2, X3, Y3 = X
        self.u = u.reshape((-1,2))
        self.A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.B0_tilde = 1/(2*self.A0)*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]]).T
        self.H = self.u.T.dot(self.B0_tilde.T)
        self.F = self.H + self.I
        self.E = 1/2*(self.H + self.H.T + self.H.T.dot(self.H))
        ## Trace is very slow; use a direct sum instead...
        # self.S = self.lame_lambda*np.trace(self.E)*self.I + 2*self.lame_mu*self.E

        # This is sick but it improves the stuff...
        self.S_voigt = self.C_SE.dot([self.E[0,0], self.E[1,1], 2*self.E[0,1]])
        self.S[0,0] , self.S[1,1], self.S[1,0], self.S[0,1] = self.S_voigt[0], self.S_voigt[1], self.S_voigt[2], self.S_voigt[2]

        # Building B0 with the product of the deformation gradient
        self.B0 = np.zeros((3, 6))
        for i in range(3):
            self.B0[:,2*i:2*i+2] = np.array([[self.B0_tilde[0,i], 0], [0, self.B0_tilde[1,i]], [self.B0_tilde[1,i], self.B0_tilde[0,i]]]).dot(self.F.T)

#    @jit
    def _f_int(self, X, u):
        '''
        Private method for the computation of the internal nodal forces without computation of the relevant tensors
        '''
        f_int = self.B0.T.dot(self.S_voigt)*self.A0*self.t
        return f_int

#    @jit
    def _k_int(self, X, u):
        '''
        Private method for computation of internal tangential stiffness matrix without an update of the internal tensors

        '''
        # as the kronecker product is very expensive, the stuff is done explicitly
        # self.K_geo = np.kron(self.K_geo_small, self.I)
        self.K_geo_small = self.B0_tilde.T.dot(self.S.dot(self.B0_tilde))*self.A0*self.t
        self.K_geo = scatter_geometric_matrix(self.K_geo_small, 2)
        self.K_mat = self.B0.T.dot(self.C_SE.dot(self.B0))*self.A0*self.t
        return self.K_mat + self.K_geo

#    @jit
    def _m_int(self, X, u):
        '''
        Bestimmt die Massenmatrix. Erstellt die Massenmatrix durch die fest einprogrammierte Darstellung aus dem Lehrbuch.
        '''
        X1, Y1, X2, Y2, X3, Y3 = X
        self.A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.M = np.array([[2, 0, 1, 0, 1, 0],
                           [0, 2, 0, 1, 0, 1],
                           [1, 0, 2, 0, 1, 0],
                           [0, 1, 0, 2, 0, 1],
                           [1, 0, 1, 0, 2, 0],
                           [0, 1, 0, 1, 0, 2]])*self.A0/12*self.t*self.rho
        # Dauert ewig mit der kron-Funktion; Viel einfacher scheint die direkte Berechnungsmethode zu sein...
#        self.M_small = np.array([[2, 1, 1], [1, 2, 1,], [1, 1, 2]])*self.A0/12*self.t*self.rho
#        self.M = np.kron(self.M_small, self.I)
        return self.M
