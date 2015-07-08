# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:13:52 2015

Element-Modul, in der die Elementformulierungen enthalten sind.


Beobachtungen aus dem Profiler:
Die Meiste Zeit im Assembly-Prozess wird mit der kron-Funktion und der
trace-funktion verbraucht; Es lohnt sich sicherlich, diese Funktionen
geschickter zu implementieren!

@author: johannesr
"""


import numpy as np
#from numpy.linalg import inv
#from numba import jit, autojit

#@autojit
class Element():
    '''
    this is the baseclass for all elements. It contains the methods needed
    for the computation of the element stuff...
    '''

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, density=1E4):
        pass

    def _compute_tensors(self, X, u):
        '''
        Virtual function for the element specific implementation of a tensor
        computation routine which will be called before _k_int and _f_int
        will be called. For many computations the tensors need to be computed
        the same way.
        '''
        pass

    def _k_int(self, X, u):
        pass

    def _f_int(self, X, u):
        pass

    def _m_int(self, X, u):
        '''
        Virtual function for the element specific implementation of the mass
        matrix;
        '''
        print('The function is not implemented yet...')
        pass

    def k_and_f_int(self, X, u):
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

    def k_int(self, X, u):
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

    def f_int(self, X, u):
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

    def m_int(self, X, u):
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
        
        
    def k_and_m_int(self, X, u):
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

#@autojit
class Tri3(Element):
    '''
    Element class for a plane triangle element in Total Lagrangian formulation.
    The displacements are given in x- and y-coordinates;

    Element-properties:
    -----------
    The Element assumes constant strain and stress over the whole element.
    Thus the approximation quality is very moderate.


    References:
    ------------
    Basis for this implementation is the Monograph of Ted Belytschko:
    Nonlinear Finite Elements for Continua and Structures.
    pp. 201 and 207.

    '''
    plane_stress = True

#    @jit
    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        # Achtung: hier gibt's ebene Dehnung
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        self.t = element_thickness
        self.rho = density
        self.I = np.eye(2)
        self.S = np.zeros((2,2))
        self.K_geo = np.zeros((6,6))
        pass

#    @jit
    def _compute_tensors(self, X, u):
        '''
        Bestimmung der tensoriellen Größen des Elements für eine Total Lagrange Betrachtungsweise. Die Tensoren werden als Objektvariablen abgespeichert, daher hat diese Funktion keine Rückgabewerte.

        Die bestimmten Größen sind:
            B0_tilde:   Die Ableitung der Ansatzfunktionen nach den x- und y-Koordinaten (2x3-Matrix)
                        In den Zeilein stehen die Koordinatenrichtungen, in den Spalten die Ansatzfunktionen
            F:          Der Deformationsgradient (2x2-Matrix)
            E:          Der Green-Lagrange Dehnungstensor (2x2-Matrix)
            S:          Der 2. Piola-Kirchhoff-Spannungstensor, berechnet auf Basis des Kirchhoff'schen Materialmodells (2x2-Matrix)

        In allen Tensorberechnungen werden nur ebene Größen betrachtet.
        Die Dickeninformation (self.t) kommt dann erst später bei der Bestimmung der internen Kräfte f_int bzw. der Massen- und Steifigkeitsmatrizen zustande.
        '''
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
        self.K_geo = np.array([[self.K_geo_small[0,0], 0, self.K_geo_small[0,1], 0, self.K_geo_small[0,2], 0],
                           [0, self.K_geo_small[0,0], 0, self.K_geo_small[0,1], 0, self.K_geo_small[0,2]],
                           [self.K_geo_small[1,0], 0, self.K_geo_small[1,1], 0, self.K_geo_small[1,2], 0],
                           [0, self.K_geo_small[1,0], 0, self.K_geo_small[1,1], 0, self.K_geo_small[1,2]],
                           [self.K_geo_small[2,0], 0, self.K_geo_small[2,1], 0, self.K_geo_small[2,2], 0],
                           [0, self.K_geo_small[2,0], 0, self.K_geo_small[2,1], 0, self.K_geo_small[2,2]]])
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

#@autojit
class Tri6(Element):
    '''
    6 node second order triangle
    Triangle Element with 6 dofs; 3 dofs at the corner, 3 dofs in the intermediate point of every face.
    '''
    plane_stress = True

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=1E4):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        # Achtung: hier gibt's ebene Dehnung
        if self.plane_stress:
            self.C_SE = E_modul/(1 - poisson_ratio**2)*np.array([[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio) / 2]])
        else: # hier gibt's ebene Dehnung
            self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        self.t = element_thickness
        self.rho = density
        self.I = np.eye(2)
        self.B0_tilde = [[],[],[]]
        self.H = [[],[],[]]
        self.F = [[],[],[]]
        self.E = [[],[],[]]
        self.S_voigt = [[],[],[]]
        self.S = [np.zeros((2,2)),np.zeros((2,2)),np.zeros((2,2))]
        self.B0 = [[],[],[]]

    def _B0_tilde_func(self, X_vec, X, Y):
        '''computes the B0_tilde matrix for a given X and Y'''
        # maybe this is in the scope here
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X_vec
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        # linear coordinates
        L1 = (X*(Y2 - Y3) + X2*Y3 - X3*Y2 + Y*(-X2 + X3))/det
        L2 = (X*(-Y1 + Y3) - X1*Y3 + X3*Y1 + Y*(X1 - X3))/det
        L3 = (X*(Y1 - Y2) + X1*Y2 - X2*Y1 + Y*(-X1 + X2))/det
        # diff of coordinates with respect to X and Y
        L1_X = (Y2-Y3)/det
        L1_Y = (X3-X2)/det
        L2_X = (-Y1+Y3)/det
        L2_Y = (X1-X3)/det
        L3_X = (Y1-Y2)/det
        L3_Y = (-X1+X2)/det
        # Full diff of coordinates
        N1_X = (4*L1 - 1)*L1_X
        N1_Y = (4*L1 - 1)*L1_Y
        N2_X = (4*L2 - 1)*L2_X
        N2_Y = (4*L2 - 1)*L2_Y
        N3_X = (4*L3 - 1)*L3_X
        N3_Y = (4*L3 - 1)*L3_Y

        N4_X = 4*(L1*L2_X + L1_X*L2)
        N4_Y = 4*(L1*L2_Y + L1_Y*L2)
        N5_X = 4*(L2*L3_X + L2_X*L3)
        N5_Y = 4*(L2*L3_Y + L2_Y*L3)
        N6_X = 4*(L1*L3_X + L1_X*L3)
        N6_Y = 4*(L1*L3_Y + L1_Y*L3)

        B0_tilde = np.array([[N1_X, N2_X, N3_X, N4_X, N5_X, N6_X], [N1_Y, N2_Y, N3_Y, N4_Y, N5_Y, N6_Y]])
        return B0_tilde

    def _compute_tensors(self, X, u):
        '''
        Tensor computation the same way as in the Tri3 element
        '''
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        self.u = u.reshape((-1,2))
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        self.A0 = det/2
        quadrature_points = np.array([[X4, Y4], [X5, Y5], [X6, Y6]])
        for i, quadrature_coord in enumerate(quadrature_points):
            self.B0_tilde[i] = self._B0_tilde_func(X, *quadrature_coord)
            self.H[i] = self.u.T.dot(self.B0_tilde[i].T)
            self.F[i] = self.H[i] + self.I
            self.E[i] = 1/2*(self.H[i] + self.H[i].T + self.H[i].T.dot(self.H[i]))
            self.S_voigt[i] = self.C_SE.dot([self.E[i][0,0], self.E[i][1,1], 2*self.E[i][0,1]])
            self.S[i][0,0] , self.S[i][1,1], self.S[i][1,0], self.S[i][0,1] = \
                self.S_voigt[i][0], self.S_voigt[i][1], self.S_voigt[i][2], self.S_voigt[i][2]
            self.B0[i] = np.zeros((3, 12))
            for j in range(6):
                self.B0[i][:,2*j:2*j+2] = np.array([[self.B0_tilde[i][0,j], 0],
                        [0, self.B0_tilde[i][1,j]], [self.B0_tilde[i][1,j],
                         self.B0_tilde[i][0,j]]]).dot(self.F[i].T)

    def _f_int(self, X, u):
        f_int = np.zeros(12)
        for i in range(3):
            f_int += self.B0[i].T.dot(self.S_voigt[i])*self.A0*self.t*1/3
        return f_int

    def _k_int(self, X, u):
        self.K_geo = np.zeros((12, 12))
        self.K_mat = np.zeros((12, 12))
        for i in range(3):
            self.K_geo_small = self.B0_tilde[i].T.dot(self.S[i].dot(self.B0_tilde[i]))*self.A0*self.t*1/3
            k = self.K_geo_small
            self.K_geo += np.array([
                [k[0,0], 0, k[0,1], 0, k[0,2], 0, k[0,3], 0, k[0,4], 0, k[0,5], 0],
                [0, k[0,0], 0, k[0,1], 0, k[0,2], 0, k[0,3], 0, k[0,4], 0, k[0,5]],
                [k[1,0], 0, k[1,1], 0, k[1,2], 0, k[1,3], 0, k[1,4], 0, k[1,5], 0],
                [0, k[1,0], 0, k[1,1], 0, k[1,2], 0, k[1,3], 0, k[1,4], 0, k[1,5]],
                [k[2,0], 0, k[2,1], 0, k[2,2], 0, k[2,3], 0, k[2,4], 0, k[2,5], 0],
                [0, k[2,0], 0, k[2,1], 0, k[2,2], 0, k[2,3], 0, k[2,4], 0, k[2,5]],
                [k[3,0], 0, k[3,1], 0, k[3,2], 0, k[3,3], 0, k[3,4], 0, k[3,5], 0],
                [0, k[3,0], 0, k[3,1], 0, k[3,2], 0, k[3,3], 0, k[3,4], 0, k[3,5]],
                [k[4,0], 0, k[4,1], 0, k[4,2], 0, k[4,3], 0, k[4,4], 0, k[4,5], 0],
                [0, k[4,0], 0, k[4,1], 0, k[4,2], 0, k[4,3], 0, k[4,4], 0, k[4,5]],
                [k[5,0], 0, k[5,1], 0, k[5,2], 0, k[5,3], 0, k[5,4], 0, k[5,5], 0],
                [0, k[5,0], 0, k[5,1], 0, k[5,2], 0, k[5,3], 0, k[5,4], 0, k[5,5]]])

            self.K_mat += self.B0[i].T.dot(self.C_SE.dot(self.B0[i]))*self.A0*self.t*1/3
        return self.K_mat + self.K_geo

    def _m_int(self, X, u):
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X6, Y6 = X
        det = X1*Y2 - X1*Y3 - X2*Y1 + X2*Y3 + X3*Y1 - X3*Y2
        self.A0 = det/2
        self.M = self.A0 / 180 * self.t * self.rho * np.array([
        [  6.,   0.,  -1.,  -0.,  -1.,  -0.,   0.,   0.,  -4.,  -0.,   0.,  0.],
        [  0.,   6.,  -0.,  -1.,  -0.,  -1.,   0.,   0.,  -0.,  -4.,   0.,  0.],
        [ -1.,  -0.,   6.,   0.,  -1.,  -0.,   0.,   0.,   0.,   0.,  -4., -0.],
        [ -0.,  -1.,   0.,   6.,  -0.,  -1.,   0.,   0.,   0.,   0.,  -0., -4.],
        [ -1.,  -0.,  -1.,  -0.,   6.,   0.,  -4.,  -0.,   0.,   0.,   0.,  0.],
        [ -0.,  -1.,  -0.,  -1.,   0.,   6.,  -0.,  -4.,   0.,   0.,   0.,  0.],
        [  0.,   0.,   0.,   0.,  -4.,  -0.,  32.,   0.,  16.,   0.,  16.,  0.],
        [  0.,   0.,   0.,   0.,  -0.,  -4.,   0.,  32.,   0.,  16.,   0., 16.],
        [ -4.,  -0.,   0.,   0.,   0.,   0.,  16.,   0.,  32.,   0.,  16.,  0.],
        [ -0.,  -4.,   0.,   0.,   0.,   0.,   0.,  16.,   0.,  32.,   0., 16.],
        [  0.,   0.,  -4.,  -0.,   0.,   0.,  16.,   0.,  16.,   0.,  32.,  0.],
        [  0.,   0.,  -0.,  -4.,   0.,   0.,   0.,  16.,   0.,  16.,   0., 32.]])
        return self.M





#@autojit
class Quad4(Element):
    
    '''
    Element Klasse fuer ebenes, viereckiges Element (Quad4)
    Verschiebungen in x- und y-Richtungen.
    '''    
    plane_stress = True
 
    def __init__(self, E_modul=1.0, poisson_ratio=0., element_thickness=1.0, 
                 density=1.0, plane_stress = True):
        '''
        Definition der Materialgrößen und Dicke, da es sich um 2D-Elemente handelt
        '''
        self.poisson_ratio = poisson_ratio
        self.e_modul = E_modul
        self.t = element_thickness
        self.rho = density     
        self.plane_stress = plane_stress
        
        # Ebene Spannung
        if self.plane_stress:
            self.C = E_modul/(1-poisson_ratio**2)*np.array(
                             [[1, poisson_ratio, 0],
                              [poisson_ratio, 1, 0],
                              [0, 0, (1-poisson_ratio)/2]])
        else: # Ebene Dehnung
            print('Ebene Dehnung noch nicht implementiert')

    def _compute_tensors(self, X, u):
        pass
              
    def _k_and_m_int(self, X, u):
        
        def gauss_quadrature(option):
            '''    
            Gauss quadrature for Q4 elements
            option 'complete' (2x2)
            option 'reduced'  (1x1)
            locations: Gauss point locations
            weights: Gauss point weights
            '''      
            def complete():
                locations = np.array(
                    [[-0.577350269189626, -0.577350269189626],
                    [0.577350269189626, -0.577350269189626],
                    [0.577350269189626,  0.577350269189626],
                    [-0.577350269189626,  0.577350269189626]])
                weights = np.array([1,1,1,1]) 
                return locations, weights
            def reduced():
                locations = np.array([0, 0])
                weights = np.array(4)
                return locations, weights
            integration = {'reduced': reduced,
                           'complete': complete}
            locations, weights = integration[option]()               
            return weights, locations
        
        def f_shape_Q4(xi, eta):
            '''
            shape function and derivatives for Q4 elements
            shape : Shape functions
            d_shape: derivatives w.r.t. xi and eta 
            xi, eta: natural coordinates (-1 ... +1)
            '''            
            shape = 1/4*np.array([(1-xi)*(1-eta),     # N1         
                                  (1+xi)*(1-eta),     # N2
                                  (1+xi)*(1+eta),     # N3
                                  (1-xi)*(1+eta)])    # N4
            d_shape=1/4*np.array([[-(1-eta), -(1-xi)],      # dN1/dxi, dN1/deta
                                  [1-eta, -(1+xi)],         # dN2/dxi, dN2/deta
                                  [1+eta, 1+xi],            # dN3/dxi, dN3/deta
                                  [-(1+eta), 1-xi]])        # dN4/dxi, dN4/deta                                 
            return shape, d_shape
        
        def jacobi(X,d_shape):
            '''
            jac: Jacobian matrix
            invjac: inverse of Jacobian Matrix
            d_shape_XY: derivatives w.r.t. x and y
            d_shape: derivatives w.r.t. xi and eta
            X: nodal coordinates at element level
            '''
            jac = X.T.dot(d_shape)
            invjac = np.linalg.inv(jac)
            d_shape_XY = d_shape.dot(invjac)            
            return jac, d_shape_XY
        
        
        self.k_el = np.zeros((8, 8))
        self.m_el = np.zeros((8, 8))
        gauss_weights, gauss_loc = gauss_quadrature('complete')
        no_gp = len(gauss_weights)
        
        # Loop over Gauss points
        for i_gp in range(no_gp):
            # Get Gauss locations            
            xi, eta = gauss_loc[i_gp,:]
            # Get shape functions and derivatives with respect to xi, eta
            shape, d_shape = f_shape_Q4(xi,eta)
            # Get Jacobi and derivatives with respect to x,y
            jac, d_shape_XY = jacobi(X.reshape(4,2),d_shape)
            
            # Build B-matrix            
            B = np.zeros((3, 8))
            B[0, [0, 2, 4, 6]] = d_shape_XY[:,0]
            B[1, [1, 3, 5, 7]] = d_shape_XY[:,1]
            B[2, [0, 2, 4, 6]] = d_shape_XY[:,1]
            B[2, [1, 3, 5, 7]] = d_shape_XY[:,0]
        
            # Build N-matrix  
            N = np.zeros((2, 8))
            N[0, [0, 2, 4, 6]]  = shape
            N[1, [1, 3, 5, 7]]  = shape   

            # Add stiffness part from Gauss point                
            self.k_el = self.k_el + (self.t*B.T.dot(self.C.dot(B))*
                                    gauss_weights[i_gp]*np.linalg.det(jac))
            self.m_el = self.m_el + (self.t*self.rho*N.T.dot(N)*
                                    gauss_weights[i_gp]*np.linalg.det(jac))

        # Make symmetric (because of round-off errors)      
        self.k_el = 1/2*(self.k_el+self.k_el.T)
        self.m_el = 1/2*(self.m_el+self.m_el.T)       
        return self.k_el, self.m_el

    def _k_int(self, X, u):
        k_el, m_el = self._k_and_m_int(X, u)     
        return k_el

    def _m_int(self, X, u):
        k_el, m_el = self._k_and_m_int(X, u)     
        return m_el


        
    def _f_int(self, X, u):
        print('The function is not implemented yet...')        
        pass

        
        
        
        
        
        
        
