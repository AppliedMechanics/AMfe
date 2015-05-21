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


class Element():
    '''
    this is the baseclass for all elements. It contains the methods needed
    for the computation of the element stuff...
    '''

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, density=10):
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

    def k_and_f_int(X, u):
        '''
        Returns the tangential stiffness matrix and the internal nodal force
        of the Element.

        Parameters:
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns:
        --------
        k_int :     The tangential stiffness matrix (numpy.ndarray of
                    dimension (ndim, ndim))

        f_int :     The nodal force vector (numpy.ndarray of dimension (ndim,))

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u), self._f_int(X, u)

    def k_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters:
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns:
        --------
        k_int :     The tangential stiffness matrix (numpy.ndarray of
                    type ndim x ndim)

        '''
        self._compute_tensors(X, u)
        return self._k_int(X, u)

    def f_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters:
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns:
        --------
        f_int :     The nodal force vector (numpy.ndarray of dimension (ndim,))

        '''
        self._compute_tensors(X, u)
        return self._f_int(X, u)

    def m_int(self, X, u):
        '''
        Returns the tangential stiffness matrix of the Element.

        Parameters:
        -----------
        X :         nodal coordinates given in Voigt notation (i.e. a 1-D-Array
                    of type [x_1, y_1, z_1, x_2, y_2, z_2 etc.])

        u :         nodal displacements given in Voigt notation

        Returns:
        --------
        m_int :     The consistent mass matrix of the element
                    (numpy.ndarray of dimension (ndim,ndim))

        '''
        return self._m_int(X, u)



class ElementPlanar(Element):
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

    def __init__(self, E_modul=210E9, poisson_ratio=0.3, element_thickness=1., density=10):
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


    def _f_int(self, X, u):
        '''
        Private method for the computation of the internal nodal forces without computation of the relevant tensors
        '''
        f_int = self.B0.T.dot(self.S_voigt)*self.A0*self.t
        return f_int

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



