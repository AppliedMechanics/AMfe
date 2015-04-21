# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:13:52 2015

Aufruf der IPython-Konsole über
    >>> ipython console

@author: johannesr
"""


import numpy as np


class ElementPlanar():
    '''
    Das ist die Klasse, in dem die Elemente definiert sind und für die Berechnung zur Verfügung stehen.
    Die Elementklasse wird zusammen mit der Mesh-Klasse in eine FE-Klasse eingebunden.
    Hier weren nur dreieckige Elemente betrachtet.
    '''

    def __init__(self, E_modul=210E9, poisson_ratio=0.3):
        self.lame_mu = E_modul / (2*(1+poisson_ratio))
        self.lame_lambda = poisson_ratio*E_modul/((1+poisson_ratio)*(1-2*poisson_ratio))
        self.C_SE = np.array([[self.lame_lambda + 2*self.lame_mu, self.lame_lambda, 0],
                         [self.lame_lambda , self.lame_lambda + 2*self.lame_mu, 0],
                         [0, 0, self.lame_mu]])
        pass

    def compute_tensors(self, x, X):
        x1, y1, x2, y2, x3, y3 = x
        X1, Y1, X2, Y2, X3, Y3 = X
        self.I = np.eye(2)
        # Darstellung der Verschiebung in Voigt-Notation;
        # Vermutlich ist das direkte Eingruppieren in ein Array schneller...
        # u_voigt = x - X
        # u = u_voigt.reshape((-1,2)).T
        # Hier scheint jedoch die Verschiebungsdarstellung in Matrix-Notation besser zu sein
        self.u = np.array([[x1 - X1, y1 - Y1], [x2 - X2, y2 - Y2], [x3 - X3, y3 - Y3]])
        # compute B_tilde-matrix:
        # A = 0.5*((x3-x2)*(y1-y2) - (x1-x2)*(y3-y2))
        # just some stuff for updated Lagrangian
        # B_tilde = 1/(2*A)*np.array([[y2-y3, x3-x2], [y3-y1, x1-x3], [y1-y2, x2-x1]])
        # B = 1/(2*A)*np.array([  [y2-y3, 0, y3-y1, 0, y1-y2, 0],
        #                         [0, x3-x2, 0, x1-x3, 0, x2-x1],
        #                         [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]])
        # Total Lagrangian values:
        self.A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        self.B0_tilde = 1/(2*self.A0)*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]]).T
        self.H = self.B0_tilde.dot(self.u)
        self.F = self.H + self.I
        self.E = 1/2*(self.H + self.H.T + self.H.T.dot(self.H))
        self.S = self.lame_lambda*np.trace(self.E)*np.eye(2) + 2*self.lame_mu*self.E
        pass

    def f_int(self, x, X):
        '''
        Beschreibung des Koordinatenvektors über x = [x1, y1, x2, y2, x3, y3]
        '''
        self.compute_tensors(x, X)
        self.P = self.S.dot(self.F.T)
        f_int = self.B0_tilde.T.dot(self.P)*self.A0
        return f_int.reshape(-1)


    def k_int(self, x, X):
        '''
        Beschreibung des Koordinatenvektors über x = [x1, y1, x2, y2, x3, y3]

        '''
        self.compute_tensors(x, X)
        # Tangentiale Steifigkeitsmatrix
        self.K_geo_small = self.B0_tilde.T.dot(self.S.dot(self.B0_tilde))*self.A0
        self.K_geo = np.kron(self.K_geo_small, self.I)
        #
        self.B0 = np.zeros((3, 6))
        for i in range(3):
            self.B0[:,2*i:2*i+2] = np.array([[self.B0_tilde[0,i], 0], [0, self.B0_tilde[1,i]], [self.B0_tilde[1,i], self.B0_tilde[0,i]]]).dot(self.F.T)
        self.K_mat = self.B0.T.dot(self.C_SE.dot(self.B0))*self.A0
        return self.K_mat + self.K_geo

    def m_int(self, x, X):
        pass



class ElementSchale():
    '''
    Schalenelement
    - Freiheitsgrad-Definition muss noch überprüft werden
    - unklar: Handling von Rotationen
    '''

    def __init(self):
        pass

    def f_int(self, x, X):
        pass

    def k_int(self, x, X):
        pass

    def m_int(self, x, X):
        pass




def jacobian(func, vec, X):
    ndof = vec.shape[0]
    jacobian = np.zeros((ndof, ndof))
    h = np.sqrt(np.finfo(float).eps)
    f = func(vec, X)
    for i in range(ndof):
        vec_tmp = vec.copy()
        vec_tmp[i] += h
        f_tmp = func(vec_tmp, X)
        print(vec_tmp - vec)
        jacobian[:,i] = (f_tmp - f) / h
    return jacobian

#
## Definition von h
#h = np.sqrt(np.finfo(float).eps)
#
#
## my_element = ElementPlanar(E_modul = 60, poisson_ratio=1/4)
#my_element = ElementPlanar(E_modul = 57.6, poisson_ratio=1/5)
#X = np.array([0, 0, 3, 1, 2, 2.])
#x = np.array([0, 0, 3.1, 1, 2, 2.])
#
#K = my_element.k_int(X, X)
#K_finite_differenzen = jacobian(my_element.f_int, X, X)
#
#
#x = X.copy()
#x[1] += h
#f_nl = my_element.f_int(x, X)
#

#
#y_stretch = 1.
#x = np.array([y_stretch*1, 0, 0, 1, 0, 0], dtype=float)
#X = np.array([y_stretch*1, 0, 0, 1, 0, 0], dtype=float)
#
#
#
#X = x_new
#
#x_tmp = x.copy()
#x_tmp[1] += h*100
#
#f = my_element.f_int(x_tmp, X)
#K = my_element.k_int(x_new, x_new)
#K_finite_diff =
#
#
#
## Test
## my_element.f_int(x_new, x)
## my_element.k_int(x_new, x)
#
#
#S_new = my_element.C_SE.dot(my_element.B0.dot(my_element.u.reshape(-1)))
#E_new = my_element.B0.dot(my_element.u.reshape(-1))

#%%

