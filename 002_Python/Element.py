# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:13:52 2015

@author: johannesr
"""

import numpy as np 

class ElementPlanar():
    '''
    Das ist die Klasse, in dem die Elemente definiert sind und für die Berechnung zur Verfügung stehen.
    Die Elementklasse wird zusammen mit der Mesh-Klasse in eine FE-Klasse eingebunden.
    Die FE-Klasse ist die Basisklasse, die Mesh, Element, Integrator etc. enthält

    '''

    def __init__(self):
        pass

    def f_int(self, x, X):
        '''
        Beschreibung des Koordinatenvektors über x = [x1, y1, x2, y2, x3, y3]

        '''
        x1, y1, x2, y2, x3, y3 = x
        X1, Y1, X2, Y2, X3, Y3 = X
        u = x - X
        # compute B_tilde-matrix:
        A = 0.5*((x3-x2)*(y1-y2) - (x1-x2)*(y3-y2))
        B_tilde = 1/(2*A)*np.array([[y2-y3, x3-x2], [y3-y1, x1-x3], [y1-y2, x2-x1]])
        B = 1/(2*A)*np.array([  [y2-y3, 0, y3-y1, 0, y1-y2, 0],
                                [0, x3-x2, 0, x1-x3, 0, x2-x1],
                                [x3-x2, y2-y3, x1-x3, y3-y1, x2-x1, y1-y2]])
        # Total Lagrangian values:
        A0 = 0.5*((X3-X2)*(Y1-Y2) - (X1-X2)*(Y3-Y2))
        B0_tilde = 1/(2*A0)*np.array([[Y2-Y3, X3-X2], [Y3-Y1, X1-X3], [Y1-Y2, X2-X1]])
        H = B0_tilde.dot(u)
        pass

    def k_int(self, x, X):
        pass


class ElementSchale():
    '''

    '''




x = np.array([1, 0, 0, 1, 0, 0])